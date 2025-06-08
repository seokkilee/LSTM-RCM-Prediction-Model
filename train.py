import torch
import gpytorch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.cluster import KMeans
from gpytorch.mlls import VariationalELBO
from config import *
from models import LSTMNet, SVGPModel
from utils import plot_loss, plot_inducing_points
import torch.optim as optim
import os

def get_inducing_points(lstm_model, train_dataset, num_inducing_points, output_dim, device):
    print("Collecting latent representations for KMeans...")
    lstm_model.eval()
    with torch.no_grad():
        latent_all = []
        for bx, _ in DataLoader(train_dataset, batch_size=256):
            latent = lstm_model(bx.to(device))
            latent_all.append(latent.cpu().numpy())
        latent_all = np.vstack(latent_all)
    lstm_model.train()
    print(f"Latent shape: {latent_all.shape}")
    kmeans = KMeans(n_clusters=num_inducing_points, random_state=SEED).fit(latent_all)
    inducing_points = torch.FloatTensor(kmeans.cluster_centers_).to(device)
    np.save(ARTIFACT_DIR + "inducing_points.npy", kmeans.cluster_centers_)
    print("Inducing points saved:", ARTIFACT_DIR + "inducing_points.npy")
    return inducing_points

def train_model(
    trainX_tensor, trainY_tensor, input_dim, hidden_dim, output_dim, num_layers, num_inducing_points,
    lambda_uncertainty, alpha_elbo_max, device, epochs, warmup_epochs,
    val_ratio=0.1,
    save_model_path=ARTIFACT_DIR + 'tb3_lstm_svgp_model.pth'
):
    # ========== Validation split ==========
    total_size = trainX_tensor.shape[0]
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    trainX_sub, valX_sub = torch.split(trainX_tensor, [train_size, val_size])
    trainY_sub, valY_sub = torch.split(trainY_tensor, [train_size, val_size])

    train_dataset = TensorDataset(trainX_sub, trainY_sub)
    val_dataset = TensorDataset(valX_sub, valY_sub)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # =====================================

    lstm_model = LSTMNet(input_dim, hidden_dim, output_dim, num_layers).to(device)
    inducing_points = get_inducing_points(lstm_model, train_dataset, num_inducing_points, output_dim, device)
    rcm_model = SVGPModel(inducing_points, num_tasks=output_dim, output_dim=output_dim).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=output_dim).to(device)
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    rcm_optimizer = torch.optim.Adam([
        {'params': rcm_model.parameters()},
        {'params': likelihood.parameters()}
    ], lr=learning_rate)
    criterion = torch.nn.MSELoss()
    mll = VariationalELBO(likelihood, rcm_model, num_data=trainX_sub.size(0)).to(device)

    # === LR Scheduler ===
    if lr_scheduler_type == "plateau":
        lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            lstm_optimizer, mode='min', patience=lr_patience, factor=lr_factor, min_lr=min_lr, verbose=True
        )
        rcm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            rcm_optimizer, mode='min', patience=lr_patience, factor=lr_factor, min_lr=min_lr, verbose=True
        )
    elif lr_scheduler_type == "step":
        lstm_scheduler = optim.lr_scheduler.StepLR(lstm_optimizer, step_size=lr_step_size, gamma=lr_factor)
        rcm_scheduler = optim.lr_scheduler.StepLR(rcm_optimizer, step_size=lr_step_size, gamma=lr_factor)
    else:
        lstm_scheduler = None
        rcm_scheduler = None

    final_losses, elbo_losses, total_losses, val_losses, epochs_list = [], [], [], [], []
    inducing_points_list = []
    latent_list = []
    residuals_list = []

    for epoch in range(epochs):
        lstm_model.train()
        rcm_model.train()
        likelihood.train()
        total_final_loss = 0.0
        total_elbo_loss = 0.0
        total_loss_value = 0.0

        # ======== WARMUP/ADAPTIVE LOSS ========
        if epoch < warmup_epochs:
            effective_alpha = alpha_elbo_max * (epoch + 1) / warmup_epochs
        else:
            effective_alpha = alpha_elbo_max
        # =======================================

        # ---- Training Loop ----
        for batch_x, batch_y in train_loader:
            lstm_optimizer.zero_grad()
            rcm_optimizer.zero_grad()
            latent_representation = lstm_model(batch_x)
            with gpytorch.settings.num_likelihood_samples(10):
                rcm_output = rcm_model(latent_representation)
                rcm_correction = likelihood(rcm_output)
            rcm_mean = rcm_correction.mean
            rcm_std  = rcm_correction.variance.sqrt()
            weight = torch.exp(-lambda_uncertainty * rcm_std)
            final_pred = latent_representation + (weight * rcm_mean)
            final_loss = criterion(final_pred, batch_y)
            elbo_loss = -mll(rcm_output, batch_y)
            loss = final_loss + (effective_alpha * elbo_loss)
            loss.backward()
            lstm_optimizer.step()
            rcm_optimizer.step()
            total_final_loss += final_loss.item()
            total_elbo_loss += elbo_loss.item()
            total_loss_value += loss.item()

        avg_final_loss = total_final_loss / len(train_loader)
        avg_elbo_loss  = total_elbo_loss / len(train_loader)
        avg_total_loss = total_loss_value / len(train_loader)
        final_losses.append(avg_final_loss)
        elbo_losses.append(avg_elbo_loss)
        total_losses.append(avg_total_loss)
        epochs_list.append(epoch + 1)

        # === Validation loss 및 LSTM residual 계산 ===
        lstm_model.eval()
        rcm_model.eval()
        likelihood.eval()
        val_final_loss = 0.0
        n_val = 0
        residuals_epoch = []
        with torch.no_grad():
            for val_x, val_y in val_loader:
                latent_pred = lstm_model(val_x)
                # residual = 실제값 - LSTM 예측값, L2 norm
                res = (val_y - latent_pred).cpu().numpy()  # (batch, output_dim)
                l2_res = np.linalg.norm(res, axis=1)       # (batch,)
                residuals_epoch.append(l2_res)
                # validation loss(RCM 보정 포함, 기존대로)
                rcm_output = rcm_model(latent_pred)
                rcm_correction = likelihood(rcm_output)
                rcm_mean = rcm_correction.mean
                rcm_std  = rcm_correction.variance.sqrt()
                weight = torch.exp(-lambda_uncertainty * rcm_std)
                final_pred = latent_pred + (weight * rcm_mean)
                val_loss = criterion(final_pred, val_y)
                val_final_loss += val_loss.item() * val_x.size(0)
                n_val += val_x.size(0)
        avg_val_final_loss = val_final_loss / n_val
        val_losses.append(avg_val_final_loss)
        # 모든 residual 합치기
        residuals_epoch = np.concatenate(residuals_epoch)  # (n_val, )
        residuals_list.append(residuals_epoch)
        # ==========================

        # === epoch별 전체 latent representation 추출 및 저장 ===
        with torch.no_grad():
            all_latent = []
            for bx, _ in DataLoader(train_dataset, batch_size=256):
                latent = lstm_model(bx.to(device))
                all_latent.append(latent.cpu().numpy())
            all_latent = np.vstack(all_latent)  # (n_samples, output_dim)
            latent_list.append(all_latent)
        # =====================================================

        # === inducing points 변화 추적 ===
        ipt = rcm_model.variational_strategy.base_variational_strategy.inducing_points
        inducing_points_list.append(ipt.cpu().detach().numpy())

        # === LR decay: val loss 사용 ===
        loss_for_scheduler = avg_val_final_loss
        if lr_scheduler_type == "plateau":
            lstm_scheduler.step(loss_for_scheduler)
            rcm_scheduler.step(loss_for_scheduler)
        elif lr_scheduler_type == "step":
            lstm_scheduler.step()
            rcm_scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Final Loss: {avg_final_loss:.6f}, "
              f"Val Final Loss: {avg_val_final_loss:.6f}, "
              f"ELBO Loss: {avg_elbo_loss:.6f}, "
              f"Total Loss: {avg_total_loss:.6f}, "
              f"Alpha(ELBO): {effective_alpha:.6f}, "
              f"LSTM LR: {lstm_optimizer.param_groups[0]['lr']:.6e}, "
              f"RCM LR: {rcm_optimizer.param_groups[0]['lr']:.6e}")

    # ======== 변화 추적 시각화/저장 ========
    plot_loss(epochs_list, total_losses, final_losses, elbo_losses)
    plot_inducing_points(inducing_points_list, epochs, num_inducing_points)
    np.save(ARTIFACT_DIR + "inducing_points_trace.npy", np.array(inducing_points_list))
    np.save(ARTIFACT_DIR + "latent_trace.npy", np.array(latent_list))
    np.save(ARTIFACT_DIR + "residuals.npy", np.array(residuals_list))   # (epochs, n_val)
    print(f"[INFO] residuals.npy saved. Shape = {np.array(residuals_list).shape}")

    torch.save({
        'lstm_state_dict': lstm_model.state_dict(),
        'rcm_state_dict': rcm_model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'lstm_optimizer': lstm_optimizer.state_dict(),
        'rcm_optimizer': rcm_optimizer.state_dict(),
    }, save_model_path)
    print(f"[INFO] Model saved to {save_model_path}")

    return lstm_model, rcm_model, likelihood, lstm_optimizer, rcm_optimizer
