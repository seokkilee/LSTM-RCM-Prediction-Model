import os
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


def get_inducing_points(lstm_model, train_dataset, num_inducing_points, output_dim, device):
    """
    train_dataset는 항상 CPU 텐서로 구성된 Dataset이어야 하며,
    여기서 배치 단위로만 GPU로 올려 사용한다.
    """
    print("Collecting latent representations for KMeans...")

    pin_memory = (device.type == "cuda")
    loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=pin_memory,
    )

    latent_all = []
    with torch.no_grad():
        for bx, _ in loader:
            if device.type == "cuda":
                bx = bx.to(device, non_blocking=True)
            else:
                bx = bx.to(device)
            latent = lstm_model(bx)
            latent_all.append(latent.cpu().numpy())

    latent_all = np.vstack(latent_all)
    print(f"Latent shape: {latent_all.shape}")

    kmeans = KMeans(n_clusters=num_inducing_points, random_state=SEED).fit(latent_all)
    inducing_points = torch.FloatTensor(kmeans.cluster_centers_).to(device)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    np.save(os.path.join(ARTIFACT_DIR, "inducing_points.npy"), kmeans.cluster_centers_)
    print("Inducing points saved:", os.path.join(ARTIFACT_DIR, "inducing_points.npy"))

    return inducing_points


def train_model(
    trainX_tensor,
    trainY_tensor,
    input_dim,
    hidden_dim,
    output_dim,
    num_layers,
    num_inducing_points,
    lambda_uncertainty,
    alpha_elbo_max,
    device,
    epochs,
    warmup_epochs,
    val_ratio=0.1,
    save_model_path=os.path.join(ARTIFACT_DIR, 'tb3_lstm_svgp_model.pth'),
    return_history=False,
):
    """
    입력 trainX_tensor, trainY_tensor는 CPU/GPU 어느 쪽이든 받아서
    내부에서 CPU로 강제 후 DataLoader를 구성하고,
    배치 단위로만 device로 올려서 계산한다.
    """

    # 항상 CPU 텐서 기준으로 DataLoader를 구성
    trainX_tensor = trainX_tensor.cpu()
    trainY_tensor = trainY_tensor.cpu()

    # ========== Validation split ==========
    total_size = trainX_tensor.shape[0]
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    trainX_sub, valX_sub = torch.split(trainX_tensor, [train_size, val_size])
    trainY_sub, valY_sub = torch.split(trainY_tensor, [train_size, val_size])

    train_dataset = TensorDataset(trainX_sub, trainY_sub)
    val_dataset = TensorDataset(valX_sub, valY_sub)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    # =====================================

    lstm_model = LSTMNet(input_dim, hidden_dim, output_dim, num_layers).to(device)
    inducing_points = get_inducing_points(
        lstm_model, train_dataset, num_inducing_points, output_dim, device
    )
    rcm_model = SVGPModel(inducing_points, num_tasks=output_dim, output_dim=output_dim).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=output_dim
    ).to(device)

    device = next(lstm_model.parameters()).device

    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)
    rcm_optimizer = torch.optim.Adam(
        [
            {"params": rcm_model.parameters()},
            {"params": likelihood.parameters()},
        ],
        lr=learning_rate,
    )
    criterion = torch.nn.MSELoss()
    mll = VariationalELBO(likelihood, rcm_model, num_data=trainX_sub.size(0)).to(device)

    # === LR Scheduler ===
    if lr_scheduler_type == "plateau":
        lstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            lstm_optimizer,
            mode='min',
            patience=lr_patience,
            factor=lr_factor,
            min_lr=min_lr,
        )
        rcm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            rcm_optimizer,
            mode='min',
            patience=lr_patience,
            factor=lr_factor,
            min_lr=min_lr,
        )
    elif lr_scheduler_type == "step":
        lstm_scheduler = optim.lr_scheduler.StepLR(
            lstm_optimizer, step_size=lr_step_size, gamma=lr_factor
        )
        rcm_scheduler = optim.lr_scheduler.StepLR(
            rcm_optimizer, step_size=lr_step_size, gamma=lr_factor
        )
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
            if device.type == "cuda":
                batch_x = batch_x.to(device, non_blocking=True)
                batch_y = batch_y.to(device, non_blocking=True)
            else:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

            lstm_optimizer.zero_grad()
            rcm_optimizer.zero_grad()

            latent_representation = lstm_model(batch_x)

            with gpytorch.settings.num_likelihood_samples(10):
                rcm_output = rcm_model(latent_representation)
                rcm_correction = likelihood(rcm_output)

            rcm_mean = rcm_correction.mean
            rcm_std = rcm_correction.variance.sqrt()

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
        avg_elbo_loss = total_elbo_loss / len(train_loader)
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
                if device.type == "cuda":
                    val_x = val_x.to(device, non_blocking=True)
                    val_y = val_y.to(device, non_blocking=True)
                else:
                    val_x = val_x.to(device)
                    val_y = val_y.to(device)

                latent_pred = lstm_model(val_x)

                # residual = 실제값 - LSTM 예측값, L2 norm
                res = (val_y - latent_pred).cpu().numpy()  # (batch, output_dim)
                l2_res = np.linalg.norm(res, axis=1)       # (batch,)
                residuals_epoch.append(l2_res)

                # validation loss (RCM 보정 포함)
                rcm_output = rcm_model(latent_pred)
                rcm_correction = likelihood(rcm_output)
                rcm_mean = rcm_correction.mean
                rcm_std = rcm_correction.variance.sqrt()

                weight = torch.exp(-lambda_uncertainty * rcm_std)
                final_pred = latent_pred + (weight * rcm_mean)

                val_loss = criterion(final_pred, val_y)
                val_final_loss += val_loss.item() * val_x.size(0)
                n_val += val_x.size(0)

        avg_val_final_loss = val_final_loss / n_val
        val_losses.append(avg_val_final_loss)

        residuals_epoch = np.concatenate(residuals_epoch)  # (n_val, )
        residuals_list.append(residuals_epoch)

        # === epoch별 전체 latent representation 추출 및 저장 ===
        with torch.no_grad():
            all_latent = []
            epoch_loader = DataLoader(
                train_dataset,
                batch_size=256,
                shuffle=False,
                pin_memory=pin_memory,
            )
            for bx, _ in epoch_loader:
                if device.type == "cuda":
                    bx = bx.to(device, non_blocking=True)
                else:
                    bx = bx.to(device)
                latent = lstm_model(bx)
                all_latent.append(latent.cpu().numpy())
            all_latent = np.vstack(all_latent)  # (n_samples, output_dim)
            latent_list.append(all_latent)
        # =====================================================

        # === inducing points 변화 추적 ===
        ipt = rcm_model.variational_strategy.base_variational_strategy.inducing_points
        inducing_points_list.append(ipt.cpu().detach().numpy())

        # === LR decay: val loss 사용 ===
        loss_for_scheduler = avg_val_final_loss
        if lr_scheduler_type == "plateau" and lstm_scheduler is not None:
            lstm_scheduler.step(loss_for_scheduler)
            rcm_scheduler.step(loss_for_scheduler)
        elif lr_scheduler_type == "step" and lstm_scheduler is not None:
            lstm_scheduler.step()
            rcm_scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs}, "
            f"Train Final Loss: {avg_final_loss:.6f}, "
            f"Val Final Loss: {avg_val_final_loss:.6f}, "
            f"ELBO Loss: {avg_elbo_loss:.6f}, "
            f"Total Loss: {avg_total_loss:.6f}, "
            f"Alpha(ELBO): {effective_alpha:.6f}, "
            f"LSTM LR: {lstm_optimizer.param_groups[0]['lr']:.6e}, "
            f"RCM LR: {rcm_optimizer.param_groups[0]['lr']:.6e}"
        )

    # ======== 변화 추적 시각화/저장 ========
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    plot_loss(epochs_list, total_losses, final_losses, elbo_losses)
    plot_inducing_points(inducing_points_list, epochs, num_inducing_points)

    np.save(os.path.join(ARTIFACT_DIR, "inducing_points_trace.npy"), np.array(inducing_points_list))
    np.save(os.path.join(ARTIFACT_DIR, "latent_trace.npy"), np.array(latent_list))
    np.save(os.path.join(ARTIFACT_DIR, "residuals.npy"), np.array(residuals_list))
    print(f"[INFO] residuals.npy saved. Shape = {np.array(residuals_list).shape}")

    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
    torch.save(
        {
            "lstm_state_dict": lstm_model.state_dict(),
            "rcm_state_dict": rcm_model.state_dict(),
            "likelihood_state_dict": likelihood.state_dict(),
            "lstm_optimizer": lstm_optimizer.state_dict(),
            "rcm_optimizer": rcm_optimizer.state_dict(),
        },
        save_model_path,
    )
    print(f"[INFO] Model saved to {save_model_path}")

    history = {
        "epochs": epochs_list,
        "final_losses": final_losses,
        "elbo_losses": elbo_losses,
        "total_losses": total_losses,
        "val_losses": val_losses,
    }

    if return_history:
        return lstm_model, rcm_model, likelihood, lstm_optimizer, rcm_optimizer, history
    else:
        return lstm_model, rcm_model, likelihood, lstm_optimizer, rcm_optimizer
