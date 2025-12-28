# eval.py (FULL REVISED)
import os
import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d

from torch.utils.data import DataLoader, TensorDataset
from config import RESULT_DIR, batch_size as train_batch_size
import matplotlib.pyplot as plt


def _inverse_transform_with_padding(arr_np: np.ndarray, scaler, power_tf):
    """
    arr_np: (N, D) in normalized space
    scaler/power_tf are fitted with n_feat features.
    Returns:
      arr_real: (N, D) transformed back to real unit, keeping original D
    """
    N, D = arr_np.shape
    n_feat = scaler.mean_.shape[0]

    if D == n_feat:
        temp = arr_np
    else:
        temp = np.zeros((N, n_feat), dtype=arr_np.dtype)
        temp[:, :min(D, n_feat)] = arr_np[:, :min(D, n_feat)]

    temp = scaler.inverse_transform(temp)
    temp = power_tf.inverse_transform(temp)
    return temp[:, :D]


def _std_rescale_with_padding(std_np: np.ndarray, scaler):
    """
    std_np: (N, D_std) in normalized std space
    scaler.scale_ has length n_feat
    Returns:
      std_real: (N, D_std) in real unit std, keeping original D_std
    """
    N, D = std_np.shape
    n_feat = scaler.mean_.shape[0]

    if D == n_feat:
        return std_np * scaler.scale_

    temp = np.zeros((N, n_feat), dtype=std_np.dtype)
    temp[:, :min(D, n_feat)] = std_np[:, :min(D, n_feat)]
    temp = temp * scaler.scale_
    return temp[:, :D]


def evaluate_and_save_logs(
    testX_tensor,
    testY_tensor,
    lstm_model,
    rcm_model,
    likelihood,
    output_dim,
    output_scaler,
    output_power_transformer,
    lambda_uncertainty=1.0,
    apply_gaussian_filter=False,
    sigma=5,
    device=None,
    batch_size_eval=None,
):
    """
    전체 테스트 시퀀스에 대해 예측/표준편차를 계산하고 .txt로 저장.
    - testX_tensor, testY_tensor: CPU 텐서 기준 (GPU로 들어와도 내부에서 CPU로 이동)
    - 내부에서 배치 단위로 device로 옮겨 사용
    """
    testX_tensor = testX_tensor.cpu()
    testY_tensor = testY_tensor.cpu()

    if device is None:
        device = next(lstm_model.parameters()).device
    if batch_size_eval is None:
        batch_size_eval = train_batch_size

    lstm_model.eval()
    rcm_model.eval()
    likelihood.eval()

    test_dataset = TensorDataset(testX_tensor, testY_tensor)
    pin_memory = (device.type == "cuda")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        pin_memory=pin_memory,
    )

    all_pred, all_target, all_std = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            latent_test = lstm_model(x)
            rcm_output_test = rcm_model(latent_test)
            rcm_correction_test = likelihood(rcm_output_test)

            rcm_mean = rcm_correction_test.mean
            rcm_std = rcm_correction_test.variance.sqrt()

            weight = torch.exp(-lambda_uncertainty * rcm_std)
            final_pred_test = latent_test + (weight * rcm_mean)

            all_pred.append(final_pred_test.cpu().numpy())
            all_target.append(y.cpu().numpy())
            all_std.append(rcm_std.cpu().numpy())

    final_pred_np = np.vstack(all_pred)
    testY_np = np.vstack(all_target)
    pred_std_np = np.vstack(all_std)

    # 역변환(패딩/슬라이싱 방어)
    final_pred_np = _inverse_transform_with_padding(final_pred_np, output_scaler, output_power_transformer)

    # 타깃도 normalized라고 가정하고 동일 역변환(차원 불일치 방어)
    testY_np = _inverse_transform_with_padding(testY_np, output_scaler, output_power_transformer)

    # 표준편차 real unit 변환(패딩/슬라이싱 방어)
    pred_std_np = _std_rescale_with_padding(pred_std_np, output_scaler)

    if apply_gaussian_filter:
        final_pred_np = gaussian_filter1d(final_pred_np, sigma=sigma, axis=0)

    output_names = ["pos_x", "pos_y", "heading", "velocity"]
    D_save = min(output_dim, final_pred_np.shape[1], testY_np.shape[1], len(output_names), pred_std_np.shape[1])

    os.makedirs(RESULT_DIR, exist_ok=True)
    for d in range(D_save):
        np.savetxt(os.path.join(RESULT_DIR, f"true_{output_names[d]}.txt"), testY_np[:, d], fmt="%.6f")
        np.savetxt(os.path.join(RESULT_DIR, f"lstm_rcm_{output_names[d]}.txt"), final_pred_np[:, d], fmt="%.6f")
        np.savetxt(os.path.join(RESULT_DIR, f"lstm_rcm_std_{output_names[d]}.txt"), pred_std_np[:, d], fmt="%.6f")

    print("[INFO] LSTM-RCM 예측 결과와 표준편차를 .txt로 저장 완료.")


def evaluate_model_full(
    lstm_model,
    rcm_model,
    likelihood,
    testX_tensor,
    testY_tensor,
    output_scaler,
    output_power_transformer,
    lambda_uncertainty=1.0,
    device=None,
    batch_size_eval=None,
    result_dir=None,
    result_prefix="",
    figure_path=None,
):
    """
    민감도 분석용 평가 함수.
    - DataLoader로 test set 전체를 평가
    - 출력: rmse_dict, pred_np, true_np

    또한 .txt 저장을 항상 동작하게 함:
    - result_dir이 None이면 config.RESULT_DIR 사용
    - result_prefix가 비어있으면 lambda 값 기반으로 자동 prefix 생성
    """
    testX_tensor = testX_tensor.cpu()
    testY_tensor = testY_tensor.cpu()

    if device is None:
        device = next(lstm_model.parameters()).device
    if batch_size_eval is None:
        batch_size_eval = train_batch_size

    lstm_model.eval()
    rcm_model.eval()
    likelihood.eval()

    test_dataset = TensorDataset(testX_tensor, testY_tensor)
    pin_memory = (device.type == "cuda")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_eval,
        shuffle=False,
        pin_memory=pin_memory,
    )

    all_pred, all_target, all_std = [], [], []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=(device.type == "cuda"))
            y = y.to(device, non_blocking=(device.type == "cuda"))

            latent = lstm_model(x)
            rcm_output = rcm_model(latent)
            rcm_corr = likelihood(rcm_output)

            rcm_mean = rcm_corr.mean
            rcm_std = rcm_corr.variance.sqrt()

            weight = torch.exp(-lambda_uncertainty * rcm_std)
            final_pred = latent + (weight * rcm_mean)

            all_pred.append(final_pred.cpu().numpy())
            all_target.append(y.cpu().numpy())
            all_std.append(rcm_std.cpu().numpy())

    pred_np_norm = np.vstack(all_pred)
    target_np_norm = np.vstack(all_target)
    std_np_norm = np.vstack(all_std)

    # ----- 역변환 (패딩/슬라이싱 방어) -----
    pred_np = _inverse_transform_with_padding(pred_np_norm, output_scaler, output_power_transformer)
    true_np = _inverse_transform_with_padding(target_np_norm, output_scaler, output_power_transformer)

    # 표준편차 스케일 조정 (패딩/슬라이싱 방어)
    std_np = _std_rescale_with_padding(std_np_norm, output_scaler)

    # ----- RMSE 계산 (가능한 차원까지만) -----
    D = min(pred_np.shape[1], true_np.shape[1])
    if D < 1:
        raise ValueError(f"pred/true dimension invalid: pred={pred_np.shape}, true={true_np.shape}")

    # 4개 state가 없을 수도 있으니 안전 계산
    def _rmse_col(i):
        if D > i:
            return float(np.sqrt(np.mean((pred_np[:, i] - true_np[:, i]) ** 2)))
        return float("nan")

    rmse_x = _rmse_col(0)
    rmse_y = _rmse_col(1)
    rmse_heading = _rmse_col(2)
    rmse_vel = _rmse_col(3)

    overall_rmse = float(np.sqrt(np.mean((pred_np[:, :D] - true_np[:, :D]) ** 2)))

    ey_rms = float("nan")
    ey_max = float("nan")
    psi_rms = float("nan")
    if D > 1:
        ey = pred_np[:, 1] - true_np[:, 1]
        ey_rms = float(np.sqrt(np.mean(ey ** 2)))
        ey_max = float(np.max(np.abs(ey)))
    if D > 2:
        psi = pred_np[:, 2] - true_np[:, 2]
        psi_rms = float(np.sqrt(np.mean(psi ** 2)))

    rmse_dict = {
        "x": rmse_x,
        "y": rmse_y,
        "heading": rmse_heading,
        "velocity": rmse_vel,
        "ey_rms": ey_rms,
        "ey_max": ey_max,
        "psi_rms": psi_rms,
        "overall_rmse": overall_rmse,
    }

    # ----- .txt 저장: sensitivity 실행 시 항상 저장되도록 -----
    if result_dir is None:
        result_dir = RESULT_DIR
    if result_prefix is None:
        result_prefix = ""
    if result_prefix == "":
        # 파일명 안전하게(소수점 포함)
        result_prefix = f"lambda_{lambda_uncertainty}".replace(".", "p")

    os.makedirs(result_dir, exist_ok=True)

    output_names = ["pos_x", "pos_y", "heading", "velocity"]
    D_save = min(4, pred_np.shape[1], true_np.shape[1], std_np.shape[1], len(output_names))

    for d in range(D_save):
        np.savetxt(
            os.path.join(result_dir, f"{result_prefix}_true_{output_names[d]}.txt"),
            true_np[:, d],
            fmt="%.6f",
        )
        np.savetxt(
            os.path.join(result_dir, f"{result_prefix}_pred_{output_names[d]}.txt"),
            pred_np[:, d],
            fmt="%.6f",
        )
        np.savetxt(
            os.path.join(result_dir, f"{result_prefix}_std_{output_names[d]}.txt"),
            std_np[:, d],
            fmt="%.6f",
        )

    # rmse 로그도 같이 저장(선택 but useful)
    np.savetxt(
        os.path.join(result_dir, f"{result_prefix}_rmse_summary.txt"),
        np.array([
            rmse_x, rmse_y, rmse_heading, rmse_vel,
            ey_rms, ey_max, psi_rms, overall_rmse
        ], dtype=float),
        fmt="%.10f",
        header="rmse_x rmse_y rmse_heading rmse_velocity ey_rms ey_max psi_rms overall_rmse",
        comments="",
    )

    # ----- Plot (선택) -----
    if figure_path is not None:
        t = np.arange(len(true_np))
        labels = ["Position X", "Position Y", "Heading", "Velocity"]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        ax = axes.flatten()
        for i in range(min(4, D_save)):
            ax[i].plot(t, true_np[:, i], label="Ground truth")
            ax[i].plot(t, pred_np[:, i], label="Prediction")
            ax[i].set_title(labels[i])
            ax[i].grid(True)
            ax[i].legend()
        plt.tight_layout()
        os.makedirs(os.path.dirname(figure_path), exist_ok=True)
        plt.savefig(figure_path, dpi=400)
        plt.close()

    return rmse_dict, pred_np, true_np
