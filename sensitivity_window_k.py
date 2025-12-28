# sensitivity_window_k.py
#
# H2: Window Length k Sensitivity
# - Train with k_train in {1, 5, 10, 15, 20}
# - Test always with k_test = 20 (1.0 s history)
# - Other hyperparameters are fixed to baseline:
#     LSTM layers = num_layers (config)
#     SVGP inducing points = num_inducing_points
#     lambda_uncertainty = baseline
# - Metrics:
#     Prediction RMSE: x, y, heading, velocity, overall
#     Closed-loop related: ey_RMS, max|ey|, psi_RMS
# - Outputs:
#     Per (k_train, run): logs (.txt) + Fig.8-style plots (.png)
#     Per k_train: mean/std saved as .npy + .txt
#     Summary plots over k_train

import os
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt

from config import *
from dataset import load_and_preprocess_data, build_dataset
from train import train_model
from eval import evaluate_model_full
from utils import ensure_dir

# -------------------------------
# 메모리 정리
# -------------------------------
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------------------
# 실험 하이퍼파라미터
# -------------------------------
N_runs = 5                               # 각 k_train 값별 반복 횟수 (평균±표준편차)
k_train_list = [1, 5, 10, 15, 20]        # 훈련 시 window length 후보
k_test = 20                              # 테스트 시 항상 사용하는 window length (1.0 s history)

# 모델이 올라갈 장치 (GPU가 있으면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 디렉토리 생성
    ensure_dir(FIGURE_DIR)
    ensure_dir(RESULT_DIR)
    ensure_dir(ARTIFACT_DIR)

    # ---------------------------
    # 데이터 로딩 및 전처리 (k와 무관하게 1회 수행)
    # ---------------------------
    print("[INFO] Loading and preprocessing dataset...")
    input_data, output_data, input_scaler, input_pt, output_scaler, output_pt = \
        load_and_preprocess_data("train_data/tb3_train_set_25.11.23.txt", input_dim, output_dim)

    time_series = np.hstack((input_data, output_data))
    total_size = len(time_series)
    train_size = int(0.9 * total_size)

    # ---------------------------
    # 테스트 셋은 항상 k_test = 20 으로 고정
    # (1.0 s history window를 사용하여 delay-bound를 커버하는 상황 가정)
    # ---------------------------
    print(f"[INFO] Building test dataset with k_test = {k_test} (history = {k_test * 50} ms)")
    testX, testY = build_dataset(time_series[train_size:], k_test, input_dim, output_dim)

    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)

    # ---------------------------
    # 결과 저장용 dict
    # ---------------------------
    results = {}

    # ---------------------------
    # Window length k_train 민감도 실험
    # ---------------------------
    for k_train in k_train_list:
        print("=" * 80)
        print(f"[INFO] Window length sensitivity: k_train = {k_train} (history = {k_train * 50} ms), "
              f"test with k_test = {k_test} (1.0 s)")
        print("=" * 80)

        # k_train별 training dataset 구성
        trainX, trainY = build_dataset(time_series[:train_size], k_train, input_dim, output_dim)

        # CPU 텐서로 유지, train/eval 내부에서 device로 옮김
        trainX_tensor = torch.FloatTensor(trainX)
        trainY_tensor = torch.FloatTensor(trainY)

        # k_train별 결과/그림 폴더
        k_result_dir = os.path.join(RESULT_DIR, f"window_ktrain_{k_train}")
        k_fig_dir    = os.path.join(FIGURE_DIR, f"window_ktrain_{k_train}")
        ensure_dir(k_result_dir)
        ensure_dir(k_fig_dir)

        ey_rms_list        = []
        ey_max_list        = []
        psi_rms_list       = []

        rmse_x_list        = []
        rmse_y_list        = []
        rmse_heading_list  = []
        rmse_vel_list      = []
        overall_rmse_list  = []

        for run in range(N_runs):
            print(f"\n[RUN {run+1}/{N_runs}] Training with k_train = {k_train}...")

            save_path = os.path.join(
                ARTIFACT_DIR, f"window_ktrain_{k_train}_run{run+1}.pth"
            )

            # 모델/학습은 device를 사용, 데이터는 CPU 텐서로 전달
            lstm_model, rcm_model, likelihood, lstm_opt, rcm_opt, history = train_model(
                trainX_tensor, trainY_tensor,
                input_dim=input_dim,
                hidden_dim=hidden_dim,                   # baseline: 128
                output_dim=output_dim,
                num_layers=num_layers,                   # baseline: 3 (고정)
                num_inducing_points=num_inducing_points, # baseline: 30 (고정)
                lambda_uncertainty=lambda_uncertainty,   # baseline λ (고정)
                alpha_elbo_max=alpha_elbo_max,
                device=device,                           # 모델/연산 장치
                epochs=iterations,
                warmup_epochs=warmup_epochs,
                val_ratio=0.1,
                save_model_path=save_path,
                return_history=True,
            )

            prefix   = f"ktrain{k_train}_run{run+1}"
            fig_path = os.path.join(k_fig_dir, f"{prefix}_prediction.png")

            # --- 테스트는 항상 k_test = 20 window 기반 데이터(testX_tensor, testY_tensor)를 사용 ---
            rmse_dict, pred_np, true_np = evaluate_model_full(
                lstm_model, rcm_model, likelihood,
                testX_tensor, testY_tensor,
                output_scaler, output_pt,
                lambda_uncertainty=lambda_uncertainty,
                device=device,
                batch_size_eval=batch_size,
                result_dir=k_result_dir,
                result_prefix=prefix,
                figure_path=fig_path,
            )

            # state-wise prediction RMSE
            rmse_x_list.append(rmse_dict["x"])
            rmse_y_list.append(rmse_dict["y"])
            rmse_heading_list.append(rmse_dict["heading"])
            rmse_vel_list.append(rmse_dict["velocity"])
            overall_rmse_list.append(rmse_dict["overall_rmse"])

            # lateral/heading 지표
            ey_rms_list.append(rmse_dict["ey_rms"])
            ey_max_list.append(rmse_dict["ey_max"])
            psi_rms_list.append(rmse_dict["psi_rms"])

            print(f"[RESULT] k_train={k_train}, run={run+1} | "
                  f"ey_RMS={rmse_dict['ey_rms']:.4f}, "
                  f"ey_max={rmse_dict['ey_max']:.4f}, "
                  f"psi_RMS={rmse_dict['psi_rms']:.4f}, "
                  f"RMSE_x={rmse_dict['x']:.4f}, "
                  f"RMSE_y={rmse_dict['y']:.4f}, "
                  f"RMSE_heading={rmse_dict['heading']:.4f}, "
                  f"RMSE_vel={rmse_dict['velocity']:.4f}, "
                  f"RMSE_overall={rmse_dict['overall_rmse']:.4f}"
            )

        # k_train별 통계량 저장
        ey_rms_arr        = np.array(ey_rms_list)
        ey_max_arr        = np.array(ey_max_list)
        psi_rms_arr       = np.array(psi_rms_list)
        rmse_x_arr        = np.array(rmse_x_list)
        rmse_y_arr        = np.array(rmse_y_list)
        rmse_heading_arr  = np.array(rmse_heading_list)
        rmse_vel_arr      = np.array(rmse_vel_list)
        overall_rmse_arr  = np.array(overall_rmse_list)

        results[k_train] = {
            # lateral/heading
            "ey_rms": ey_rms_arr,
            "ey_max": ey_max_arr,
            "psi_rms": psi_rms_arr,
            "ey_rms_mean": float(ey_rms_arr.mean()),
            "ey_rms_std":  float(ey_rms_arr.std()),
            "ey_max_mean": float(ey_max_arr.mean()),
            "ey_max_std":  float(ey_max_arr.std()),
            "psi_rms_mean": float(psi_rms_arr.mean()),
            "psi_rms_std":  float(psi_rms_arr.std()),
            # state-wise RMSE
            "rmse_x": rmse_x_arr,
            "rmse_y": rmse_y_arr,
            "rmse_heading": rmse_heading_arr,
            "rmse_vel": rmse_vel_arr,
            "rmse_x_mean": float(rmse_x_arr.mean()),
            "rmse_x_std":  float(rmse_x_arr.std()),
            "rmse_y_mean": float(rmse_y_arr.mean()),
            "rmse_y_std":  float(rmse_y_arr.std()),
            "rmse_heading_mean": float(rmse_heading_arr.mean()),
            "rmse_heading_std":  float(rmse_heading_arr.std()),
            "rmse_vel_mean": float(rmse_vel_arr.mean()),
            "rmse_vel_std":  float(rmse_vel_arr.std()),
            # overall prediction RMSE
            "overall_rmse": overall_rmse_arr,
            "overall_rmse_mean": float(overall_rmse_arr.mean()),
            "overall_rmse_std":  float(overall_rmse_arr.std()),
        }

    # ---------------------------
    # 결과 저장 및 그래프 작성
    # ---------------------------
    np.save(os.path.join(RESULT_DIR, "window_ktrain_sensitivity_results.npy"), results, allow_pickle=True)
    print(f"[INFO] Saved window length (k_train) sensitivity raw results to {RESULT_DIR}")

    # k_train vs RMS ey 그래프
    ks = sorted(results.keys())
    ey_means = [results[k]["ey_rms_mean"] for k in ks]
    ey_stds  = [results[k]["ey_rms_std"]  for k in ks]

    plt.figure(figsize=(6, 4))
    plt.errorbar(ks, ey_means, yerr=ey_stds, fmt='-o', capsize=4)
    plt.xlabel("Training window length $k_{\\text{train}}$ (steps)")
    plt.ylabel("RMS Lateral Error $e_y$ (m)")
    plt.title("Training Window Length Sensitivity (RMS $e_y$), Test with $k_{\\text{test}}=20$")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "window_ktrain_sensitivity_rms_ey.png"), dpi=500)
    plt.close()
    print("[INFO] Saved training window length sensitivity figure (RMS ey).")

    # state-wise RMSE 요약 (mean)
    rmse_x_means = [results[k]["rmse_x_mean"] for k in ks]
    rmse_y_means = [results[k]["rmse_y_mean"] for k in ks]
    rmse_h_means = [results[k]["rmse_heading_mean"] for k in ks]
    rmse_v_means = [results[k]["rmse_vel_mean"] for k in ks]
    overall_means = [results[k]["overall_rmse_mean"] for k in ks]

    # --------------------------------------
    # Save state-wise RMSE + overall RMSE 데이터를 TXT로 저장
    # --------------------------------------
    statewise_rmse_dir = os.path.join(RESULT_DIR, "window_ktrain_sensitivity_statewise_rmse")
    ensure_dir(statewise_rmse_dir)

    # Save k_train 값
    np.savetxt(
        os.path.join(statewise_rmse_dir, "ktrain_values.txt"),
        np.array(ks),
        fmt="%d"
    )

    # Save RMSE means
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_x_mean.txt"),
        np.array(rmse_x_means),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_y_mean.txt"),
        np.array(rmse_y_means),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_heading_mean.txt"),
        np.array(rmse_h_means),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_velocity_mean.txt"),
        np.array(rmse_v_means),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "overall_rmse_mean.txt"),
        np.array(overall_means),
        fmt="%.6f"
    )

    # Save standard deviations
    rmse_x_stds = [results[k]["rmse_x_std"] for k in ks]
    rmse_y_stds = [results[k]["rmse_y_std"] for k in ks]
    rmse_h_stds = [results[k]["rmse_heading_std"] for k in ks]
    rmse_v_stds = [results[k]["rmse_vel_std"] for k in ks]
    overall_stds = [results[k]["overall_rmse_std"] for k in ks]

    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_x_std.txt"),
        np.array(rmse_x_stds),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_y_std.txt"),
        np.array(rmse_y_stds),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_heading_std.txt"),
        np.array(rmse_h_stds),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_velocity_std.txt"),
        np.array(rmse_v_stds),
        fmt="%.6f"
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "overall_rmse_std.txt"),
        np.array(overall_stds),
        fmt="%.6f"
    )

    print(f"[INFO] Saved state-wise + overall RMSE txt files to {statewise_rmse_dir}")

    # state-wise RMSE 플롯
    plt.figure(figsize=(6, 4))
    plt.plot(ks, rmse_x_means, '-o', label='RMSE X')
    plt.plot(ks, rmse_y_means, '-o', label='RMSE Y')
    plt.plot(ks, rmse_h_means, '-o', label='RMSE Heading')
    plt.plot(ks, rmse_v_means, '-o', label='RMSE Velocity')
    plt.xlabel("Training window length $k_{\\text{train}}$ (steps)")
    plt.ylabel("Prediction RMSE (state)")
    plt.title("Training Window Length Sensitivity (State-wise Prediction RMSE)\nTest with $k_{\\text{test}}=20$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "window_ktrain_sensitivity_statewise_rmse.png"), dpi=500)
    plt.close()
    print("[INFO] Saved training window length sensitivity figure (state-wise RMSE).")

    # overall prediction RMSE 플롯
    plt.figure(figsize=(6, 4))
    plt.plot(ks, overall_means, '-o', label='Overall Prediction RMSE')
    plt.xlabel("Training window length $k_{\\text{train}}$ (steps)")
    plt.ylabel("Overall Prediction RMSE")
    plt.title("Training Window Length Sensitivity (Overall Prediction RMSE)\nTest with $k_{\\text{test}}=20$")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "window_ktrain_sensitivity_overall_rmse.png"), dpi=500)
    plt.close()
    print("[INFO] Saved training window length sensitivity figure (overall RMSE).")


if __name__ == "__main__":
    main()
