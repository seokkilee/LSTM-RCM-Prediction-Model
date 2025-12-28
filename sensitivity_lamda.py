# sensitivity_lambda.py (MATCHED TO sensitivity_layers.py)

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
N_runs = 5  # 각 lambda별 반복 횟수 (평균±표준편차)
lambda_list = [0.1, 0.3, 0.5, 0.7, 1.0]

# 모델이 올라갈 장치 (GPU가 있으면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 디렉토리 생성 (layers와 동일)
    ensure_dir(FIGURE_DIR)
    ensure_dir(RESULT_DIR)
    ensure_dir(ARTIFACT_DIR)

    # ---------------------------
    # 데이터 로딩 및 전처리
    # ---------------------------
    print("[INFO] Loading and preprocessing dataset...")
    data_path = "train_data/tb3_train_set_25.11.23.txt"
    input_data, output_data, input_scaler, input_pt, output_scaler, output_pt = \
        load_and_preprocess_data(data_path, input_dim, output_dim)

    time_series = np.hstack((input_data, output_data))
    total_size = len(time_series)
    train_size = int(0.9 * total_size)

    trainX, trainY = build_dataset(time_series[:train_size], seq_length, input_dim, output_dim)
    testX, testY = build_dataset(time_series[train_size:], seq_length, input_dim, output_dim)

    # CPU 텐서로 유지, train/eval 내부에서 device로 옮김
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)
    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)

    # ---------------------------
    # 결과 저장용 dict
    # ---------------------------
    results = {}

    # ---------------------------
    # Lambda 민감도 실험 (layers와 동일한 저장 구조)
    # ---------------------------
    for lam in lambda_list:
        print("=" * 70)
        print(f"[INFO] Lambda sensitivity: lambda_uncertainty = {lam}")
        print("=" * 70)

        lam_tag = f"lambda_{lam}".replace(".", "p")
        lam_result_dir = os.path.join(RESULT_DIR, f"{lam_tag}")
        lam_fig_dir = os.path.join(FIGURE_DIR, f"{lam_tag}")
        ensure_dir(lam_result_dir)
        ensure_dir(lam_fig_dir)

        ey_rms_list = []
        ey_max_list = []
        psi_rms_list = []

        rmse_x_list = []
        rmse_y_list = []
        rmse_heading_list = []
        rmse_vel_list = []

        for run in range(N_runs):
            print(f"\n[RUN {run+1}/{N_runs}] Training with lambda={lam} ...")

            save_path = os.path.join(
                ARTIFACT_DIR, f"lstm_lambda_{lam_tag}_run{run+1}.pth"
            )

            # train_model 호출은 layers 코드 스타일로 "키워드 인자" 사용 (시그니처 변경에도 안전)
            lstm_model, rcm_model, likelihood, lstm_opt, rcm_opt, history = train_model(
                trainX_tensor,
                trainY_tensor,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,  # layers 실험과 달리 config의 기본 num_layers 사용
                num_inducing_points=num_inducing_points,
                lambda_uncertainty=lam,
                alpha_elbo_max=alpha_elbo_max,
                device=device,
                epochs=iterations,
                warmup_epochs=warmup_epochs,
                val_ratio=0.1,
                save_model_path=save_path,
                return_history=True,
            )

            prefix = f"{lam_tag}_run{run+1}"
            fig_path = os.path.join(lam_fig_dir, f"{prefix}_prediction.png")

            rmse_dict, pred_np, true_np = evaluate_model_full(
                lstm_model,
                rcm_model,
                likelihood,
                testX_tensor,
                testY_tensor,
                output_scaler,
                output_pt,
                lambda_uncertainty=lam,
                device=device,
                batch_size_eval=batch_size,
                result_dir=lam_result_dir,
                result_prefix=prefix,
                figure_path=fig_path,
            )

            rmse_x_list.append(rmse_dict["x"])
            rmse_y_list.append(rmse_dict["y"])
            rmse_heading_list.append(rmse_dict["heading"])
            rmse_vel_list.append(rmse_dict["velocity"])

            ey_rms_list.append(rmse_dict["ey_rms"])
            ey_max_list.append(rmse_dict["ey_max"])
            psi_rms_list.append(rmse_dict["psi_rms"])

            print(
                f"[RESULT] lambda={lam}, run={run+1} | "
                f"ey_RMS={rmse_dict['ey_rms']:.4f}, "
                f"ey_max={rmse_dict['ey_max']:.4f}, "
                f"psi_RMS={rmse_dict['psi_rms']:.4f}, "
                f"RMSE_x={rmse_dict['x']:.4f}, "
                f"RMSE_y={rmse_dict['y']:.4f}, "
                f"RMSE_heading={rmse_dict['heading']:.4f}, "
                f"RMSE_vel={rmse_dict['velocity']:.4f}"
            )

        ey_rms_arr = np.array(ey_rms_list)
        ey_max_arr = np.array(ey_max_list)
        psi_rms_arr = np.array(psi_rms_list)
        rmse_x_arr = np.array(rmse_x_list)
        rmse_y_arr = np.array(rmse_y_list)
        rmse_heading_arr = np.array(rmse_heading_list)
        rmse_vel_arr = np.array(rmse_vel_list)

        results[lam] = {
            "ey_rms": ey_rms_arr,
            "ey_max": ey_max_arr,
            "psi_rms": psi_rms_arr,
            "ey_rms_mean": float(ey_rms_arr.mean()),
            "ey_rms_std": float(ey_rms_arr.std()),
            "ey_max_mean": float(ey_max_arr.mean()),
            "ey_max_std": float(ey_max_arr.std()),
            "psi_rms_mean": float(psi_rms_arr.mean()),
            "psi_rms_std": float(psi_rms_arr.std()),
            "rmse_x": rmse_x_arr,
            "rmse_y": rmse_y_arr,
            "rmse_heading": rmse_heading_arr,
            "rmse_vel": rmse_vel_arr,
            "rmse_x_mean": float(rmse_x_arr.mean()),
            "rmse_x_std": float(rmse_x_arr.std()),
            "rmse_y_mean": float(rmse_y_arr.mean()),
            "rmse_y_std": float(rmse_y_arr.std()),
            "rmse_heading_mean": float(rmse_heading_arr.mean()),
            "rmse_heading_std": float(rmse_heading_arr.std()),
            "rmse_vel_mean": float(rmse_vel_arr.mean()),
            "rmse_vel_std": float(rmse_vel_arr.std()),
        }

    # ---------------------------
    # 결과 저장 및 그래프 작성 (layers와 동일 구조)
    # ---------------------------
    np.save(os.path.join(RESULT_DIR, "lambda_sensitivity_results.npy"), results, allow_pickle=True)
    print(f"[INFO] Saved lambda sensitivity raw results to {RESULT_DIR}")

    lams = sorted(results.keys())
    ey_means = [results[lam]["ey_rms_mean"] for lam in lams]
    ey_stds = [results[lam]["ey_rms_std"] for lam in lams]

    plt.figure(figsize=(6, 4))
    plt.errorbar(lams, ey_means, yerr=ey_stds, fmt='-o', capsize=4)
    plt.xlabel("Lambda Uncertainty (λ)")
    plt.ylabel("RMS Lateral Error (m)")
    plt.title("Lambda Sensitivity (RMS $e_y$)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "lambda_sensitivity_rms_ey.png"), dpi=500)
    plt.close()
    print("[INFO] Saved lambda sensitivity figure (RMS ey).")

    rmse_x_means = [results[lam]["rmse_x_mean"] for lam in lams]
    rmse_y_means = [results[lam]["rmse_y_mean"] for lam in lams]
    rmse_h_means = [results[lam]["rmse_heading_mean"] for lam in lams]
    rmse_v_means = [results[lam]["rmse_vel_mean"] for lam in lams]

    statewise_rmse_dir = os.path.join(RESULT_DIR, "lambda_sensitivity_statewise_rmse")
    ensure_dir(statewise_rmse_dir)

    np.savetxt(os.path.join(statewise_rmse_dir, "lambdas.txt"), np.array(lams), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_x_mean.txt"), np.array(rmse_x_means), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_y_mean.txt"), np.array(rmse_y_means), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_heading_mean.txt"), np.array(rmse_h_means), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_velocity_mean.txt"), np.array(rmse_v_means), fmt="%.6f")

    rmse_x_stds = [results[lam]["rmse_x_std"] for lam in lams]
    rmse_y_stds = [results[lam]["rmse_y_std"] for lam in lams]
    rmse_h_stds = [results[lam]["rmse_heading_std"] for lam in lams]
    rmse_v_stds = [results[lam]["rmse_vel_std"] for lam in lams]

    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_x_std.txt"), np.array(rmse_x_stds), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_y_std.txt"), np.array(rmse_y_stds), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_heading_std.txt"), np.array(rmse_h_stds), fmt="%.6f")
    np.savetxt(os.path.join(statewise_rmse_dir, "rmse_velocity_std.txt"), np.array(rmse_v_stds), fmt="%.6f")

    print(f"[INFO] Saved state-wise RMSE text data to {statewise_rmse_dir}")

    plt.figure(figsize=(6, 4))
    plt.plot(lams, rmse_x_means, '-o', label='RMSE X')
    plt.plot(lams, rmse_y_means, '-o', label='RMSE Y')
    plt.plot(lams, rmse_h_means, '-o', label='RMSE Heading')
    plt.plot(lams, rmse_v_means, '-o', label='RMSE Velocity')
    plt.xlabel("Lambda Uncertainty (λ)")
    plt.ylabel("RMSE (state)")
    plt.title("Lambda Sensitivity (State-wise RMSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "lambda_sensitivity_statewise_rmse.png"), dpi=500)
    plt.close()
    print("[INFO] Saved lambda sensitivity figure (state-wise RMSE).")


if __name__ == "__main__":
    main()
