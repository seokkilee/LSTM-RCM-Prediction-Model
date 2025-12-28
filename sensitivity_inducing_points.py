# sensitivity_inducing_points.py
#
# H3: SVGP Inducing Points M Sensitivity

import os
import gc
import time
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
N_runs = 5                               # 각 M 값별 반복 횟수 (평균±표준편차)
M_list = [10, 30, 60, 120]               # SVGP inducing points 후보들

# 모델이 올라갈 장치 (GPU가 있으면 GPU, 아니면 CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # 디렉토리 생성
    ensure_dir(FIGURE_DIR)
    ensure_dir(RESULT_DIR)
    ensure_dir(ARTIFACT_DIR)

    # ---------------------------
    # 데이터 로딩 및 전처리 (M과 무관하게 1회 수행)
    # ---------------------------
    print("[INFO] Loading and preprocessing dataset...")
    input_data, output_data, input_scaler, input_pt, output_scaler, output_pt = \
        load_and_preprocess_data("train_data/tb3_train_set_25.11.23.txt", input_dim, output_dim)

    time_series = np.hstack((input_data, output_data))
    total_size = len(time_series)
    train_size = int(0.9 * total_size)

    # baseline window length = seq_length (config), train/test 동일
    k = seq_length
    print(f"[INFO] Building datasets with seq_length = {k} (history = {k * 50} ms)")

    trainX, trainY = build_dataset(time_series[:train_size], k, input_dim, output_dim)
    testX, testY = build_dataset(time_series[train_size:], k, input_dim, output_dim)

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
    # Inducing Points M 민감도 실험
    # ---------------------------
    for M in M_list:
        print("=" * 80)
        print(f"[INFO] Inducing points sensitivity: M = {M}")
        print("=" * 80)

        M_result_dir = os.path.join(RESULT_DIR, f"inducing_M{M}")
        M_fig_dir = os.path.join(FIGURE_DIR, f"inducing_M{M}")
        ensure_dir(M_result_dir)
        ensure_dir(M_fig_dir)

        ey_rms_list = []
        ey_max_list = []
        psi_rms_list = []

        rmse_x_list = []
        rmse_y_list = []
        rmse_heading_list = []
        rmse_vel_list = []
        overall_rmse_list = []

        train_time_list = []

        for run in range(N_runs):
            print(f"\n[RUN {run+1}/{N_runs}] Training with M = {M} inducing points...")

            save_path = os.path.join(
                ARTIFACT_DIR, f"inducing_M{M}_run{run+1}.pth"
            )

            t_start = time.perf_counter()

            lstm_model, rcm_model, likelihood, lstm_opt, rcm_opt, history = train_model(
                trainX_tensor,
                trainY_tensor,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_layers=num_layers,
                num_inducing_points=M,
                lambda_uncertainty=lambda_uncertainty,
                alpha_elbo_max=alpha_elbo_max,
                device=device,
                epochs=iterations,
                warmup_epochs=warmup_epochs,
                val_ratio=0.1,
                save_model_path=save_path,
                return_history=True,
            )

            t_end = time.perf_counter()
            train_time = t_end - t_start
            train_time_list.append(train_time)

            print(f"[TIME ] M={M}, run={run+1} | train_time = {train_time:.2f} s")

            prefix = f"M{M}_run{run+1}"
            fig_path = os.path.join(M_fig_dir, f"{prefix}_prediction.png")

            rmse_dict, pred_np, true_np = evaluate_model_full(
                lstm_model,
                rcm_model,
                likelihood,
                testX_tensor,
                testY_tensor,
                output_scaler,
                output_pt,
                lambda_uncertainty=lambda_uncertainty,
                device=device,
                batch_size_eval=batch_size,
                result_dir=M_result_dir,
                result_prefix=prefix,
                figure_path=fig_path,
            )

            rmse_x_list.append(rmse_dict["x"])
            rmse_y_list.append(rmse_dict["y"])
            rmse_heading_list.append(rmse_dict["heading"])
            rmse_vel_list.append(rmse_dict["velocity"])
            overall_rmse_list.append(rmse_dict["overall_rmse"])

            ey_rms_list.append(rmse_dict["ey_rms"])
            ey_max_list.append(rmse_dict["ey_max"])
            psi_rms_list.append(rmse_dict["psi_rms"])

            print(
                f"[RESULT] M={M}, run={run+1} | "
                f"ey_RMS={rmse_dict['ey_rms']:.4f}, "
                f"ey_max={rmse_dict['ey_max']:.4f}, "
                f"psi_RMS={rmse_dict['psi_rms']:.4f}, "
                f"RMSE_x={rmse_dict['x']:.4f}, "
                f"RMSE_y={rmse_dict['y']:.4f}, "
                f"RMSE_heading={rmse_dict['heading']:.4f}, "
                f"RMSE_vel={rmse_dict['velocity']:.4f}, "
                f"RMSE_overall={rmse_dict['overall_rmse']:.4f}"
            )

        ey_rms_arr = np.array(ey_rms_list)
        ey_max_arr = np.array(ey_max_list)
        psi_rms_arr = np.array(psi_rms_list)
        rmse_x_arr = np.array(rmse_x_list)
        rmse_y_arr = np.array(rmse_y_list)
        rmse_heading_arr = np.array(rmse_heading_list)
        rmse_vel_arr = np.array(rmse_vel_list)
        overall_rmse_arr = np.array(overall_rmse_list)
        train_time_arr = np.array(train_time_list)

        results[M] = {
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
            "overall_rmse": overall_rmse_arr,
            "overall_rmse_mean": float(overall_rmse_arr.mean()),
            "overall_rmse_std": float(overall_rmse_arr.std()),
            "train_time": train_time_arr,
            "train_time_mean": float(train_time_arr.mean()),
            "train_time_std": float(train_time_arr.std()),
        }

    np.save(
        os.path.join(RESULT_DIR, "inducing_points_sensitivity_results.npy"),
        results,
        allow_pickle=True,
    )
    print(f"[INFO] Saved inducing points sensitivity raw results to {RESULT_DIR}")

    Ms = sorted(results.keys())

    ey_means = [results[M]["ey_rms_mean"] for M in Ms]
    ey_stds = [results[M]["ey_rms_std"] for M in Ms]

    plt.figure(figsize=(6, 4))
    plt.errorbar(Ms, ey_means, yerr=ey_stds, fmt='-o', capsize=4)
    plt.xlabel("Number of Inducing Points M")
    plt.ylabel("RMS Lateral Error $e_y$ (m)")
    plt.title("Inducing Points Sensitivity (RMS $e_y$)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "inducing_sensitivity_rms_ey.png"), dpi=500)
    plt.close()
    print("[INFO] Saved inducing points sensitivity figure (RMS ey).")

    rmse_x_means = [results[M]["rmse_x_mean"] for M in Ms]
    rmse_y_means = [results[M]["rmse_y_mean"] for M in Ms]
    rmse_h_means = [results[M]["rmse_heading_mean"] for M in Ms]
    rmse_v_means = [results[M]["rmse_vel_mean"] for M in Ms]
    overall_means = [results[M]["overall_rmse_mean"] for M in Ms]

    train_time_means = [results[M]["train_time_mean"] for M in Ms]
    train_time_stds = [results[M]["train_time_std"] for M in Ms]

    statewise_rmse_dir = os.path.join(RESULT_DIR, "inducing_points_sensitivity_statewise_rmse")
    ensure_dir(statewise_rmse_dir)

    np.savetxt(
        os.path.join(statewise_rmse_dir, "M_values.txt"),
        np.array(Ms),
        fmt="%d",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_x_mean.txt"),
        np.array(rmse_x_means),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_y_mean.txt"),
        np.array(rmse_y_means),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_heading_mean.txt"),
        np.array(rmse_h_means),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_velocity_mean.txt"),
        np.array(rmse_v_means),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "overall_rmse_mean.txt"),
        np.array(overall_means),
        fmt="%.6f",
    )

    rmse_x_stds = [results[M]["rmse_x_std"] for M in Ms]
    rmse_y_stds = [results[M]["rmse_y_std"] for M in Ms]
    rmse_h_stds = [results[M]["rmse_heading_std"] for M in Ms]
    rmse_v_stds = [results[M]["rmse_vel_std"] for M in Ms]
    overall_stds = [results[M]["overall_rmse_std"] for M in Ms]

    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_x_std.txt"),
        np.array(rmse_x_stds),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_y_std.txt"),
        np.array(rmse_y_stds),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_heading_std.txt"),
        np.array(rmse_h_stds),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "rmse_velocity_std.txt"),
        np.array(rmse_v_stds),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "overall_rmse_std.txt"),
        np.array(overall_stds),
        fmt="%.6f",
    )

    np.savetxt(
        os.path.join(statewise_rmse_dir, "train_time_mean.txt"),
        np.array(train_time_means),
        fmt="%.6f",
    )
    np.savetxt(
        os.path.join(statewise_rmse_dir, "train_time_std.txt"),
        np.array(train_time_stds),
        fmt="%.6f",
    )

    print(f"[INFO] Saved state-wise + overall RMSE + train time txt files to {statewise_rmse_dir}")

    plt.figure(figsize=(6, 4))
    plt.plot(Ms, rmse_x_means, '-o', label='RMSE X')
    plt.plot(Ms, rmse_y_means, '-o', label='RMSE Y')
    plt.plot(Ms, rmse_h_means, '-o', label='RMSE Heading')
    plt.plot(Ms, rmse_v_means, '-o', label='RMSE Velocity')
    plt.xlabel("Number of Inducing Points M")
    plt.ylabel("Prediction RMSE (state)")
    plt.title("Inducing Points Sensitivity (State-wise Prediction RMSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "inducing_sensitivity_statewise_rmse.png"), dpi=500)
    plt.close()
    print("[INFO] Saved inducing points sensitivity figure (state-wise RMSE).")

    plt.figure(figsize=(6, 4))
    plt.errorbar(Ms, overall_means, yerr=overall_stds, fmt='-o', capsize=4, label='Overall Prediction RMSE')
    plt.xlabel("Number of Inducing Points M")
    plt.ylabel("Overall Prediction RMSE")
    plt.title("Inducing Points Sensitivity (Overall Prediction RMSE)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "inducing_sensitivity_overall_rmse.png"), dpi=500)
    plt.close()
    print("[INFO] Saved inducing points sensitivity figure (overall RMSE).")

    plt.figure(figsize=(6, 4))
    plt.errorbar(Ms, train_time_means, yerr=train_time_stds, fmt='-o', capsize=4, label='Training Time')
    plt.xlabel("Number of Inducing Points M")
    plt.ylabel("Training Time per Run (s)")
    plt.title("Inducing Points vs Training Time")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, "inducing_sensitivity_train_time.png"), dpi=500)
    plt.close()
    print("[INFO] Saved inducing points sensitivity figure (training time).")


if __name__ == "__main__":
    main()
