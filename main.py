import os
import torch
import numpy as np

from config import *
from dataset import load_and_preprocess_data, build_dataset
from train import train_model
from eval import evaluate_and_save_logs


def main():
    # 경로 생성 (없으면 자동 생성)
    for _dir in [FIGURE_DIR, ARTIFACT_DIR, RESULT_DIR]:
        os.makedirs(_dir, exist_ok=True)

    input_data, output_data, input_scaler, input_power_transformer, output_scaler, output_power_transformer = \
        load_and_preprocess_data("train_data/tb3_train_set_25.11.23.txt", input_dim, output_dim)

    time_series = np.hstack((input_data, output_data))
    total_size = len(time_series)
    train_size = int(0.9 * total_size)

    trainX, trainY = build_dataset(time_series[:train_size], seq_length, input_dim, output_dim)
    testX, testY = build_dataset(time_series[train_size:], seq_length, input_dim, output_dim)

    # 데이터는 항상 CPU 텐서로 유지
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)
    testX_tensor = torch.FloatTensor(testX)
    testY_tensor = torch.FloatTensor(testY)

    # train_model 내부에서 배치 단위로만 GPU로 옮겨 사용
    lstm_model, rcm_model, likelihood, lstm_optimizer, rcm_optimizer = train_model(
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
        iterations,
        warmup_epochs,
    )

    # 평가 시에도 CPU 텐서를 넘기고, 내부에서 배치 단위로 GPU 사용
    evaluate_and_save_logs(
        testX_tensor,
        testY_tensor,
        lstm_model,
        rcm_model,
        likelihood,
        output_dim,
        output_scaler,
        output_power_transformer,
        lambda_uncertainty,
        apply_gaussian_filter=True,
        sigma=5,
    )


if __name__ == "__main__":
    main()
