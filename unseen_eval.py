# main.py
import torch
import numpy as np
import os

from config import *
from dataset import load_and_preprocess_data, build_dataset
from train import train_model
from eval import evaluate_and_save_logs


def main():
    # 0. 결과 디렉토리 생성
    for _dir in [FIGURE_DIR, ARTIFACT_DIR, RESULT_DIR]:
        os.makedirs(_dir, exist_ok=True)

    # 1. Train 데이터: 전체를 학습에 사용 (여기서만 scaler / power_transformer fit)
    train_input_data, train_output_data, input_scaler, input_power_transformer, output_scaler, output_power_transformer = \
        load_and_preprocess_data(
            filepath="train_data/tb3_train_set_25.11.23.txt",
            input_dim=input_dim,
            output_dim=output_dim,
            fit=True,  # train: fit + transform
        )

    # train 시계열 구성 및 dataset 생성
    time_series_train = np.hstack((train_input_data, train_output_data))
    trainX, trainY = build_dataset(time_series_train, seq_length, input_dim, output_dim)

    # 2. Test용 입력 시계열 구성
    #    └ train 전체를 학습에 사용하되, 평가 입력 역시 train에서 만든 시퀀스를 그대로 사용한다고 가정
    #       (필요하면 여기서 따로 구간을 잘라서 평가에 쓸 수 있음)
    testX = trainX.copy()  # 입력 시퀀스는 train과 동일하게 사용
    # trainY는 모델 학습 전용 타깃 (스케일된 상태). 평가에서는 unseen 파일의 두 열만 실제 정답으로 사용.

    # 3. unseen_test_data.txt 에서 "정답 2개" 로드 (스케일링 없이 원래 단위 그대로)
    #    파일 형태 가정: 각 행이 [y1, y2] 두 개의 타깃 값
    unseen_target = np.loadtxt("test_data/unseen_test_data.txt")  # shape: (T_unseen, 2)
    if unseen_target.ndim == 1:
        unseen_target = unseen_target.reshape(-1, 2)

    # 3-1. 길이 맞추기: testX 시퀀스 개수와 unseen_target 행 수 중 작은 쪽으로 맞춤
    n_seq = testX.shape[0]
    n_unseen = unseen_target.shape[0]
    n_eval = min(n_seq, n_unseen)

    testX = testX[-n_eval:]                # 뒤에서 n_eval개 사용
    unseen_target = unseen_target[-n_eval:]  # 뒤에서 n_eval개 사용

    # 4. 텐서 변환 (CPU 텐서로 유지 → train.py 의 DataLoader(pin_memory)와 호환)
    trainX_tensor = torch.FloatTensor(trainX)
    trainY_tensor = torch.FloatTensor(trainY)

    testX_tensor = torch.FloatTensor(testX)
    # unseen_target 은 원래 단위 그대로 (output_dim_eval = 2)
    testY_unscaled_tensor = torch.FloatTensor(unseen_target)  # (n_eval, 2)
    output_dim_eval = 2

    # 5. 모델 학습 (train 전체 사용)
    lstm_model, rcm_model, likelihood, lstm_optimizer, rcm_optimizer = train_model(
        trainX_tensor, trainY_tensor,
        input_dim, hidden_dim, output_dim, num_layers, num_inducing_points,
        lambda_uncertainty, alpha_elbo_max,
        device, iterations, warmup_epochs
    )

    # 6. 평가
    #    여기서는 다음과 같이 동작하도록 eval 함수를 설계해야 합니다:
    #    1) testX_tensor 를 스케일된 입력 공간에서 모델에 넣어 예측 (output_dim 전체)
    #    2) 예측 결과를 output_power_transformer, output_scaler 로 inverse transform → 원래 단위 (output_dim 전체)
    #    3) 그중 첫 2차원만 잘라서 unseen_target (원래 단위, testY_unscaled_tensor) 와 비교
    #       (MSE, RMSE 등)
    #
    #    따라서 evaluate_and_save_logs 에서는:
    #    - testY_tensor 를 "원래 단위의 2차원 타깃" 으로 받고
    #    - output_scaler, output_power_transformer 는 "예측을 역변환"하는 데만 사용
    #    - 내부에서 출력 차원 0,1만 이용해서 metric 계산
    evaluate_and_save_logs(
        testX_tensor,              # 입력 시퀀스 (스케일된 상태)
        testY_unscaled_tensor,     # 정답 2차원 (원래 단위)
        lstm_model, rcm_model, likelihood,
        output_dim_eval,           # 평가에 쓸 출력 차원 수 = 2
        output_scaler, output_power_transformer,
        lambda_uncertainty,
        apply_gaussian_filter=True,
        sigma=5,
    )


if __name__ == "__main__":
    main()
