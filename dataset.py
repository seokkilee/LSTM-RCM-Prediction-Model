import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib
from config import ARTIFACT_DIR

def load_and_preprocess_data(
    filepath,
    input_dim,
    output_dim,
    input_scaler=None,
    input_power_transformer=None,
    output_scaler=None,
    output_power_transformer=None,
    fit: bool = True,
):
    """
    - fit=True:
        주어진 데이터로 PowerTransformer/Scaler를 새로 학습(fit) + 변환(transform)
        그리고 ARTIFACT_DIR에 pkl로 저장
    - fit=False:
        이미 학습된 PowerTransformer/Scaler로만 transform (저장 X)
        이 경우 input_scaler, input_power_transformer, output_scaler, output_power_transformer가 모두 필요
    """
    data = np.loadtxt(filepath)
    input_data = data[:, :input_dim]
    output_data = data[:, input_dim:input_dim + output_dim]

    # ===== 입력 쪽 =====
    if fit:
        if input_power_transformer is None:
            input_power_transformer = PowerTransformer(method='yeo-johnson')
        input_data = input_power_transformer.fit_transform(input_data)

        if input_scaler is None:
            input_scaler = StandardScaler()
        input_data = input_scaler.fit_transform(input_data)
    else:
        if input_power_transformer is None or input_scaler is None:
            raise ValueError("fit=False 인 경우, input_scaler와 input_power_transformer가 필요합니다.")
        input_data = input_power_transformer.transform(input_data)
        input_data = input_scaler.transform(input_data)

    # ===== 출력 쪽 =====
    if fit:
        if output_power_transformer is None:
            output_power_transformer = PowerTransformer(method='yeo-johnson')
        output_data = output_power_transformer.fit_transform(output_data)

        if output_scaler is None:
            output_scaler = StandardScaler()
        output_data = output_scaler.fit_transform(output_data)
    else:
        if output_power_transformer is None or output_scaler is None:
            raise ValueError("fit=False 인 경우, output_scaler와 output_power_transformer가 필요합니다.")
        output_data = output_power_transformer.transform(output_data)
        output_data = output_scaler.transform(output_data)

    # ===== 스케일러/변환기 저장 (train 시점에만) =====
    if fit:
        joblib.dump(input_power_transformer,  ARTIFACT_DIR + "input_power_transformer.pkl")
        joblib.dump(input_scaler,             ARTIFACT_DIR + "input_scaler.pkl")
        joblib.dump(output_power_transformer, ARTIFACT_DIR + "output_power_transformer.pkl")
        joblib.dump(output_scaler,            ARTIFACT_DIR + "output_scaler.pkl")

    return input_data, output_data, input_scaler, input_power_transformer, output_scaler, output_power_transformer


def build_dataset(time_series, seq_length, input_dim, output_dim):
    X, Y = [], []
    for i in range(len(time_series) - seq_length):
        X.append(time_series[i:i + seq_length, :input_dim])
        Y.append(time_series[i + seq_length, input_dim:input_dim + output_dim])
    return np.array(X), np.array(Y)
