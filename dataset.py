import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
import joblib
from config import ARTIFACT_DIR

def load_and_preprocess_data(filepath, input_dim, output_dim):
    data = np.loadtxt(filepath)
    input_data = data[:, :input_dim]
    output_data = data[:, input_dim:input_dim + output_dim]
    input_power_transformer = PowerTransformer(method='yeo-johnson')
    input_data = input_power_transformer.fit_transform(input_data)
    input_scaler = StandardScaler()
    input_data = input_scaler.fit_transform(input_data)
    output_power_transformer = PowerTransformer(method='yeo-johnson')
    output_data = output_power_transformer.fit_transform(output_data)
    output_scaler = StandardScaler()
    output_data = output_scaler.fit_transform(output_data)
    # 각종 스케일러 저장
    joblib.dump(input_power_transformer, ARTIFACT_DIR + "input_power_transformer.pkl")
    joblib.dump(input_scaler, ARTIFACT_DIR + "input_scaler.pkl")
    joblib.dump(output_power_transformer, ARTIFACT_DIR + "output_power_transformer.pkl")
    joblib.dump(output_scaler, ARTIFACT_DIR + "output_scaler.pkl")
    return input_data, output_data, input_scaler, input_power_transformer, output_scaler, output_power_transformer

def build_dataset(time_series, seq_length, input_dim, output_dim):
    X, Y = [], []
    for i in range(len(time_series) - seq_length):
        X.append(time_series[i:i + seq_length, :input_dim])
        Y.append(time_series[i + seq_length, input_dim:input_dim + output_dim])
    return np.array(X), np.array(Y)
