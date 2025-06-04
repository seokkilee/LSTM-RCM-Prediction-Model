import torch
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
from config import RESULT_DIR

def evaluate_and_save_logs(
    testX_tensor, testY_tensor,
    lstm_model, rcm_model, likelihood,
    output_dim, output_scaler, output_power_transformer,
    lambda_uncertainty=1.0, apply_gaussian_filter=False, sigma=5
):
    with torch.no_grad():
        latent_test = lstm_model(testX_tensor)
        rcm_output_test = rcm_model(latent_test)
        rcm_correction_test = likelihood(rcm_output_test)
        rcm_mean = rcm_correction_test.mean
        rcm_std  = rcm_correction_test.variance.sqrt()
        weight = torch.exp(-lambda_uncertainty * rcm_std)
        final_pred_test = latent_test + (weight * rcm_mean)
        pred_std_test = rcm_std
    final_pred_np = final_pred_test.cpu().numpy()
    testY_np      = testY_tensor.cpu().numpy()
    pred_std_np   = pred_std_test.cpu().numpy()
    final_pred_np = output_scaler.inverse_transform(final_pred_np)
    final_pred_np = output_power_transformer.inverse_transform(final_pred_np)
    testY_np      = output_scaler.inverse_transform(testY_np)
    testY_np      = output_power_transformer.inverse_transform(testY_np)
    pred_std_np   = pred_std_np * output_scaler.scale_
    if apply_gaussian_filter:
        final_pred_np = gaussian_filter1d(final_pred_np, sigma=sigma, axis=0)
    output_names = ['pos_x', 'pos_y', 'heading', 'velocity']
    for d in range(output_dim):
        np.savetxt(os.path.join(RESULT_DIR, f"true_{output_names[d]}.txt"), testY_np[:, d], fmt="%.6f")
        np.savetxt(os.path.join(RESULT_DIR, f"lstm_rcm_{output_names[d]}.txt"), final_pred_np[:, d], fmt="%.6f")
        np.savetxt(os.path.join(RESULT_DIR, f"lstm_rcm_std_{output_names[d]}.txt"), pred_std_np[:, d], fmt="%.6f")
    print("[INFO] LSTM-RCM 예측 결과와 표준편차를 .txt로 저장 완료.")
