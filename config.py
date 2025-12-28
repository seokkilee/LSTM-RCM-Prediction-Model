import torch
import numpy as np
import random

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# 하이퍼파라미터
seq_length = 20
input_dim = 6
hidden_dim = 128
output_dim = 4
num_layers = 3
learning_rate = 1e-3
batch_size = 64
iterations = 50
num_inducing_points = 30
lambda_uncertainty = 0.5

# ELBO(정규화) 손실의 warmup 및 가중치
alpha_elbo_max = 0.001         # ELBO 손실 가중치 최대값
warmup_epochs = 25             # warmup 기간(epoch 수)

# Learning Rate Decay (scheduler) 옵션
lr_scheduler_type = "plateau"  # "plateau" or "step"
lr_patience = 5                # plateau: 몇 epoch 동안 개선 없으면 decay
lr_factor = 0.5                # learning rate 곱셈 factor
lr_step_size = 10              # step decay일 경우 step size
min_lr = 1e-9                  # 최소 학습률

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print("===================================================")
    print(f"CUDA is available --> {torch.cuda.is_available()}")
    print(f"CUDA device count --> {torch.cuda.device_count()}")
    print(f"Current CUDA device --> {torch.cuda.current_device()}")
    print(f"CUDA device name --> {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print("===================================================")
else:
    print("===================================================")
    print("CUDA is not available. Using CPU.")
    print("===================================================")

# 저장 디렉토리
FIGURE_DIR = 'figures/'
ARTIFACT_DIR = 'artifacts/'
RESULT_DIR = 'results/'
