"""Central configuration for Data Poisoning & Detection pipeline."""

import os
import torch

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _dir in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)

# Hardware — RTX Ada 4500 (24 GB VRAM), Threadripper (~200 GB RAM)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
NUM_WORKERS = min(os.cpu_count() or 8, 16)
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"

# Training
EPOCHS = 30
LEARNING_RATE = 0.1
WEIGHT_DECAY = 5e-4
MOMENTUM = 0.9
LR_MILESTONES = [15, 25]
LR_GAMMA = 0.1

# Attack
POISON_RATE = 0.05
LABEL_FLIP_SOURCE = 0           # airplane → truck
LABEL_FLIP_TARGET = 9
BACKDOOR_TARGET_LABEL = 0       # backdoor target: airplane
BACKDOOR_PATCH_SIZE = 3          # 3×3 white patch

# Detection
LOSS_OUTLIER_PERCENTILE = 95
SPECTRAL_EPSILON = 1.5
KNN_K = 10
ACTIVATION_N_CLUSTERS = 2

# Reproducibility
SEED = 42
