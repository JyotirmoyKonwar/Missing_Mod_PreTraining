# --- configs.py ---

import torch

# -- Environment --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2 # ADDED: For DataLoader, as per system suggestion

# -- Data & File Paths --
UAH_DATA_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/UAH_data_all.h5'
PHYSIO_DATA_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/Physionet_all.h5'
UAH_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/uah_best_model.pth'
PHYSIO_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/physionet_best_model.pth'

# -- Model Saving Paths --
BEST_STAGE1_MODEL_PATH = "best_stage1_model.pth"
BEST_STAGE2_MODEL_PATH = "best_stage2_model.pth"
BEST_STAGE3_MODEL_PATH = "best_stage3_model.pth"
LAST_EPOCH_MODEL_PATH = "last_epoch_model.pth"

# -- Data Splitting --
VALIDATION_SPLIT = 0.2
SHUFFLE_DATASET = True
RANDOM_SEED = 42

# -- Model Hyperparameters --
ENCODER_DIM = 128
SHARED_DIM = 128
HIDDEN_DIM = 256
NUM_CLASSES = 3
ATTENTION_HEADS = 4

# -- Training Hyperparameters --
BATCH_SIZE = 128
NUM_EPOCHS_STAGE1 = 10
NUM_EPOCHS_STAGE2 = 25
NUM_EPOCHS_STAGE3 = 15
INITIAL_LR = 1e-4

# -- Stage-Specific Config --
# Stage 1: Warm-up
STAGE1_ALPHA, STAGE1_BETA = 0.05, 0.05
# Stage 2: Joint Training
STAGE2_ALPHA, STAGE2_BETA = 0.2, 0.15
# Stage 3: Fine-tuning
STAGE3_ALPHA, STAGE3_BETA = 0.3, 0.25

# -- Advanced Configs from Roadmap --
ENCODER_LR_FACTOR = 0.1  # Encoder LR = INITIAL_LR * 0.1
PHYSIO_ONLY_LOSS_WEIGHT = 2.0

# -- Loss Function Weights --
GAMMA = 0.1       # Regularization
DELTA = 0.2       # Auxiliary classification
EPSILON = 0.15    # Cross-modal reconstruction
TEMPERATURE = 0.1