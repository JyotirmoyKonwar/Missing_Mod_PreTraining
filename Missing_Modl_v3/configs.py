# --- configs.py ---

import torch

# -- Environment --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Data & File Paths --
UAH_DATA_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/UAH_data_all.h5'
PHYSIO_DATA_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/Physionet_all.h5'
UAH_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/uah_best_model.pth'
PHYSIO_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/physionet_best_model.pth'

# -- Model Saving Paths --
BEST_STAGE1_MODEL_PATH = "best_stage1_model_v3.pth"
BEST_STAGE2_MODEL_PATH = "best_stage2_model_v3.pth"
BEST_STAGE3_MODEL_PATH = "best_stage3_model_v3.pth"
LAST_EPOCH_MODEL_PATH = "last_epoch_model_v3.pth"

# -- Data Splitting (UPDATED) --
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1
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
INITIAL_LR = 1e-4
NUM_EPOCHS_STAGE1 = 20
NUM_EPOCHS_STAGE2 = 50
NUM_EPOCHS_STAGE3 = 30

# -- Stage-Specific Loss Weights --
STAGE1_ALPHA, STAGE1_BETA = 0.05, 0.05 # Contrastive, Consistency
STAGE2_ALPHA, STAGE2_BETA = 0.2, 0.15
STAGE3_ALPHA, STAGE3_BETA = 0.3, 0.25

# -- General Loss Component Weights --
GAMMA = 0.1       # Regularization
DELTA = 0.2       # Auxiliary classification
EPSILON = 0.15    # Cross-modal reconstruction
TEMPERATURE = 0.1

# -- SOLUTION 1: MODALITY-SPECIFIC LEARNING RATE CONFIGS --
ENABLE_MODALITY_SPECIFIC_LR = True
UAH_LR_FACTOR = 1.0      # UAH encoder gets a standard learning rate
PHYSIO_LR_FACTOR = 1.5   # Physio encoder gets a higher learning rate

# -- SOLUTION 4: BALANCED PROGRESSIVE UNFREEZING CONFIGS --
ENABLE_MODALITY_SPECIFIC_UNFREEZING = True
UAH_UNFREEZE_SCHEDULE = {
    10: ['classifier'],
    20: ['transformer.layers.2'],
    30: ['transformer.layers.1', 'transformer.layers.0'],
    50: ['embed']
}
PHYSIO_UNFREEZE_SCHEDULE = {
    8: ['classifier'],      # Earlier unfreezing for the weaker modality
    15: ['transformer.layers.2'],
    25: ['transformer.layers.1', 'transformer.layers.0'],
    40: ['embed']
}