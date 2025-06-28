# --- configs.py ---

import torch

# -- Environment --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Data & File Paths --
# TODO: Update these paths to your single HDF5 data files
UAH_DATA_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/UAH_data_all.h5'
PHYSIO_DATA_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/Physionet_all.h5'

# TODO: Update these paths to your actual pre-trained model files
UAH_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/uah_best_model.pth'
PHYSIO_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/physionet_best_model.pth'

# Path to save the final fusion model
BEST_FUSION_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/best_missing_modality_model.pth'

# -- Data Splitting --
VALIDATION_SPLIT = 0.2  # 20% of the data will be used for validation
SHUFFLE_DATASET = True  # Whether to shuffle the data before splitting
RANDOM_SEED = 69        # Seed for reproducibility of the split

# -- Model Hyperparameters --
ENCODER_DIM = 128      # Output dimension of pretrained encoders
SHARED_DIM = 128       # Dimension of the shared latent space
HIDDEN_DIM = 256       # Hidden dimension for the final classifier
NUM_CLASSES = 3        # Number of target classes
ATTENTION_HEADS = 4    # Number of heads for attention mechanisms

# -- Training Hyperparameters --
BATCH_SIZE = 256
NUM_EPOCHS = 50       # Total epochs for the 3-stage training
INITIAL_LR = 1e-4

# -- 3-Stage Training Config --
# Stage 1: Warm-up
STAGE1_EPOCHS = 10      # OG = 20
STAGE1_ALPHA = 0.1     # Contrastive loss weight
STAGE1_BETA = 0.1      # Consistency loss weight
STAGE1_MISSING_RATE = 0.3

# Stage 2: Joint Training
STAGE2_EPOCHS = 35     # OG = 70
STAGE2_ALPHA = 0.3
STAGE2_BETA = 0.2
STAGE2_MISSING_RATE = 0.5

# Stage 3: Fine-tuning
STAGE3_ALPHA = 0.5
STAGE3_BETA = 0.3
STAGE3_LR_FACTOR = 0.1 # lr is divided by 10 for fine-tuning
STAGE3_MISSING_RATE = 0.5

# -- Loss Function Weights --
GAMMA = 0.1            # Regularization loss weight
TEMPERATURE = 0.1      # Temperature for InfoNCE loss


#############################################

# --- configs.py ---

import torch

# -- Environment --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Data & File Paths --
# TODO: Update these paths to your single HDF5 data files
UAH_DATA_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/UAH_data_all.h5'
PHYSIO_DATA_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/Physionet_all.h5'

# TODO: Update these paths to your actual pre-trained model files
UAH_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/UAH_Encoder/uah_best_model.pth'
PHYSIO_MODEL_PATH = '/csehome/p23iot002/Missing_Mod/Physionet_Encoder/physionet_best_model.pth'

# -- Model Saving Paths --
# Path for the best model from each stage
BEST_STAGE1_MODEL_PATH = "best_stage1_model.pth"
BEST_STAGE2_MODEL_PATH = "best_stage2_model.pth"
BEST_STAGE3_MODEL_PATH = "best_stage3_model.pth"
# Path for the model from the final epoch
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
BATCH_SIZE = 256
NUM_EPOCHS_STAGE1 = 10
NUM_EPOCHS_STAGE2 = 35  # Total epochs for this stage (e.g., 70 - 20)
NUM_EPOCHS_STAGE3 = 15  # Total epochs for this stage (e.g., 100 - 70)
INITIAL_LR = 1e-4

# -- Stage-Specific Config --
# Stage 1: Warm-up
STAGE1_ALPHA = 0.1
STAGE1_BETA = 0.1
STAGE1_MISSING_RATE = 0.3

# Stage 2: Joint Training
STAGE2_ALPHA = 0.3
STAGE2_BETA = 0.2
STAGE2_MISSING_RATE = 0.4

# Stage 3: Fine-tuning
STAGE3_ALPHA = 0.5
STAGE3_BETA = 0.3
STAGE3_LR_FACTOR = 0.1
STAGE3_MISSING_RATE = 0.5

# -- Loss Function Weights --
GAMMA = 0.1
TEMPERATURE = 0.1