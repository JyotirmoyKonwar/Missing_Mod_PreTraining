# --- configs.py ---

import torch

# -- Environment --
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -- Data & File Paths --
# TODO: Update these paths to your actual pre-trained model files
UAH_MODEL_PATH = 'UAH_Encoder/uah_best_model.pth'
PHYSIO_MODEL_PATH = 'Physionet_Encoder/physionet_best_model.pth'
# TODO: Update these paths to your actual data files
UAH_DATA_PATH = 'path/to/your/uah_data.h5'
PHYSIO_DATA_PATH = 'path/to/your/physio_data.h5'
LABELS_PATH = 'path/to/your/labels.h5'
# Path to save the final fusion model
BEST_FUSION_MODEL_PATH = 'best_missing_modality_model.pth'


# -- Model Hyperparameters --
ENCODER_DIM = 128      # Output dimension of pretrained encoders
SHARED_DIM = 128       # Dimension of the shared latent space
HIDDEN_DIM = 256       # Hidden dimension for the final classifier
NUM_CLASSES = 3        # Number of target classes
ATTENTION_HEADS = 4    # Number of heads for attention mechanisms

# -- Training Hyperparameters --
BATCH_SIZE = 32
NUM_EPOCHS = 100       # Total epochs for the 3-stage training
INITIAL_LR = 1e-4

# -- 3-Stage Training Config --
# Stage 1: Warm-up
STAGE1_EPOCHS = 20
STAGE1_ALPHA = 0.1     # Contrastive loss weight
STAGE1_BETA = 0.1      # Consistency loss weight
STAGE1_MISSING_RATE = 0.3

# Stage 2: Joint Training
STAGE2_EPOCHS = 70     # Ends at epoch 70
STAGE2_ALPHA = 0.3
STAGE2_BETA = 0.2
STAGE2_MISSING_RATE = 0.4

# Stage 3: Fine-tuning
STAGE3_ALPHA = 0.5
STAGE3_BETA = 0.3
STAGE3_LR_FACTOR = 0.1 # lr is divided by 10 for fine-tuning
STAGE3_MISSING_RATE = 0.5

# -- Loss Function Weights --
GAMMA = 0.1            # Regularization loss weight
TEMPERATURE = 0.1      # Temperature for InfoNCE loss