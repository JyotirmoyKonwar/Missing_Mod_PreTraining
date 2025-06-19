# Physionet Encoder Configuration

# -- Data Configuration --
TRAIN_PATH = 'train.h5'
VAL_PATH = 'val.h5'

# -- Model Hyperparameters --
INPUT_DIM = 6       # Number of features in the Physionet dataset
NUM_CLASSES = 3     # Number of target classes
D_MODEL = 128       # Dimension of embeddings
N_HEAD = 4          # Number of attention heads
NUM_LAYERS = 3      # Number of Transformer encoder layers
DIM_FEEDFORWARD = 256 # Hidden dimension of the feed-forward network

# -- Training Hyperparameters --
BATCH_SIZE = 32
NUM_EPOCHS = 10     # Number of epochs to train for
LEARNING_RATE = 1e-4

# -- Logging --
LOG_FILE = "training_log.txt"