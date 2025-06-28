# --- utils.py ---

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import configs

class MissingModalityDataset(Dataset):
    """
    Custom PyTorch dataset that serves a specific subset of data (a "view")
    from two HDF5 files using a provided list of indices.
    """
    def __init__(self, uah_h5_file, physio_h5_file, indices, missing_rate=0.0, training=True):
        self.uah_file = uah_h5_file
        self.physio_file = physio_h5_file
        self.indices = indices
        self.training = training
        self.missing_rate = missing_rate

        self.uah_x = self.uah_file['X']
        self.physio_x = self.physio_file['X']
        self.labels = self.uah_file['y'] # Assume labels are the same in both

    def __len__(self):
        # The length of the dataset is the number of indices in the split
        return len(self.indices)

    def __getitem__(self, idx):
        # Get the actual data index from our list of indices for this split
        data_idx = self.indices[idx]
        
        # Fetch data and convert to the correct type
        uah_seq = torch.from_numpy(self.uah_x[data_idx].astype(np.float32))
        physio_seq = torch.from_numpy(self.physio_x[data_idx].astype(np.float32))
        label = torch.tensor(self.labels[data_idx], dtype=torch.long).squeeze()

        modality_mask = torch.tensor([True, True])
        
        # Simulate missing modalities during training
        if self.training and np.random.random() < self.missing_rate:
            missing_modality = np.random.choice([0, 1])
            modality_mask[missing_modality] = False
            
            if missing_modality == 0:
                uah_seq = torch.zeros_like(uah_seq)
            else:
                physio_seq = torch.zeros_like(physio_seq)
        
        return uah_seq, physio_seq, modality_mask, label

def get_dataloaders():
    """
    Opens the main HDF5 files, performs a train/validation split on the indices,
    and returns the corresponding DataLoaders.
    """
    try:
        uah_h5_file = h5py.File(configs.UAH_DATA_PATH, 'r')
        physio_h5_file = h5py.File(configs.PHYSIO_DATA_PATH, 'r')
    except FileNotFoundError as e:
        print(f"FATAL: Could not find data file at paths specified in configs.py.")
        print(f"Please update UAH_DATA_PATH and PHYSIO_DATA_PATH.")
        print(f"Details: {e}")
        exit() # Exit the script if data is not found

    # Determine the number of aligned samples
    num_samples = min(len(uah_h5_file['X']), len(physio_h5_file['X']))

    # Create and shuffle indices for splitting
    indices = list(range(num_samples))
    if configs.SHUFFLE_DATASET:
        np.random.seed(configs.RANDOM_SEED)
        np.random.shuffle(indices)

    # Perform the split
    split_point = int(np.floor(configs.VALIDATION_SPLIT * num_samples))
    val_indices, train_indices = indices[:split_point], indices[split_point:]

    print(f"Dataset split: {len(train_indices)} training samples, {len(val_indices)} validation samples.")

    # Create dataset instances for the train and validation splits
    train_dataset = MissingModalityDataset(
        uah_h5_file, physio_h5_file, train_indices,
        missing_rate=configs.STAGE1_MISSING_RATE, # Initial rate
        training=True
    )
    val_dataset = MissingModalityDataset(
        uah_h5_file, physio_h5_file, val_indices,
        missing_rate=0.0, # No missing modalities in validation
        training=False
    )

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader