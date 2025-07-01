# --- utils.py ---

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import configs

class MissingModalityDataset(Dataset):
    """Initializes lazily to be compatible with multi-worker loading."""
    def __init__(self, uah_h5_path, physio_h5_path, indices):
        self.uah_h5_path = uah_h5_path
        self.physio_h5_path = physio_h5_path
        self.indices = indices
        # Files are opened in __getitem__ if not already opened
        self.uah_file, self.physio_file = None, None
        self.uah_x, self.physio_x, self.labels = None, None, None

    def _init_files(self):
        """Opens HDF5 files."""
        self.uah_file = h5py.File(self.uah_h5_path, 'r')
        self.physio_file = h5py.File(self.physio_h5_path, 'r')
        self.uah_x = self.uah_file['X']
        self.physio_x = self.physio_file['X']
        self.labels = self.uah_file['y']

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.uah_file is None: self._init_files()
            
        data_idx = self.indices[idx]
        uah_seq = torch.from_numpy(self.uah_x[data_idx].astype(np.float32))
        physio_seq = torch.from_numpy(self.physio_x[data_idx].astype(np.float32))
        label = torch.tensor(self.labels[data_idx], dtype=torch.long).squeeze()
        return uah_seq, physio_seq, label

def stratified_collate_fn(batch):
    """
    Custom collate function to create missing modality scenarios for training.
    It splits a batch into three parts: UAH-only, Physio-only, and Both.
    """
    uah_seqs, physio_seqs, labels = zip(*batch)
    uah_batch, physio_batch = torch.stack(uah_seqs, 0), torch.stack(physio_seqs, 0)
    labels_batch = torch.stack(labels, 0)
    
    batch_size = len(labels)
    mask = torch.ones(batch_size, 2, dtype=torch.bool)
    
    # Create missing modality scenarios
    one_third, two_thirds = batch_size // 3, 2 * (batch_size // 3)
    
    # First third: UAH only (mask out Physio)
    mask[:one_third, 1] = False
    physio_batch[:one_third].zero_() 
    
    # Second third: Physio only (mask out UAH)
    mask[one_third:two_thirds, 0] = False
    uah_batch[one_third:two_thirds].zero_()
    
    # Final third has both modalities (mask remains [True, True])
    return uah_batch, physio_batch, mask, labels_batch

def get_dataloaders():
    """
    Splits the data into training, validation, and test sets and returns their DataLoaders.
    """
    try:
        with h5py.File(configs.UAH_DATA_PATH, 'r') as f:
            num_samples = len(f['X'])
    except FileNotFoundError as e:
        print(f"FATAL: Could not find data file. Update paths in configs.py. Details: {e}")
        exit()

    indices = list(range(num_samples))
    if configs.SHUFFLE_DATASET:
        np.random.seed(configs.RANDOM_SEED)
        np.random.shuffle(indices)

    # --- UPDATED: 80/10/10 Train/Validation/Test Split ---
    val_split_point = int(np.floor(configs.VALIDATION_SPLIT * num_samples))
    test_split_point = val_split_point + int(np.floor(configs.TEST_SPLIT * num_samples))
    
    val_indices = indices[:val_split_point]
    test_indices = indices[val_split_point:test_split_point]
    train_indices = indices[test_split_point:]

    print(f"Dataset split: {len(train_indices)} training, {len(val_indices)} validation, {len(test_indices)} testing samples.")

    # Create dataset instances
    train_dataset = MissingModalityDataset(configs.UAH_DATA_PATH, configs.PHYSIO_DATA_PATH, train_indices)
    val_dataset = MissingModalityDataset(configs.UAH_DATA_PATH, configs.PHYSIO_DATA_PATH, val_indices)
    test_dataset = MissingModalityDataset(configs.UAH_DATA_PATH, configs.PHYSIO_DATA_PATH, test_indices)

    # --- UPDATED: Create DataLoaders without num_workers ---
    train_loader = DataLoader(
        train_dataset, 
        batch_size=configs.BATCH_SIZE, 
        shuffle=True,
        pin_memory=True, 
        collate_fn=stratified_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=configs.BATCH_SIZE, 
        shuffle=False,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=configs.BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader