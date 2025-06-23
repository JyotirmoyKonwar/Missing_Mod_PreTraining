# --- utils.py ---

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import configs

class MissingModalityDataset(Dataset):
    def __init__(self, uah_data, physio_data, labels, missing_rate=0.0, training=True):
        self.uah_data = uah_data
        self.physio_data = physio_data
        self.labels = labels
        self.missing_rate = missing_rate
        self.training = training
        self.length = min(len(uah_data), len(physio_data), len(labels))
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        uah_seq = torch.from_numpy(self.uah_data[idx].astype(np.float32))
        physio_seq = torch.from_numpy(self.physio_data[idx].astype(np.float32))
        label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()
        
        modality_mask = torch.tensor([True, True])
        
        if self.training and np.random.random() < self.missing_rate:
            missing_modality = np.random.choice([0, 1])
            modality_mask[missing_modality] = False
            
        return uah_seq, physio_seq, modality_mask, label

def get_dataloaders():
    # This is a placeholder for your actual data loading logic
    # TODO: Replace this with your h5py data loading
    print("Warning: Using dummy data. Replace with actual data loading in utils.py")
    num_train, num_val = 1000, 200
    uah_train = np.random.randn(num_train, 64, 9)
    physio_train = np.random.randn(num_train, 1000, 6)
    train_labels = np.random.randint(0, configs.NUM_CLASSES, num_train)
    
    uah_val = np.random.randn(num_val, 64, 9)
    physio_val = np.random.randn(num_val, 1000, 6)
    val_labels = np.random.randint(0, configs.NUM_CLASSES, num_val)

    train_dataset = MissingModalityDataset(
        uah_train, physio_train, train_labels, 
        missing_rate=configs.STAGE1_MISSING_RATE, training=True
    )
    val_dataset = MissingModalityDataset(
        uah_val, physio_val, val_labels,
        missing_rate=0.0, training=False # No missing modalities in validation
    )
    
    train_loader = DataLoader(train_dataset, batch_size=configs.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=configs.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader