import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import config

class H5Dataset(Dataset):
    """
    A custom PyTorch dataset to load data from an HDF5 file.
    It assumes the HDF5 file contains 'X' and 'y' datasets.
    """
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.x_data = self.h5_file['X']
        self.y_data = self.h5_file['y']

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.tensor(self.x_data[idx], dtype=torch.float32)
        y = torch.tensor(self.y_data[idx], dtype=torch.long).squeeze()
        return x, y

    def close(self):
        self.h5_file.close()

def get_data_loaders():
    """
    Creates and returns training and validation data loaders using paths from config.
    """
    train_dataset = H5Dataset(config.TRAIN_PATH)
    val_dataset = H5Dataset(config.VAL_PATH)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader