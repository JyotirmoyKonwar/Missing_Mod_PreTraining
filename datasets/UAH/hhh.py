import h5py
import os

file_path = 'val_data.h5' # Or 'val_data.h5'

if os.path.exists(file_path):
    with h5py.File(file_path, 'r') as f:
        print(f"Keys in {file_path}: {list(f.keys())}")
else:
    print(f"File not found: {file_path}")