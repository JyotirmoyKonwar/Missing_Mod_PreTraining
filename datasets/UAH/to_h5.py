"""import os
import pickle
import numpy as np
import h5py
from sklearn.model_selection import train_test_split

# === Configuration ===
DATA_DIR = "./"  # path where the .pkl files are located
SAVE_DIR = "./"      # root directory where train/ and val/ folders will be saved
TEST_SIZE = 0.2
RANDOM_SEED = 69

def load_pickle_data(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    return data[b'dataset'], data[b'labels']

def save_to_h5(path, X, y):
    with h5py.File(path, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('y', data=y)

def main():
    # Load both datasets
    motorway_X, motorway_y = load_pickle_data(os.path.join(DATA_DIR, "motorway_dataset.pkl"))
    secondary_X, secondary_y = load_pickle_data(os.path.join(DATA_DIR, "secondary_dataset.pkl"))

    # Concatenate
    X = np.concatenate([motorway_X, secondary_X], axis=0)
    y = np.concatenate([motorway_y, secondary_y], axis=0)

    # Shuffle and split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, shuffle=True
    )

    # Create output directories
    train_dir = os.path.join(SAVE_DIR, "train")
    val_dir = os.path.join(SAVE_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Save to .h5
    save_to_h5(os.path.join(train_dir, "train_data.h5"), X_train, y_train)
    save_to_h5(os.path.join(val_dir, "val_data.h5"), X_val, y_val)

    print("Data converted and saved to:")
    print(f"  - {train_dir}/train_data.h5")
    print(f"  - {val_dir}/val_data.h5")

if __name__ == "__main__":
    main()
"""

import h5py
import os

def inspect_h5(file_path):
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return

    print(f"📂 Inspecting file: {file_path}")
    with h5py.File(file_path, 'r') as f:
        for key in f.keys():
            data = f[key]
            print(f"  - Dataset '{key}': shape = {data.shape}, dtype = {data.dtype}")

if __name__ == "__main__":
    inspect_h5("./train/train_data.h5")
    inspect_h5("./val/val_data.h5")
