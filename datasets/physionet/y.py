import h5py
import numpy as np
import os

def check_unique_labels_in_h5(file_path: str, label_dataset_name: str = 'y', chunk_size: int = 10000) -> None:
    """
    Checks and prints all unique labels present in a specified HDF5 dataset.

    Args:
        file_path (str): Path to the HDF5 file (e.g., 'Physionet_all.h5').
        label_dataset_name (str): The name of the dataset containing labels (default is 'y').
        chunk_size (int): The number of labels to read at a time.
                          Adjust based on available memory and file size.
    """
    print(f"Checking unique labels in '{label_dataset_name}' from file: '{file_path}'")
    print(f"Reading in chunks of {chunk_size} labels.")

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    unique_labels = set()
    total_labels_processed = 0

    try:
        with h5py.File(file_path, 'r') as f:
            if label_dataset_name not in f:
                print(f"Error: Dataset '{label_dataset_name}' not found in '{file_path}'.")
                print(f"Available datasets: {list(f.keys())}")
                return

            label_dataset = f[label_dataset_name]
            num_labels = label_dataset.shape[0]
            print(f"Found dataset '{label_dataset_name}' with total {num_labels} labels.")

            for i in range(0, num_labels, chunk_size):
                end_idx = min(i + chunk_size, num_labels)
                labels_chunk = label_dataset[i:end_idx]
                unique_labels.update(np.unique(labels_chunk))
                total_labels_processed += (end_idx - i)
                print(f"\rProcessed {total_labels_processed}/{num_labels} labels...", end='')

            print(f"\n\n--- Analysis Complete ---")
            print(f"Found {len(unique_labels)} unique labels.")
            print(f"Unique labels: {sorted(list(unique_labels))}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    # Choose which HDF5 file you want to inspect:
    # Option 1: Your original training file
    # H5_FILE_TO_CHECK = 'train_data_1000_normalized.h5'
    # Option 2: Your original validation file
    # H5_FILE_TO_CHECK = 'val_data_1000_normalized.h5'
    # Option 3: Your newly merged file
    H5_FILE_TO_CHECK = 'Physionet_all.h5' # <-- Set this to the file you want to check

    LABEL_DATASET_NAME = 'y' # Based on your previous output, 'y' is the correct key for labels

    # Adjust chunk_size if needed, but 10000 is a good starting point.
    # For a 1D array of 582690 labels, this is very efficient.
    CHUNK_SIZE = 10000

    # --- Run the check function ---
    check_unique_labels_in_h5(
        file_path=H5_FILE_TO_CHECK,
        label_dataset_name=LABEL_DATASET_NAME,
        chunk_size=CHUNK_SIZE
    )
