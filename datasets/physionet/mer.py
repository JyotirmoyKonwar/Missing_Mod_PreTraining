import h5py
import numpy as np
import os

def merge_h5_datasets_memory_efficient(
    input_file1_path: str,
    input_file2_path: str,
    output_file_path: str,
    data_keys_to_merge: list[str],
    global_keys_to_copy: list[str] = None,
    chunk_size: int = 10000 # Number of rows to process at a time
) -> None:
    """
    Merges specified datasets from two HDF5 files into a new HDF5 file in a memory-efficient way.
    It avoids loading entire large datasets into memory by processing them in chunks.
    Optionally copies global keys (like mean/std) from the first input file.

    Args:
        input_file1_path (str): Path to the first input HDF5 file (e.g., 'train_data_1000_normalized.h5').
        input_file2_path (str): Path to the second input HDF5 file (e.g., 'val_data_1000_normalized.h5').
        output_file_path (str): Path for the new merged HDF5 file (e.g., 'Physionet_all.h5').
        data_keys_to_merge (list[str]): A list of dataset names (e.g., ['X', 'y'])
                                        that exist in both input HDF5 files and
                                        should be concatenated row-wise.
        global_keys_to_copy (list[str], optional): A list of dataset names (e.g., ['mean', 'std'])
                                                   that exist in the input HDF5 files and
                                                   should be copied as-is from the first
                                                   input file to the output file. Defaults to None.
        chunk_size (int): The number of rows to read and write at a time. Adjust based on
                          available memory. Smaller chunks use less memory but might be slower.
    """
    if global_keys_to_copy is None:
        global_keys_to_copy = []

    print(f"Attempting to merge HDF5 files (memory-efficiently):")
    print(f"  File 1: '{input_file1_path}'")
    print(f"  File 2: '{input_file2_path}'")
    print(f"Output will be saved to: '{output_file_path}'")
    print(f"Data keys to concatenate: {data_keys_to_merge}")
    print(f"Global keys to copy from File 1: {global_keys_to_copy}")
    print(f"Processing chunk size: {chunk_size} rows")
    print("\nNote: For datasets of this size, physical shuffling during merge is complex "
          "and often memory-intensive. Data will be written sequentially (File 1 then File 2). "
          "It's highly recommended to shuffle during model data loading.")


    # Check if input files exist
    if not os.path.exists(input_file1_path):
        print(f"Error: Input file '{input_file1_path}' not found.")
        return
    if not os.path.exists(input_file2_path):
        print(f"Error: Input file '{input_file2_path}' not found.")
        return

    try:
        with h5py.File(input_file1_path, 'r') as f1, \
             h5py.File(input_file2_path, 'r') as f2, \
             h5py.File(output_file_path, 'w') as f_out:

            # 1. Copy Global Keys (e.g., mean, std) from the first file
            print("\n--- Copying Global Keys ---")
            for key in global_keys_to_copy:
                if key in f1:
                    f_out.create_dataset(key, data=f1[key][()])
                    print(f"  Copied global dataset '{key}' from '{input_file1_path}'. Shape: {f1[key].shape}")
                else:
                    print(f"  Warning: Global key '{key}' not found in '{input_file1_path}'. Skipping.")

            # 2. Prepare and Copy Data Keys (e.g., X, y) in chunks
            print("\n--- Processing Data Keys ---")
            if not data_keys_to_merge:
                print("No data keys specified for merging. Skipping data concatenation.")
                return # Exit if no main data to merge

            # Determine total merged rows and create output datasets
            first_data_key = data_keys_to_merge[0]
            num_rows_f1 = f1[first_data_key].shape[0]
            num_rows_f2 = f2[first_data_key].shape[0]
            merged_total_rows = num_rows_f1 + num_rows_f2
            print(f"  Total merged rows for main datasets: {merged_total_rows}")

            output_datasets = {}
            for key in data_keys_to_merge:
                if key not in f1 or key not in f2:
                    raise KeyError(f"Dataset '{key}' missing from one of the input files.")

                # Determine the shape for the output dataset
                base_shape = f1[key].shape[1:] # All dimensions except the first (samples)
                merged_shape = (merged_total_rows,) + base_shape

                # Create the dataset in the output file
                output_datasets[key] = f_out.create_dataset(
                    key,
                    shape=merged_shape,
                    dtype=f1[key].dtype,
                    chunks=True, # Enable chunking for efficient I/O
                    compression="lzf" # Optional: LZF compression for smaller file size
                )
                print(f"  Created output dataset '{key}' with shape {merged_shape}.")

            # Copy data from file 1 to output file in chunks
            print(f"\n  Copying data from '{input_file1_path}' to '{output_file_path}'...")
            current_output_idx = 0
            for i in range(0, num_rows_f1, chunk_size):
                end_idx = min(i + chunk_size, num_rows_f1)
                for key in data_keys_to_merge:
                    output_datasets[key][current_output_idx : current_output_idx + (end_idx - i)] = f1[key][i:end_idx]
                current_output_idx += (end_idx - i)
                print(f"\r  Processed {current_output_idx}/{num_rows_f1} rows from File 1...", end='')
            print(f"\n  Finished copying from '{input_file1_path}'.")

            # Copy data from file 2 to output file in chunks
            print(f"\n  Copying data from '{input_file2_path}' to '{output_file_path}'...")
            for i in range(0, num_rows_f2, chunk_size):
                end_idx = min(i + chunk_size, num_rows_f2)
                for key in data_keys_to_merge:
                    output_datasets[key][current_output_idx : current_output_idx + (end_idx - i)] = f2[key][i:end_idx]
                current_output_idx += (end_idx - i)
                print(f"\r  Processed {i + (end_idx - i)}/{num_rows_f2} rows from File 2...", end='') # Report progress for f2
            print(f"\n  Finished copying from '{input_file2_path}'.")

        print("\nMerging process completed successfully!")

    except KeyError as e:
        print(f"Error: Dataset '{e}' not found in one of the HDF5 files. "
              "Please check the 'data_keys_to_merge' and 'global_keys_to_copy' lists "
              "and ensure they match the dataset names in your .h5 files.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration for your Physionet Dataset ---
    TRAIN_FILE = 'train_data_1000_normalized.h5'
    VAL_FILE = 'val_data_1000_normalized.h5'
    MERGED_FILE = 'Physionet_all.h5'

    # These are the datasets that will be concatenated row-wise
    DATA_KEYS_TO_MERGE = ['X', 'y']

    # These are the datasets that will be copied as-is from the FIRST file
    GLOBAL_KEYS_TO_COPY = ['mean', 'std']

    # Adjust chunk_size based on your system's available RAM.
    # 10000 rows of X (1000, 6, float32) is approx 10000 * 1000 * 6 * 4 bytes = 240 MB.
    # This should be well within typical RAM limits for a single chunk.
    CHUNK_SIZE = 10000

    # --- Run the merge function ---
    merge_h5_datasets_memory_efficient(
        input_file1_path=TRAIN_FILE,
        input_file2_path=VAL_FILE,
        output_file_path=MERGED_FILE,
        data_keys_to_merge=DATA_KEYS_TO_MERGE,
        global_keys_to_copy=GLOBAL_KEYS_TO_COPY,
        chunk_size=CHUNK_SIZE
    )

    # --- Example of how to verify the merged file (Optional) ---
    print("\n--- Verifying the merged file (Optional) ---")
    if os.path.exists(MERGED_FILE):
        try:
            with h5py.File(MERGED_FILE, 'r') as f_merged:
                print(f"Datasets in '{MERGED_FILE}': {list(f_merged.keys())}")
                if 'X' in f_merged:
                    print(f"  Dataset 'X' shape: {f_merged['X'].shape}")
                if 'y' in f_merged:
                    print(f"  Dataset 'y' shape: {f_merged['y'].shape}")
                if 'mean' in f_merged:
                    print(f"  Dataset 'mean' shape: {f_merged['mean'].shape}")
                if 'std' in f_merged:
                    print(f"  Dataset 'std' shape: {f_merged['std'].shape}")
        except Exception as e:
            print(f"Error verifying merged file: {e}")
    else:
        print(f"Merged file '{MERGED_FILE}' was not created.")
