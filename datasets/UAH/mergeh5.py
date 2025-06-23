import h5py
import numpy as np
import os

def merge_h5_datasets_uah(
    input_file1_path: str,
    input_file2_path: str,
    output_file_path: str,
    shuffle_data: bool = True
) -> None:
    """
    Merges 'X' and 'y' from two UAH HDF5 files into a new HDF5 file.

    Args:
        input_file1_path (str): Path to the first input HDF5 file (e.g., 'train_data.h5').
        input_file2_path (str): Path to the second input HDF5 file (e.g., 'val_data.h5').
        output_file_path (str): Path for the new merged HDF5 file.
        shuffle_data (bool): If True, shuffles the merged data row-wise consistently
                             across both 'X' and 'y'. Defaults to True.
    """
    # Specific dataset names for UAH based on your verification
    # These MUST match the keys found in your .h5 files.
    dataset_names = ['X', 'y']

    print(f"Attempting to merge UAH dataset files:")
    print(f"  File 1: '{input_file1_path}'")
    print(f"  File 2: '{input_file2_path}'")
    print(f"Output will be saved to: '{output_file_path}'")
    print(f"Datasets to merge: {dataset_names}")
    print(f"Shuffle data: {shuffle_data}")

    # Check if input files exist
    if not os.path.exists(input_file1_path):
        print(f"Error: Input file '{input_file1_path}' not found.")
        return
    if not os.path.exists(input_file2_path):
        print(f"Error: Input file '{input_file2_path}' not found.")
        return

    try:
        # Load data from the first HDF5 file
        with h5py.File(input_file1_path, 'r') as f1:
            # Accessing datasets using the correct names 'X' and 'y'
            X1 = f1['X'][()]
            y1 = f1['y'][()]
            print(f"Loaded from '{input_file1_path}':")
            print(f"  'X' shape: {X1.shape}")
            print(f"  'y' shape: {y1.shape}")

        # Load data from the second HDF5 file
        with h5py.File(input_file2_path, 'r') as f2:
            # Accessing datasets using the correct names 'X' and 'y'
            X2 = f2['X'][()]
            y2 = f2['y'][()]
            print(f"Loaded from '{input_file2_path}':")
            print(f"  'X' shape: {X2.shape}")
            print(f"  'y' shape: {y2.shape}")

        # Basic shape validation (optional but good practice)
        if X1.shape[0] != y1.shape[0] or \
           X2.shape[0] != y2.shape[0]:
            print("Warning: Mismatch in number of rows between 'X' and 'y' within one of the files.")

        # Concatenate data from both files
        merged_X = np.concatenate((X1, X2), axis=0)
        merged_y = np.concatenate((y1, y2), axis=0)
        print(f"Concatenated 'X'. New shape: {merged_X.shape}")
        print(f"Concatenated 'y'. New shape: {merged_y.shape}")

        # Shuffle data if requested
        if shuffle_data:
            print("Shuffling data...")
            # Get the total number of rows in the merged dataset
            total_rows = merged_X.shape[0] # Can use either X or y for row count
            # Create a permutation of indices
            shuffled_indices = np.random.permutation(total_rows)

            # Apply the same shuffling to both merged datasets
            merged_X = merged_X[shuffled_indices]
            merged_y = merged_y[shuffled_indices]
            print("Data shuffled successfully.")

        # Save the merged (and optionally shuffled) data to the new HDF5 file
        with h5py.File(output_file_path, 'w') as f_out:
            # Saving with the correct names 'X' and 'y'
            f_out.create_dataset('X', data=merged_X)
            f_out.create_dataset('y', data=merged_y)
            print(f"Saved dataset 'X' to '{output_file_path}' with shape {merged_X.shape}")
            print(f"Saved dataset 'y' to '{output_file_path}' with shape {merged_y.shape}")

        print("\nMerging process completed successfully!")

    except KeyError as e:
        print(f"Error: Expected dataset '{e}' not found in one of the HDF5 files. "
              "Please ensure your files contain 'X' and 'y' as top-level datasets.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # --- Configuration for UAH Dataset ---
    TRAIN_FILE = 'train_data.h5' # Adjusted based on your provided output
    VAL_FILE = 'val_data.h5'     # Adjusted based on your provided output
    MERGED_FILE = 'UAH_data_all.h5' # Adjusted based on your provided output
    SHUFFLE_RESULT = True # Set to True to randomize, False to keep original order

    # --- Run the merge function ---
    merge_h5_datasets_uah(
        input_file1_path=TRAIN_FILE,
        input_file2_path=VAL_FILE,
        output_file_path=MERGED_FILE,
        shuffle_data=SHUFFLE_RESULT
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
        except Exception as e:
            print(f"Error verifying merged file: {e}")
    else:
        print(f"Merged file '{MERGED_FILE}' was not created.")

