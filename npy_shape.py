import numpy as np
import os

def analyze_large_npy_file(file_path):
    """
    Instantly reads the shape and field names of a .npy file using memory mapping.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    try:
        # mmap_mode='r' is the magic key here. 
        # It reads metadata instantly without loading the array into RAM.
        data = np.load(file_path, mmap_mode='r', allow_pickle=False)
        
        print(f"--- Fast Analysis for: {file_path} ---")
        
        # 1. Print the shape
        print(f"Shape: {data.shape}")
        
        # 2. Check for and print field names
        if data.dtype.names is not None:
            print(f"Field Names: {data.dtype.names}")
        else:
            print("Field Names: None (Standard array)")
            
        print("-" * 38)
        
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    sample_file = 'your_massive_file.npy'
    analyze_large_npy_file(sample_file)