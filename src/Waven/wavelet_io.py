import numpy as np
import zarr
from numcodecs import Blosc
from tqdm import trange
import os

def convert_npy_to_zarr(npy_dir, zarr_dir, hz):
    """
    Converts downsampled wavelet .npy files to chunked, compressed Zarr format
    using configurations derived dynamically from the project JSON settings.
    
    Parameters:
        npy_dir (str): Directory where the .npy files are located (e.g., your zebra_movie folder)
        zarr_dir (str): Directory where the .zarr files should be saved (e.g., your zarr folder)
        hz (int): Acquisition frequency of the video data. Default is 30.
    """
    # Dynamically build paths
    npy_i = os.path.join(npy_dir, "dwt_videodata2_i.npy")
    npy_r = os.path.join(npy_dir, "dwt_videodata2_r.npy")

    zarr_i = os.path.join(zarr_dir, "dwt_videodata2_i.zarr")
    zarr_r = os.path.join(zarr_dir, "dwt_videodata2_r.zarr")

    SECONDS_PER_MINUTE = 60
    FRAMES_PER_MINUTE = hz * SECONDS_PER_MINUTE
    chunks = (FRAMES_PER_MINUTE, 1, 1, 8, 5, 4)

    compressor = Blosc(
        cname="zstd",
        clevel=3,
        shuffle=Blosc.BITSHUFFLE,
    )

    # ======================
    # LOAD NPY AS MEMMAP
    # ======================
    print(f"Opening NPY files as memmap from: {npy_dir}")
    if not os.path.exists(npy_i) or not os.path.exists(npy_r):
        raise FileNotFoundError(
            f"Could not find the required .npy files ('dwt_videodata2_i.npy' or 'dwt_videodata2_r.npy') in {npy_dir}"
        )

    w_i = np.load(npy_i, mmap_mode="r")
    w_r = np.load(npy_r, mmap_mode="r")

    assert w_i.shape == w_r.shape
    shape = w_i.shape
    dtype = w_i.dtype

    print("Shape:", shape)
    print("Dtype:", dtype)

    # Ensure output zarr directory exists
    os.makedirs(zarr_dir, exist_ok=True)

    # ======================
    # CREATE ZARR ARRAYS
    # ======================
    print(f"Creating Zarr arrays at: {zarr_dir}")

    z_i = zarr.open(
        zarr_i,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )

    z_r = zarr.open(
        zarr_r,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )

    # ======================
    # COPY DATA BY TIME CHUNKS
    # ======================
    t_chunk = chunks[0]
    nT = shape[0]

    print("Converting...")
    for t in trange(0, nT, t_chunk, desc="Time chunks"):
        t1 = min(t + t_chunk, nT)

        # Important: slice only time, rest stays contiguous
        z_i[t:t1] = w_i[t:t1]
        z_r[t:t1] = w_r[t:t1]

    print("✅ Conversion finished successfully.")
    print("Zarr files written:")
    print(zarr_i)
    print(zarr_r)


# Example usage block if run directly as a script
if __name__ == "__main__":
    # This acts as a fallback default, but you should pass these values from your config loader
    import json
    
    print("This file is now a module function. Call `convert_npy_to_zarr(npy_dir, zarr_dir, hz)` from your pipeline.")