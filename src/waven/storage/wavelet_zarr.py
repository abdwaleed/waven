"""Wavelet Zarr module."""
import os
import time

import numpy as np

from ..runtime.task_control import check_cancelled, progress_message


def _open_zarr_array(zarr_module, path, **kwargs):
    """Open a Zarr array across Zarr 2/3 compressor API differences."""
    try:
        return zarr_module.open(path, **kwargs)
    except TypeError:
        compressor = kwargs.pop("compressor", None)
        if compressor is not None:
            kwargs["compressors"] = [compressor]
            try:
                return zarr_module.open(path, **kwargs)
            except Exception:
                kwargs.pop("compressors", None)
        return zarr_module.open(path, **kwargs)


def convert_npy_to_zarr(
    npy_dir,
    zarr_dir,
    hz,
    n_orientations=None,
    n_sigmas=None,
    n_frequencies=None,
    delete_source=False,
    cancel_event=None,
):
    """
    Convert full-model wavelet ``.npy`` files to chunked, compressed Zarr.

    Parameters:
        npy_dir (str): Directory containing ``dwt_videodata2_i/r.npy``.
        zarr_dir (str): Directory where ``.zarr`` files should be saved.
        hz (int): Acquisition frequency of the video data.
        n_orientations (int, optional): Number of Gabor orientations.
        n_sigmas (int, optional): Number of sigma scales.
        n_frequencies (int, optional): Number of spatial frequencies.
        delete_source (bool): Delete converted source ``.npy`` files after success.
        cancel_event: Optional threading event used for cooperative cancellation.
    """
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ImportError(
            "Zarr wavelet output requires the 'zarr' and 'numcodecs' packages. "
            "Install project requirements or select 'npy' as the full-model format."
        ) from exc

    npy_i = os.path.join(npy_dir, "dwt_videodata2_i.npy")
    npy_r = os.path.join(npy_dir, "dwt_videodata2_r.npy")
    zarr_i = os.path.join(zarr_dir, "dwt_videodata2_i.zarr")
    zarr_r = os.path.join(zarr_dir, "dwt_videodata2_r.zarr")

    frames_per_minute = int(hz) * 60

    print(f"Opening NPY files as memmap from: {npy_dir}")
    if not os.path.exists(npy_i) or not os.path.exists(npy_r):
        raise FileNotFoundError(
            "Could not find the required .npy files "
            "('dwt_videodata2_i.npy' or 'dwt_videodata2_r.npy') in "
            f"{npy_dir}"
        )

    w_i = np.load(npy_i, mmap_mode="r")
    w_r = np.load(npy_r, mmap_mode="r")

    if w_i.shape != w_r.shape:
        raise ValueError(f"Wavelet NPY shapes differ: {w_i.shape} vs {w_r.shape}")
    shape = w_i.shape
    dtype = w_i.dtype

    if n_orientations is None:
        n_orientations = shape[3]
    if n_sigmas is None:
        n_sigmas = shape[4]
    if n_frequencies is None:
        n_frequencies = shape[5]

    chunks = (
        min(shape[0], max(1, frames_per_minute)),
        1,
        1,
        min(shape[3], int(n_orientations)),
        min(shape[4], int(n_sigmas)),
        min(shape[5], int(n_frequencies)),
    )

    print("Shape:", shape)
    print("Dtype:", dtype)
    print("Chunks:", chunks)

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    os.makedirs(zarr_dir, exist_ok=True)

    print(f"Creating Zarr arrays at: {zarr_dir}")
    z_i = _open_zarr_array(
        zarr,
        zarr_i,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )
    z_r = _open_zarr_array(
        zarr,
        zarr_r,
        mode="w",
        shape=shape,
        chunks=chunks,
        dtype=dtype,
        compressor=compressor,
    )

    t_chunk = chunks[0]
    n_frames = shape[0]
    total_chunks = max(1, (n_frames + t_chunk - 1) // t_chunk)
    completed_chunks = 0
    progress_start = time.time()

    print("Converting...")
    for t in range(0, n_frames, t_chunk):
        check_cancelled(cancel_event)
        t1 = min(t + t_chunk, n_frames)
        z_i[t:t1] = w_i[t:t1]
        z_r[t:t1] = w_r[t:t1]
        completed_chunks += 1
        print(
            progress_message(
                "NPY to Zarr conversion",
                completed_chunks,
                total_chunks,
                progress_start,
                unit="chunks",
            )
        )

    print("Conversion finished successfully.")
    print("Zarr files written:")
    print(zarr_i)
    print(zarr_r)

    if delete_source:
        del w_i, w_r
        for source_path in (npy_i, npy_r):
            try:
                os.remove(source_path)
                print(f"Removed converted NPY source: {source_path}")
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"Could not remove converted NPY source {source_path}: {exc}")


if __name__ == "__main__":
    print("This file is a module. Call convert_npy_to_zarr(npy_dir, zarr_dir, hz).")
