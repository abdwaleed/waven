"""Coarse wavelet cache loading and construction.

This module owns the legacy ``dwt_videodata_*`` and
``dwt_downsampled_videodata.npy`` paths used by coarse RF analysis and
``run_Model``. Full-model wavelet loading lives in
:mod:`waven.stimulus.full_model`.
"""
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import numpy as np
import torch
import torch.nn.functional as F
from skimage import transform

from ..config import coarse_grid_dimensions
from ..runtime.performance import coarse_wavelet_chunk_size_gpu_or_cpu, get_gpu_count, has_enough_ram
from ..runtime.task_control import check_cancelled, progress_message
from ..storage.array_store import load_array


def _ensure_trailing_sep(path):
    """Return ``path`` with a trailing path separator for legacy string joins."""
    path = str(path)
    if not path.endswith(("/", "\\")):
        path = path + os.sep
    return path


def _wavelet_file(path, filename):
    """Return a path pointing to either a .npy file or a sibling .zarr store."""
    candidate = os.path.join(path, filename)
    if os.path.exists(candidate):
        return candidate

    if filename.endswith(".npy"):
        sibling = candidate[:-4] + ".zarr"
        if os.path.isdir(sibling):
            return sibling

    return candidate


def _coerce_legacy_coarse_wavelets(wavelets):
    """Return coarse wavelets in the 5D layout consumed by run_Model."""
    if wavelets.ndim == 5:
        return wavelets
    if wavelets.ndim == 6 and wavelets.shape[-1] == 1:
        return wavelets[..., 0]
    raise ValueError(
        "run_Model-compatible coarse wavelets must have shape "
        "(time, nx, ny, n_thetas, n_sigmas). Multiple frequency planes belong "
        "in dwt_videodata2_{r,i}, not dwt_downsampled_videodata."
    )


def _validate_legacy_coarse_shape(wavelets, nx, ny, no, ns, label):
    expected = (nx, ny, no, ns)
    if wavelets.ndim != 5 or wavelets.shape[1:5] != expected:
        raise ValueError(
            f"{label} has shape {wavelets.shape}; expected "
            f"(time, {nx}, {ny}, {no}, {ns}) for legacy run_Model."
        )


def _expected_coarse_shape(nx0, ny0, no, ns, nf):
    coarse_nx, coarse_ny = coarse_grid_dimensions(nx0, ny0)
    return coarse_nx, coarse_ny, (coarse_nx, coarse_ny, no, ns, nf)


def load_stimulus_simple_cell(
    path,
    nx,
    ny,
    no,
    ns,
    nf=1,
    downsampling=False,
    nx0=None,
    ny0=None,
):
    """Load coarse-analysis wavelets from ``dwt_videodata_{0,1}.npy``.

    Source arrays have shape ``(T, nx0, ny0, n_orientations, n_sigmas)`` or
    ``(T, nx0, ny0, n_orientations, n_sigmas, n_frequencies)``. When
    ``downsampling`` is True, spatial axes are resized to ``(nx, ny)``.
    """
    path = _ensure_trailing_sep(path)
    wavelets_r = load_array(_wavelet_file(path, "dwt_videodata_0.npy"), mmap_mode="r")
    wavelets_i = load_array(_wavelet_file(path, "dwt_videodata_1.npy"), mmap_mode="r")
    print("Loaded coarse wavelets:", wavelets_r.shape, wavelets_i.shape)

    if wavelets_r.shape != wavelets_i.shape:
        raise ValueError(
            f"Coarse real/imag wavelets must share the same shape, got "
            f"{wavelets_r.shape} and {wavelets_i.shape}"
        )

    if wavelets_r.ndim not in (5, 6):
        raise ValueError(
            f"Unsupported coarse wavelet shape: {wavelets_r.shape}. "
            "Expected 5D or 6D time-first arrays."
        )

    # Legacy compatibility: some saved files encode time at axis 3.
    if wavelets_r.shape[3] > wavelets_r.shape[0]:
        wavelets_r = np.moveaxis(wavelets_r, 3, 0)
        wavelets_i = np.moveaxis(wavelets_i, 3, 0)
        print("Converted coarse wavelets to time-first ordering:", wavelets_r.shape)

    if nx0 is None:
        nx0 = wavelets_r.shape[1]
    if ny0 is None:
        ny0 = wavelets_r.shape[2]

    if wavelets_r.shape[3] != no or wavelets_i.shape[3] != no:
        raise ValueError(
            f"Expected {no} orientations, got {wavelets_r.shape[3]} and "
            f"{wavelets_i.shape[3]} in dwt_videodata_0/1."
        )
    if wavelets_r.shape[4] != ns or wavelets_i.shape[4] != ns:
        raise ValueError(
            f"Expected {ns} sigmas, got {wavelets_r.shape[4]} and "
            f"{wavelets_i.shape[4]} in dwt_videodata_0/1."
        )
    if wavelets_r.ndim == 6 and wavelets_r.shape[5] != nf:
        raise ValueError(
            f"Expected {nf} frequencies, got {wavelets_r.shape[5]} in "
            "dwt_videodata_0/1."
        )

    if downsampling:
        target = (wavelets_r.shape[0], nx, ny, no, ns)
        if wavelets_r.ndim == 6:
            target = (wavelets_r.shape[0], nx, ny, no, ns, nf)
        wavelets_r = transform.resize(
            wavelets_r,
            target,
            anti_aliasing=True,
        )
        wavelets_i = transform.resize(
            wavelets_i,
            target,
            anti_aliasing=True,
        )

    return wavelets_r, wavelets_i


def load_stimulus_simple_cell2_i(
    path="/media/sophie/Expansion1/UCL/datatest/",
    tt=None,
    downsampling=False,
    nx0=None,
    ny0=None,
    no=None,
    ns=None,
    nf=None,
):
    """Load imaginary full-model wavelets from ``dwt_videodata2_i.npy``."""
    path = _ensure_trailing_sep(path)
    wavelets_i = load_array(_wavelet_file(path, "dwt_videodata2_i.npy"), mmap_mode="r")
    if tt is not None:
        wavelets_i = wavelets_i[tt[0]:tt[1]]
    if downsampling:
        if nx0 is None or ny0 is None or no is None or ns is None or nf is None:
            raise ValueError(
                "downsampling full-model wavelets requires nx0, ny0, no, ns, and nf"
            )
        nx, ny, _ = _expected_coarse_shape(nx0, ny0, no, ns, nf)
        duration = wavelets_i.shape[0]
        wavelets_i = transform.resize(
            wavelets_i,
            (duration, nx, ny, no, ns, nf),
            anti_aliasing=True,
        )
    return wavelets_i


def load_stimulus_simple_cell2_r(
    path="/media/sophie/Expansion1/UCL/datatest/",
    tt=None,
    downsampling=False,
    nx0=None,
    ny0=None,
    no=None,
    ns=None,
    nf=None,
):
    """Load real full-model wavelets from ``dwt_videodata2_r.npy``."""
    path = _ensure_trailing_sep(path)
    wavelets_r = load_array(_wavelet_file(path, "dwt_videodata2_r.npy"), mmap_mode="r")
    if tt is not None:
        wavelets_r = wavelets_r[tt[0]:tt[1]]
    if downsampling:
        if nx0 is None or ny0 is None or no is None or ns is None or nf is None:
            raise ValueError(
                "downsampling full-model wavelets requires nx0, ny0, no, ns, and nf"
            )
        nx, ny, _ = _expected_coarse_shape(nx0, ny0, no, ns, nf)
        duration = wavelets_r.shape[0]
        wavelets_r = transform.resize(
            wavelets_r,
            (duration, nx, ny, no, ns, nf),
            anti_aliasing=True,
        )
    return wavelets_r


def load_stimulus_simple_cell2(
    path="/media/sophie/Expansion1/UCL/datatest/",
    tt=None,
    downsampling=False,
    nx0=None,
    ny0=None,
    no=None,
    ns=None,
    nf=None,
):
    w_i = load_stimulus_simple_cell2_i(
        path,
        tt,
        downsampling,
        nx0=nx0,
        ny0=ny0,
        no=no,
        ns=ns,
        nf=nf,
    )
    w_r = load_stimulus_simple_cell2_r(
        path,
        tt,
        downsampling,
        nx0=nx0,
        ny0=ny0,
        no=no,
        ns=ns,
        nf=nf,
    )
    return w_r, w_i






def coarseWavelet(
    path,
    downsampling,
    nx0,
    ny0,
    no,
    ns,
    nf,
    nx=None,
    ny=None,
    chunk_size=None,
    cancel_event=None,
):
    """Load or build spatially downsampled wavelets for receptive-field analysis."""
    
    if nx is None:
        nx = round(nx0 * 0.20)
    if ny is None:
        ny = round(ny0 * 0.20)
        
    print(f"Original dims: {nx0}x{ny0} | Coarse dims (20%): {nx}x{ny}")

    if chunk_size is None:
        chunk_size = coarse_wavelet_chunk_size_gpu_or_cpu(nx0=nx0, ny0=ny0, no=no, ns=ns, nf=nf)
        print(f"Hardware-aware chunk size set to: {chunk_size} frames per batch")

    cache_path = os.path.join(path, "dwt_downsampled_videodata.npy")
    print("Loading wavelets...")
    if os.path.exists(cache_path):
        print("Already downsampled. Loading from cache.")
        wavelets_downsampled = np.load(cache_path, mmap_mode="r")
        if wavelets_downsampled.shape[0] != 3:
            raise ValueError(
                "Cached coarse wavelets must contain real, imaginary, and "
                f"complex arrays on axis 0; got {wavelets_downsampled.shape}. "
                "Delete dwt_downsampled_videodata.npy and regenerate."
            )
        w_r_cached = _coerce_legacy_coarse_wavelets(wavelets_downsampled[0])
        w_i_cached = _coerce_legacy_coarse_wavelets(wavelets_downsampled[1])
        w_c_cached = _coerce_legacy_coarse_wavelets(wavelets_downsampled[2])
        _validate_legacy_coarse_shape(w_r_cached, nx, ny, no, ns, "Cached real wavelets")
        _validate_legacy_coarse_shape(w_i_cached, nx, ny, no, ns, "Cached imaginary wavelets")
        _validate_legacy_coarse_shape(w_c_cached, nx, ny, no, ns, "Cached complex wavelets")
        return w_r_cached, w_i_cached, w_c_cached

    print("Beginning downsampling...")
    wavelets_r, wavelets_i = load_stimulus_simple_cell(
        path, nx, ny, no, ns, nf, downsampling
    )
    
    wavelets_r = _coerce_legacy_coarse_wavelets(wavelets_r)
    wavelets_i = _coerce_legacy_coarse_wavelets(wavelets_i)
    source_nx, source_ny = wavelets_r.shape[1:3]
    n_frames = wavelets_r.shape[0]
    n_chunks = math.ceil(n_frames / chunk_size)

    target_shape_full = (n_frames, nx, ny, no, ns)
    bytes_per_array = math.prod(target_shape_full) * 4
    total_required = bytes_per_array * 3
    use_memmap = not has_enough_ram(total_required, safety_margin=1.20)

    if use_memmap:
        print("Low RAM detected: using disk-backed arrays for coarse wavelet cache.")
        mmap_r = os.path.join(path, "dwt_r_downsampled.mmap")
        mmap_i = os.path.join(path, "dwt_i_downsampled.mmap")
        mmap_c = os.path.join(path, "dwt_c_downsampled.mmap")
        w_r_downsampled = np.lib.format.open_memmap(mmap_r, mode="w+", dtype=np.float32, shape=target_shape_full)
        w_i_downsampled = np.lib.format.open_memmap(mmap_i, mode="w+", dtype=np.float32, shape=target_shape_full)
        w_c_downsampled = np.lib.format.open_memmap(mmap_c, mode="w+", dtype=np.float32, shape=target_shape_full)
    else:
        print(f"Using RAM for coarse wavelet cache working arrays ({total_required / (1024**3):.2f} GB).")
        w_r_downsampled = np.empty(target_shape_full, dtype=np.float32)
        w_i_downsampled = np.empty(target_shape_full, dtype=np.float32)
        w_c_downsampled = np.empty(target_shape_full, dtype=np.float32)

    num_gpus = get_gpu_count()
    # Use 1 worker per GPU, or 1 worker total if falling back to pure CPU
    max_workers = max(1, num_gpus)

    def process_chunk(chunk_index):
        """Worker function to process a single chunk on a specific device."""
        check_cancelled(cancel_event)
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, n_frames)
        chunk_len = end - start
        
        w_r_slice = wavelets_r[start:end]
        w_i_slice = wavelets_i[start:end]

        _, _, _, actual_no, actual_ns = w_r_slice.shape

        success = False
        error_msg = ""

        # --- ATTEMPT 1: GPU ACCELERATION ---
        if num_gpus > 0:
            device_id = chunk_index % num_gpus
            device = f"cuda:{device_id}"
            
            try:
                with torch.no_grad():
                    w_r_clean = np.ascontiguousarray(w_r_slice).copy()
                    w_i_clean = np.ascontiguousarray(w_i_slice).copy()

                    w_r_gpu = torch.from_numpy(w_r_clean).to(device).float()
                    w_i_gpu = torch.from_numpy(w_i_clean).to(device).float()
                    w_c_gpu = w_r_gpu.square() + w_i_gpu.square()

                    def resize_tensor_gpu(t):
                        t = t.permute(0, 3, 4, 1, 2)
                        t = t.reshape(chunk_len, actual_no * actual_ns, source_nx, source_ny)
                        t = F.interpolate(t, size=(nx, ny), mode='bilinear', align_corners=False, antialias=True)
                        t = t.reshape(chunk_len, actual_no, actual_ns, nx, ny)
                        return t.permute(0, 3, 4, 1, 2)

                    out_r = resize_tensor_gpu(w_r_gpu).cpu().numpy()
                    out_i = resize_tensor_gpu(w_i_gpu).cpu().numpy()
                    out_c = resize_tensor_gpu(w_c_gpu).cpu().numpy()

                    del w_r_clean, w_i_clean, w_r_gpu, w_i_gpu, w_c_gpu
                    torch.cuda.empty_cache() 
                    success = True
                    return chunk_index, start, end, out_r, out_i, out_c, f"GPU {device_id}", error_msg

            except RuntimeError as e:
                torch.cuda.empty_cache()
                success = False
                error_msg = str(e)

        # --- ATTEMPT 2: CPU FALLBACK ---
        if not success:
            wavelets_complex = np.square(w_r_slice) + np.square(w_i_slice)

            target_shape_chunk = (chunk_len,) + target_shape_full[1:]
            out_r = transform.resize(w_r_slice, target_shape_chunk, anti_aliasing=True)
            out_i = transform.resize(w_i_slice, target_shape_chunk, anti_aliasing=True)
            out_c = transform.resize(wavelets_complex, target_shape_chunk, anti_aliasing=True)
            
            del wavelets_complex
            return chunk_index, start, end, out_r, out_i, out_c, "CPU", error_msg
    
    # Execute workers and populate output arrays
    print(f"Dispatching to {max_workers} concurrent worker(s)...")
    progress_start = time.time()
    completed_chunks = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, i) for i in range(n_chunks)]
        
        for future in as_completed(futures):
            check_cancelled(cancel_event)
            chunk_idx, start, end, out_r, out_i, out_c, hw_used, err = future.result()
            
            if err:
                print(f"Chunk {chunk_idx + 1} GPU failure. Reason: {err}")
                
            w_r_downsampled[start:end] = out_r
            w_i_downsampled[start:end] = out_i
            w_c_downsampled[start:end] = out_c
            completed_chunks += 1
            print(
                f"{progress_message('Coarse wavelet cache', completed_chunks, n_chunks, progress_start, unit='chunks')} "
                f"[Hardware: {hw_used}]"
            )

    print("\nSaving cache to disk...")
    _validate_legacy_coarse_shape(w_r_downsampled, nx, ny, no, ns, "Real coarse wavelets")
    _validate_legacy_coarse_shape(w_i_downsampled, nx, ny, no, ns, "Imaginary coarse wavelets")
    _validate_legacy_coarse_shape(w_c_downsampled, nx, ny, no, ns, "Complex coarse wavelets")
    cache = np.lib.format.open_memmap(
        cache_path,
        mode="w+",
        dtype=np.float32,
        shape=(3,) + target_shape_full,
    )
    cache[0] = w_r_downsampled
    cache[1] = w_i_downsampled
    cache[2] = w_c_downsampled
    cache.flush()
    del cache
    return w_r_downsampled, w_i_downsampled, w_c_downsampled
