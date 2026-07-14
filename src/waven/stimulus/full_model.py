"""Full-resolution stimulus wavelet loading helpers.

These functions load real/imaginary wavelet arrays, optionally resize them to a
coarse analysis grid, and build feature matrices for model fitting. They do not
load neural recordings; see :mod:`waven.data.neural` for that.
"""
import gc
import math
import os
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
from ..runtime.performance import get_gpu_count
from ..storage.array_store import load_array

def _feature_shape(wavelets, no=None, ns=None, nf=None):
    """Infer orientation, sigma, and optional frequency dimensions."""
    if wavelets.ndim == 5:
        inferred_no, inferred_ns = wavelets.shape[3:5]
        inferred_nf = 1
    elif wavelets.ndim == 6:
        inferred_no, inferred_ns, inferred_nf = wavelets.shape[3:6]
    else:
        raise ValueError(
            f"Expected time-first wavelets with 5 or 6 dimensions, got {wavelets.shape}"
        )
    return (
        int(no if no is not None else inferred_no),
        int(ns if ns is not None else inferred_ns),
        int(nf if nf is not None else inferred_nf),
    )


def load_wavelets(
    pathdir,
    nx,
    ny,
    wavelets_r,
    wavelets_i,
    direction=False,
    chunk_size=1000,
    no=None,
    ns=None,
    nf=None,
):
    """Combine real and imaginary wavelet phases into a normalized magnitude map."""
    n_frames = wavelets_r.shape[0]
    no, ns, nf = _feature_shape(wavelets_r, no=no, ns=ns, nf=nf)
    if wavelets_i.shape != wavelets_r.shape:
        raise ValueError(
            f"Real/imag wavelets must have matching shapes, got {wavelets_r.shape} "
            f"and {wavelets_i.shape}"
        )
    source_has_frequency = wavelets_r.ndim == 6
    feature_shape = (no, ns) if not source_has_frequency else (no, ns, nf)
    
    w_r = wavelets_r.reshape((n_frames, nx, ny) + feature_shape)
    w_i = wavelets_i.reshape((n_frames, nx, ny) + feature_shape)
    
    # Pre-allocate output array in system RAM
    pn_wavelets = np.empty((n_frames, nx, ny) + feature_shape, dtype=np.float32)
    
    num_gpus = get_gpu_count()
    max_workers = max(1, num_gpus)
    n_chunks = math.ceil(n_frames / chunk_size)

    def process_chunk(chunk_index):
        """Function for process chunk.

        Args:
            chunk_index: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, n_frames)
        
        w_r_slice = w_r[start:end]
        w_i_slice = w_i[start:end]
        
        success = False
        
        if num_gpus > 0:
            device_id = chunk_index % num_gpus
            device = f"cuda:{device_id}"
            try:
                with torch.no_grad():
                    w_r_gpu = torch.from_numpy(w_r_slice).to(device)
                    w_i_gpu = torch.from_numpy(w_i_slice).to(device)
                    
                    pn_gpu = torch.abs(w_r_gpu) + torch.abs(w_i_gpu)
                    pn_gpu = pn_gpu.reshape((end - start, nx, ny, -1))
                    
                    sigma = 1.0
                    sum_wavelets = torch.sum(pn_gpu, dim=1, keepdim=True)
                    pn_gpu = pn_gpu / (sigma + sum_wavelets)
                    
                    out_chunk = pn_gpu.cpu().numpy().reshape(
                        (end - start, nx, ny) + feature_shape
                    )
                    
                    del w_r_gpu, w_i_gpu, pn_gpu, sum_wavelets
                    success = True
                    return chunk_index, start, end, out_chunk
            except RuntimeError:
                torch.cuda.empty_cache()
                success = False

        if not success:
            # Fallback to CPU
            pn_chunk = np.abs(w_r_slice) + np.abs(w_i_slice)
            pn_chunk = pn_chunk.reshape((end - start, nx, ny, -1))
            sigma = 1.0
            pn_chunk = pn_chunk / (sigma + np.sum(pn_chunk, axis=1, keepdims=True))
            out_chunk = np.reshape(pn_chunk, (end - start, nx, ny) + feature_shape)
            return chunk_index, start, end, out_chunk

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_chunk, i) for i in range(n_chunks)]
        for future in as_completed(futures):
            chunk_idx, start, end, out_chunk = future.result()
            pn_wavelets[start:end] = out_chunk

    print(pn_wavelets.shape)
    del w_r, w_i
    gc.collect()

    if direction:
        sh = np.array(wavelets_i.shape)
        sh[0] = sh[0] + 1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = wavelets_i / wavelets_r
            phase = np.insert(
                np.nan_to_num(np.arctan(ratio)),
                [0],
                np.zeros((nx, ny) + feature_shape),
                0,
            ).reshape(sh)
        phase_diff = np.diff(phase, axis=0)
        return pn_wavelets, phase_diff
        
    return pn_wavelets


def load_stimulus(
    pathdir,
    wavelets_r,
    wavelets_i,
    nx=138,
    ny=112,
    chunk_size=500,
    no=None,
    ns=None,
    nf=None,
    target_nx=None,
    target_ny=None,
    scale=None,
):
    # Dynamically determine frames to avoid the hardcoded 9000 mismatch
    """Function for load stimulus.

    Args:
        pathdir: Input value for this operation.
        wavelets_r: Input value for this operation.
        wavelets_i: Input value for this operation.
        nx: Input value for this operation.
        ny: Input value for this operation.
        chunk_size: Input value for this operation.
        no: Input value for this operation.
        ns: Input value for this operation.
        nf: Input value for this operation.
        target_nx: Input value for this operation.
        target_ny: Input value for this operation.
        scale: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    n_frames = wavelets_r.shape[0]
    no, ns, nf = _feature_shape(wavelets_r, no=no, ns=ns, nf=nf)
    target_nx, target_ny = (
        (target_nx, target_ny)
        if target_nx is not None and target_ny is not None
        else coarse_grid_dimensions(nx, ny)
    )
    scale = min(ns - 1, 2) if scale is None else int(scale)
    
    direct = True
    
    if not direct:
        wavelets = load_wavelets(
            pathdir,
            nx,
            ny,
            wavelets_r,
            wavelets_i,
            direction=direct,
            no=no,
            ns=ns,
            nf=nf,
        )
        if wavelets.ndim == 5:
            wavelets = wavelets[:, :, :, :, scale]
            resize_shape = (n_frames, target_nx, target_ny, no)
        else:
            wavelets = wavelets[:, :, :, :, scale, :]
            resize_shape = (n_frames, target_nx, target_ny, no, nf)
        wavelets = transform.resize(wavelets, resize_shape)
        wavelets = wavelets.reshape((n_frames, wavelets.shape[1] * wavelets.shape[2] * wavelets.shape[3]))
        return wavelets
        
    # DIRECT MODE: Multi-GPU Chunked Resize + Phase Calculation
    n_chunks = math.ceil(n_frames / chunk_size)
    num_gpus = get_gpu_count()
    max_workers = max(1, num_gpus)

    feature_shape = (no, ns) if wavelets_r.ndim == 5 else (no, ns, nf)
    target_shape = (n_frames, target_nx, target_ny) + feature_shape
    w_r_out = np.empty(target_shape, dtype=np.float32)
    w_i_out = np.empty(target_shape, dtype=np.float32)
    pn_wavelets = np.empty(target_shape, dtype=np.float32)

    def process_stimulus_chunk(chunk_index):
        """Function for process stimulus chunk.

        Args:
            chunk_index: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        start = chunk_index * chunk_size
        end = min((chunk_index + 1) * chunk_size, n_frames)
        chunk_len = end - start
        
        w_r_slice = wavelets_r[start:end].reshape((chunk_len, nx, ny) + feature_shape)
        w_i_slice = wavelets_i[start:end].reshape((chunk_len, nx, ny) + feature_shape)
        
        success = False
        if num_gpus > 0:
            device_id = chunk_index % num_gpus
            device = f"cuda:{device_id}"
            
            try:
                with torch.no_grad():
                    w_r_gpu = torch.from_numpy(np.ascontiguousarray(w_r_slice)).to(device).float()
                    w_i_gpu = torch.from_numpy(np.ascontiguousarray(w_i_slice)).to(device).float()
                    
                    def resize_gpu(t):
                        """Function for resize gpu.

                        Args:
                            t: Input value for this operation.

                        Returns:
                            Result produced by the operation.
                        """
                        if t.ndim == 5:
                            t = t.permute(0, 3, 4, 1, 2)
                            t = t.reshape(chunk_len, no * ns, nx, ny)
                            t = F.interpolate(
                                t,
                                size=(target_nx, target_ny),
                                mode='bilinear',
                                align_corners=False,
                                antialias=True,
                            )
                            t = t.reshape(chunk_len, no, ns, target_nx, target_ny)
                            return t.permute(0, 3, 4, 1, 2)
                        t = t.permute(0, 3, 4, 5, 1, 2)
                        t = t.reshape(chunk_len, no * ns * nf, nx, ny)
                        t = F.interpolate(
                            t,
                            size=(target_nx, target_ny),
                            mode='bilinear',
                            align_corners=False,
                            antialias=True,
                        )
                        t = t.reshape(chunk_len, no, ns, nf, target_nx, target_ny)
                        return t.permute(0, 4, 5, 1, 2, 3)

                    w_r_res = resize_gpu(w_r_gpu)
                    w_i_res = resize_gpu(w_i_gpu)
                    pn_res = torch.abs(w_r_res) + torch.abs(w_i_res)
                    
                    res_r = w_r_res.cpu().numpy()
                    res_i = w_i_res.cpu().numpy()
                    res_pn = pn_res.cpu().numpy()
                    
                    del w_r_gpu, w_i_gpu, w_r_res, w_i_res, pn_res
                    success = True
                    return start, end, res_r, res_i, res_pn
            except RuntimeError:
                torch.cuda.empty_cache()
                success = False
                
        if not success:
            res_r = transform.resize(w_r_slice, (chunk_len,) + target_shape[1:], anti_aliasing=True)
            res_i = transform.resize(w_i_slice, (chunk_len,) + target_shape[1:], anti_aliasing=True)
            res_pn = np.abs(res_r) + np.abs(res_i)
            return start, end, res_r, res_i, res_pn

    print(f"Dispatching load_stimulus to {max_workers} worker(s)...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_stimulus_chunk, i) for i in range(n_chunks)]
        for future in as_completed(futures):
            start, end, out_r, out_i, out_pn = future.result()
            w_r_out[start:end] = out_r
            w_i_out[start:end] = out_i
            pn_wavelets[start:end] = out_pn

    with np.errstate(divide='ignore', invalid='ignore'):
        phase = np.nan_to_num(np.arctan(w_i_out / w_r_out))
    phase_diff = np.diff(phase, axis=0)

    del w_r_out, w_i_out, wavelets_r, wavelets_i
    gc.collect()

    # Replaced 9000 with n_frames dynamically
    zero_phase = np.zeros(target_shape[1:], dtype=np.float32)
    pn_wavelets_fwd = pn_wavelets * np.insert(
        np.clip(phase_diff, 0, None),
        [0],
        zero_phase,
        0,
    ).reshape(target_shape)
    pn_wavelets_bkwd = pn_wavelets * np.insert(
        np.clip(-phase_diff, 0, None),
        [0],
        zero_phase,
        0,
    ).reshape(target_shape)
    
    wavelets = np.stack([pn_wavelets, pn_wavelets_fwd, pn_wavelets_bkwd], axis=-1)
    
    del pn_wavelets_bkwd, pn_wavelets_fwd, pn_wavelets
    gc.collect()

    if not source_has_frequency:
        wavelets = wavelets[:, :, :, :, scale, :]
    else:
        wavelets = wavelets[:, :, :, :, scale, :, :]
    
    wavelets = wavelets.reshape((wavelets.shape[0], -1))
    return wavelets
