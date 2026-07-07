"""Video downsampling and Gabor wavelet decomposition.

Functions here consume stimulus movies or downsampled movie arrays and write
wavelet coefficient arrays to disk. Filter-bank construction lives in
:mod:`waven.wavelets.filters`.
"""
import gc
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import cv2
import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
from tqdm import tqdm

from ..performance import resolve_compute_device, video_downsample_chunk_size, wavelet_filter_chunk_size
from .filters import has_enough_ram

def _process_binary_chunk(frames_buffer, xi, xe, yi, ye, shape, actual_len):
    """Background worker for binary downsampling."""
    chunk_arr = np.stack(frames_buffer, axis=0) > 100
    cropped = chunk_arr[:, xi:xe, yi:ye]
    resized = skimage.transform.resize(cropped, (actual_len, shape[0], shape[1]), anti_aliasing=True)
    return (resized >= 0.5).astype(bool)


def downsample_video_binary(
    path,
    visual_coverage,
    analysis_coverage,
    shape=(54, 135),
    chunk_size: Optional[int] = None,
    ratios=(1, 1),
    save_path=None,
):
    """Downsample a binary stimulus movie to the analysis grid via disk streaming.

    Frames are read in chunks, cropped to the analysis field of view, resized
    with anti-aliasing, and written to a memory-mapped ``_downsampled.npy`` file
    so peak RAM stays bounded regardless of movie length.
    """
    if chunk_size is None:
        chunk_size = video_downsample_chunk_size()
    import cv2
    import numpy as np
    import skimage.transform
    import gc

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error opening video: {path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ratio_x, ratio_y = ratios
    vis_cov = np.array(visual_coverage)
    ana_cov = np.array(analysis_coverage)
    
    xi = int(abs((vis_cov - ana_cov)[2]))
    yi = int(abs((vis_cov - ana_cov)[0]))
    
    # Grab one frame to dynamically determine cropping bounds
    ret, first_img = cap.read()
    if not ret: return
    img_gray = first_img[:, :, 0] > 100
    xe = int(ratio_y * img_gray.shape[0])
    ye = int(ratio_x * img_gray.shape[1])
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset video
    
    # Pre-allocate output directly on disk to save RAM
    if save_path is None:
        save_path = path[:-4] + '_downsampled.npy'
    output_shape = (total_frames, shape[0], shape[1])
    output_mmap = np.lib.format.open_memmap(save_path, mode='w+', dtype=bool, shape=output_shape)
    
    frames_buffer = []
    frame_idx = 0
    
    print(f"Downsampling {total_frames} frames directly to disk...", end="\n\n")
    while True:
        ret, img = cap.read()
        if not ret:
            break
            
        frames_buffer.append(img[:, :, 0])
        
        if len(frames_buffer) == chunk_size:
            # CAST TO FLOAT32 HERE to prevent scikit-image ValueError
            chunk_arr = (np.stack(frames_buffer, axis=0) > 100).astype(np.float32)
            chunk_cropped = chunk_arr[:, xi:xe, yi:ye]
            chunk_resized = skimage.transform.resize(chunk_cropped, (chunk_size, shape[0], shape[1]), anti_aliasing=True)
            
            # Write chunk directly to disk
            output_mmap[frame_idx : frame_idx + chunk_size] = chunk_resized >= 0.5
            frame_idx += chunk_size
            frames_buffer.clear()
            gc.collect()
            
    # Process remaining frames
    if frames_buffer:
        rem_size = len(frames_buffer)
        # CAST TO FLOAT32 HERE as well
        chunk_arr = (np.stack(frames_buffer, axis=0) > 100).astype(np.float32)
        chunk_cropped = chunk_arr[:, xi:xe, yi:ye]
        chunk_resized = skimage.transform.resize(chunk_cropped, (rem_size, shape[0], shape[1]), anti_aliasing=True)
        output_mmap[frame_idx : frame_idx + rem_size] = chunk_resized >= 0.5
        
    output_mmap.flush()
    del output_mmap
    gc.collect()
    print(f"Success! Saved optimized binary array to: {save_path}")


def _process_uint_chunk(frames_buffer, shape, actual_len):
    """Background worker for uint downsampling."""
    chunk_arr = np.stack(frames_buffer, axis=0)
    resized = skimage.transform.resize(chunk_arr, (actual_len, shape[0], shape[1]), anti_aliasing=True, preserve_range=True)
    return resized.astype(np.uint8)


def downsample_video_uint(path, shape=(54, 135)):
    """
    Auto-scaling UInt Downsampler.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Estimate bytes: total_frames * width * height * 1 byte (uint8) 
    # Multiply by 2 as a buffer for the background threads
    estimated_bytes = total_frames * shape[0] * shape[1] * 2 
    
    # Use our new smart checker!
    high_ram = has_enough_ram(estimated_bytes)
    
    chunk_size = 1000 if high_ram else 300
    max_workers = 4 if high_ram else 2

    futures = []
    frames_buffer = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=total_frames, desc="Downsampling UInt") as pbar:
        while True:
            ret, img = cap.read()
            if not ret:
                break
                
            frames_buffer.append(img[:, :, 0])
            pbar.update(1)
            
            if len(frames_buffer) == chunk_size:
                futures.append(executor.submit(_process_uint_chunk, list(frames_buffer), shape, chunk_size))
                frames_buffer.clear()

        if len(frames_buffer) > 0:
            futures.append(executor.submit(_process_uint_chunk, list(frames_buffer), shape, len(frames_buffer)))

    cap.release()
    processed_chunks = [future.result() for future in futures]
    
    if processed_chunks:
        final_video = np.concatenate(processed_chunks, axis=0)
        save_path = path[:-4] + '_downsampled.npy'
        np.save(save_path, final_video)
        print(f"Success! Saved optimized uint array to: {save_path}")


@torch.no_grad()
def getWTfromNPY(
    videodata,
    waveletLibrary,
    phase,
    WT_flat,
    s_idx,
    filter_chunk_size: Optional[int] = None,
    frequency_index: Optional[int] = None,
):
    """Project video frames onto one Gabor scale, writing into ``WT_flat``.

    The filter bank is applied in batches on the best available device (CUDA when
    present, otherwise CPU).  Results stream directly into ``WT_flat`` so the
    full wavelet tensor never has to exist on the GPU at once.
    """
    device = resolve_compute_device(prefer_gpu=True)
    if filter_chunk_size is None:
        filter_chunk_size = wavelet_filter_chunk_size()

    if not isinstance(videodata, torch.Tensor):
        video_tensor = torch.as_tensor(videodata, dtype=torch.float32, device=device)
    else:
        video_tensor = videodata.to(device=device, dtype=torch.float32)

    num_frames = video_tensor.shape[0]
    video_flat = video_tensor.reshape(num_frames, -1).t()
    spatial_pixels = video_flat.shape[0]

    if waveletLibrary.ndim == 6:
        if frequency_index is None:
            raise ValueError(
                "This Gabor slice still has an independent frequency axis. "
                "Pass frequency_index explicitly, or use the coupled coarse "
                "library for coarse RF decomposition. Refusing to silently "
                "use frequency index 0."
            )
        lib_phase = waveletLibrary[:, :, :, int(frequency_index), phase, :]
    else:
        lib_phase = waveletLibrary[:, :, :, phase, :]

    if lib_phase.shape[-1] != spatial_pixels:
        raise ValueError(
            "Gabor library pixel axis does not match video frames: "
            f"library has {lib_phase.shape[-1]} pixels, video has {spatial_pixels}. "
            "Check NX/NY and the downsampled movie shape in config.json."
        )

    lib_flat = lib_phase.reshape(-1, spatial_pixels)
    num_filters = lib_flat.shape[0]

    for start in range(0, num_filters, filter_chunk_size):
        end = min(start + filter_chunk_size, num_filters)
        lib_chunk = torch.as_tensor(
            lib_flat[start:end, :],
            dtype=torch.float32,
            device=device,
        )
        product = torch.matmul(lib_chunk, video_flat)
        WT_flat[start:end, :, s_idx] = product.cpu().numpy()
        del lib_chunk, product

    if device == "cuda":
        torch.cuda.empty_cache()


def waveletTransform(frame,phase, L):
    output=L[:, :, :,phase]@torch.Tensor(frame.flatten()).cuda()
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletTransform3D(frame, L):
    output=L@torch.Tensor(frame.flatten()).cuda()
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletDecomposition(videodata, phase, sigmas, folder_path, library_path):
    """Decompose a downsampled movie into Gabor wavelet coefficients.

    Chooses in-memory or memory-mapped output based on ``has_enough_ram``, then
    iterates over sigma scales and fills ``dwt_videodata_{phase}.npy``.
    """
    from ..zarr_compat import load_array

    print(f"Loading Gabor library from {library_path} (mmap_mode='r')...", end="\n\n")
    L = load_array(library_path, mmap_mode='r')
    if L.ndim >= 7:
        raise ValueError(
            "waveletDecomposition is the coarse RF path and expects a coupled "
            "coarse library without an independent frequency axis. Build/use "
            "the Coarse Library for coarse RF analysis, and reserve the Fine "
            "Library for waveletDecompositionFull."
        )
    
    prefix_shape = L.shape[:3]  # (lx, ly, thetas)
    num_filters = int(np.prod(prefix_shape))
    T = videodata.shape[0]
    if len(sigmas) > L.shape[3]:
        raise ValueError(
            f"Requested {len(sigmas)} sigma values, but the Gabor library "
            f"contains {L.shape[3]} sigma planes."
        )
    
    # Legacy analysis expects time-first wavelet arrays for coarse decomposition.
    final_shape = (T,) + prefix_shape + (len(sigmas),)
    
    required_bytes = math.prod(final_shape) * 4
    
    save_path = os.path.join(folder_path, f'dwt_videodata_{phase}.npy')
    
    # --- DYNAMIC HARDWARE ROUTING ---
    if has_enough_ram(required_bytes, safety_margin=1.15):
        WT_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
    else:
        WT_final = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=final_shape)
        use_mmap = True

    temp_flat = np.empty((num_filters, T, 1), dtype=np.float32)

    for s, ss in enumerate(sigmas):
        print(f"Processing sigma {s + 1}/{len(sigmas)}...", end="\n\n")
        temp_flat.fill(0)
        getWTfromNPY(videodata, L[:, :, :, s], phase, WT_flat=temp_flat, s_idx=0)
        spatial = temp_flat[:, :, 0].reshape(prefix_shape + (T,))
        WT_final[..., s] = np.transpose(spatial, (3, 0, 1, 2))
        
        gc.collect() 
        torch.cuda.empty_cache() 
        
    if use_mmap:
        WT_final.flush()
        del WT_final, temp_flat
        print(f"Success! Saved streamed disk array to {save_path}")
    else:
        print("Saving array to disk...", end="\n\n")
        np.save(save_path, WT_final)
        print(f"Success! Saved RAM array to {save_path}", end="\n\n")


def waveletDecompositionFull(
    videodata,
    phase,
    sigmas,
    frequencies,
    folder_path,
    library_path,
    library_sigmas=None,
    sigma_indices=None,
):
    """Decompose a movie into full-resolution wavelets for ``run_Full_Model``.

    Writes ``dwt_videodata2_r.npy`` (phase 0) or ``dwt_videodata2_i.npy`` (phase 1)
    with shape ``(T, nx, ny, n_orientations, n_sigmas, n_frequencies)``.
    """
    from ..config import resolve_sigma_indices

    print(
        f"Loading Gabor library from {library_path} (mmap_mode='r') "
        "for full-model decomposition...",
        end="\n\n",
    )
    from ..zarr_compat import load_array

    library = load_array(library_path, mmap_mode="r")
    lx, ly, num_t = int(library.shape[0]), int(library.shape[1]), int(library.shape[2])
    num_frames = videodata.shape[0]
    sigmas = np.asarray(sigmas, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    ns = len(sigmas)
    nf = len(frequencies)
    if library.ndim < 7 and nf > 1:
        raise ValueError(
            "Full-model wavelet generation was asked for multiple frequencies, "
            "but the Gabor library has no frequency axis. Regenerate the library "
            "from config.json with the configured Frequencies list."
        )

    if sigma_indices is None:
        if library_sigmas is None:
            if library.ndim >= 7:
                raise ValueError(
                    "library_sigmas is required when the Gabor library includes "
                    "a dedicated sigma axis"
                )
            library_sigmas = tuple(range(library.shape[3]))
        sigma_indices = resolve_sigma_indices(library_sigmas, sigmas)
    else:
        sigma_indices = tuple(int(i) for i in sigma_indices)

    phase_suffix = "_r" if phase == 0 else "_i"
    save_path = os.path.join(folder_path, f"dwt_videodata2{phase_suffix}.npy")
    final_shape = (num_frames, lx, ly, num_t, ns, nf)
    required_bytes = math.prod(final_shape) * 4

    if has_enough_ram(required_bytes, safety_margin=1.15):
        wt_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
    else:
        wt_final = np.lib.format.open_memmap(
            save_path,
            mode="w+",
            dtype=np.float32,
            shape=final_shape,
        )
        use_mmap = True

    num_filters = lx * ly * num_t
    temp_flat = np.zeros((num_filters, num_frames, 1), dtype=np.float32)

    for out_s, lib_s in enumerate(sigma_indices):
        for f_idx in range(nf):
            if library.ndim >= 7:
                lib_slice = library[:, :, :, lib_s, f_idx]
            elif library.ndim == 6:
                lib_slice = library[:, :, :, lib_s]
            else:
                lib_slice = library[:, :, :, lib_s]

            getWTfromNPY(
                videodata,
                lib_slice,
                phase,
                WT_flat=temp_flat,
                s_idx=0,
            )
            spatial = temp_flat[:, :, 0].reshape(lx, ly, num_t, num_frames)
            wt_final[:, :, :, :, out_s, f_idx] = np.transpose(
                spatial,
                (3, 0, 1, 2),
            )
            gc.collect()
            if resolve_compute_device(prefer_gpu=True) == "cuda":
                torch.cuda.empty_cache()

    if use_mmap:
        wt_final.flush()
        del wt_final
        print(f"Success! Saved streamed full-model array to {save_path}")
    else:
        print("Saving full-model array to disk...", end="\n\n")
        np.save(save_path, wt_final)
        print(f"Success! Saved full-model array to {save_path}", end="\n\n")


def getTrueRF(idx, rfs, L):
    rf=rfs[idx, :, :, :]#.swapaxes(0, 1)
    # rf = skimage.transform.resize(rf, (135, 54, 8),order=5, anti_aliasing=True)
    rfv=rf.reshape(1, -1)@L[:, :, :, 2, 0, :].reshape(-1,7290)

    plt.figure()
    plt.imshow(rfv.reshape(54, 135)[5:-5, 5:-5],  vmin=-np.max(rfv), vmax=np.max(rfv) ,cmap='coolwarm')#vmin=-0.0014, vmax=0.0014,

