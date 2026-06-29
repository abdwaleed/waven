"""Gabor filter construction and GPU-accelerated wavelet decomposition.

This module builds the spatial filter bank, downsamples stimulus movies to the
analysis grid, and projects video frames onto Gabor wavelets.  Large arrays are
streamed to disk when free RAM is insufficient; filter–video products are
computed in GPU batches sized from available VRAM.
"""
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from typing import Optional

import matplotlib
if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import scipy.io as sio
import skimage
from skimage import transform
from skimage.measure import block_reduce
import cv2
import gc
import torch
from tqdm import tqdm
import shutil
import math
from skimage.filters import gabor_kernel
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from .performance import (
    available_ram_bytes,
    has_enough_ram as _has_enough_ram,
    resolve_compute_device,
    video_downsample_chunk_size,
    wavelet_filter_chunk_size,
)


def has_enough_ram(required_bytes: int, safety_margin: float = 1.20) -> bool:
    """Return True when an array of ``required_bytes`` can live in free RAM.

    Logs the chosen route (in-memory vs disk streaming) for transparency.
    """
    available_bytes = available_ram_bytes()
    required_gb = required_bytes / (1024**3)
    available_gb = available_bytes / (1024**3)
    is_safe = _has_enough_ram(required_bytes, safety_margin)
    print(
        f"Memory check: needs {required_gb:.2f} GB, "
        f"{available_gb:.2f} GB free → "
        f"{'in-memory' if is_safe else 'disk streaming'}",
        end="\n\n",
    )
    return is_safe


def makeGaborFilter(i, j, angle, sigma, phase, f=0.4, lx=54, ly=135, plot=False, freq=True):
    backgrd=np.zeros((lx, ly))
    if freq:
        gk = gabor_kernel(frequency=f, theta=angle, sigma_x=sigma, sigma_y=sigma, offset=phase)
    else:
        gk = gabor_kernel(frequency=(-0.016*sigma)+0.148, theta=angle, sigma_x=sigma, sigma_y=sigma,offset=phase)
    # plt.figure()
    # plt.imshow(gk.real)
    #
    # plt.figure()
    # plt.imshow(canvas, vmin=0, vmax=0.006)

    canvas=np.ones((lx+(2*gk.shape[0]), ly+(2*gk.shape[1])))
    canvas[gk.shape[0]:gk.shape[0]+lx, gk.shape[1]:gk.shape[1]+ly]=backgrd

    dp=(gk.shape[0]-1)/2

    x=i+gk.shape[0]
    y=j+gk.shape[1]

    canvas[int(x-dp):int(x+dp+1), int(y-dp):int(y+dp+1)]=gk.real
    backgrd=canvas[gk.shape[0]:gk.shape[0]+lx, gk.shape[1]:gk.shape[1]+ly]
    if plot:
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.imshow(backgrd.T, cmap='Greys')
    return backgrd.T.astype('float16')


def makeGaborFilter3D(i, j, angle, sigma, tp_w, f=0.4, lx=54, ly=135, alpha1=0, alpha2=np.pi/4):
    """
    Backwards-compatible standalone 3D version. 
    Relies entirely on the optimized makeGaborFilter above.
    """
    phases = np.linspace(alpha1, alpha2, tp_w)
    f3d = np.array([makeGaborFilter(i, j, angle, sigma, phase, f=f, lx=lx, ly=ly) for phase in phases])
    return f3d.astype('float16')


def _universal_gabor_engine(save_path, xs, ys, base_shape, kernels):
    """
    The hidden core engine. 
    Dynamically handles both 6D and 7D arrays using itertools.
    """
    lx, ly = len(xs), len(ys)
    
    # Dynamically build the final dimensions
    final_shape = (lx, ly) + base_shape + (lx * ly,)
    chunk_shape = (ly,) + base_shape + (lx * ly,)
    
    required_bytes = math.prod(final_shape) * 2
    
    # Safegaurds
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    free_bytes = shutil.disk_usage(os.path.dirname(os.path.abspath(save_path))).free
    if free_bytes < required_bytes:
        raise OSError(f"Insufficient disk space! Needed {required_bytes/(1024**3):.2f} GB.")
    
    print(f"Allocating {len(final_shape)}D file ({required_bytes / (1024**3):.2f} GB) at {save_path}...")
    library_map = np.lib.format.open_memmap(save_path, mode='w+', dtype='float16', shape=final_shape)
    data_offset = library_map.offset
    del library_map 
    gc.collect()

    print("Streaming optimized chunks to physical disk...")
    x_chunk = np.zeros(chunk_shape, dtype='float16')
    canvas = np.zeros((lx, ly), dtype='float16')
    
    # Creates dynamic iterators for whatever dimensions the front-end passed
    inner_ranges = [range(dim) for dim in base_shape]
    
    with open(save_path, 'r+b') as file_stream:
        file_stream.seek(data_offset) 
        
        for x_idx, x in enumerate(tqdm(xs, desc="Processing X coords")):
            x_chunk.fill(0)
            
            for y_idx, y in enumerate(ys):
                
                # itertools.product unrolls 3 or 4 nested loops automatically!
                for inner_idx in itertools.product(*inner_ranges):
                    canvas.fill(0)
                    gk_real = kernels[inner_idx]
                    
                    dp_x, dp_y = gk_real.shape[0] // 2, gk_real.shape[1] // 2
                    x_min, x_max = max(0, x - dp_x), min(lx, x + dp_x + 1)
                    y_min, y_max = max(0, y - dp_y), min(ly, y + dp_y + 1)
                    k_x_min, k_x_max = dp_x - (x - x_min), dp_x + (x_max - x)
                    k_y_min, k_y_max = dp_y - (y - y_min), dp_y + (y_max - y)
                    
                    canvas[x_min:x_max, y_min:y_max] = gk_real[k_x_min:k_x_max, k_y_min:k_y_max]
                    
                    # Construct the slice index dynamically (e.g., [y_idx, t, s, o, :])
                    chunk_idx = (y_idx,) + inner_idx + (slice(None),)
                    x_chunk[chunk_idx] = canvas.T.flatten()

            x_chunk.tofile(file_stream)
            file_stream.flush()

    return np.lib.format.open_memmap(save_path, mode='r', dtype='float16', shape=final_shape)


# =====================================================================
# LEGACY ROUTERS
# =====================================================================

def makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq=True):
    """
    builds the Gabor library

    Parameters:
        thetas (int): number of orientatuion equally spaced between 0 and 180 degree.
        Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
        f (list): spatial frequencies expressed in pixels per cycles.
        offsets (list): 0 and pi/2.
        xs (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
        ys (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
        freq (boolean): if True the, takes into account the frequencies list to generate the gabors filters, if False, there is a linear relationship between the size and the spatial frequencies as found in ref paper

    Returns:
        npy file containing all the generated gabor filters of shape (nx, ny, n_orientation, n_sizes, n_freq (if defined independantly from sizes, n_phases, nx*ny))
    """
    import numpy as np

    lx, ly = len(xs), len(ys)
    num_t, num_s, num_o = len(thetas), len(sigmas), len(offsets)

    # Calculate the flat pixel size required by running a single dummy filter
    test_kernel = makeGaborFilter(xs[0], ys[0], thetas[0], sigmas[0], offsets[0], f, lx=lx, ly=ly, freq=freq)
    flat_size = test_kernel.size

    # Allocate exactly the RAM needed once
    library = np.zeros((lx, ly, num_t, num_s, num_o, flat_size), dtype=np.float16)

    for i, x in enumerate(xs):
        print(x)
        for j, y in enumerate(ys):
            for t_idx, t in enumerate(thetas):
                for s_idx, s in enumerate(sigmas):
                    for o_idx, o in enumerate(offsets):
                        filt = makeGaborFilter(x, y, t, s, o, f, lx=lx, ly=ly, freq=freq)
                        # Flatten and assign directly to pre-allocated memory
                        library[i, j, t_idx, s_idx, o_idx, :] = filt.flatten()

    return library


def makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies):
    """
    Pre-allocated array approach. Stops RAM fragmentation and speeds up CPU processing.
    """
    import numpy as np
    
    lx, ly = len(xs), len(ys)
    num_t, num_s, num_f, num_o = len(thetas), len(sigmas), len(frequencies), len(offsets)
    
    # Calculate the flat pixel size required by running a single dummy filter
    test_kernel = makeGaborFilter(xs[0], ys[0], thetas[0], sigmas[0], offsets[0], frequencies[0], lx=lx, ly=ly, freq=True)
    flat_size = test_kernel.size
    
    # Allocate exactly the RAM needed once
    library = np.zeros((lx, ly, num_t, num_s, num_f, num_o, flat_size), dtype=np.float16)
    
    for i, x in enumerate(xs):
        print(f"Processing X coordinate: {x}")
        for j, y in enumerate(ys):
            for t_idx, t in enumerate(thetas):
                for s_idx, s in enumerate(sigmas):
                    for f_idx, f in enumerate(frequencies):
                        for o_idx, o in enumerate(offsets):
                            filt = makeGaborFilter(x, y, t, s, o, f, lx=lx, ly=ly, freq=True)
                            library[i, j, t_idx, s_idx, f_idx, o_idx, :] = filt.flatten()
                            
    return library


def makeFilterLibrary3D(xs, ys, thetas, sigmas, offsets, f, tp_w, alpha1, alpha2, filename):
    """
    Optimized 3D Gabor Library generation.
    Pre-allocates the exact multidimensional array to prevent RAM spikes.
    Maintains the unused 'offsets' parameter for strict backwards compatibility.
    """
    import numpy as np

    lx, ly = len(xs), len(ys)
    num_t, num_s = len(thetas), len(sigmas)

    # Pre-allocate exactly as the original expected: (lx, ly, thetas, sigmas, tp_w, ly, lx)
    fp = np.zeros((lx, ly, num_t, num_s, tp_w, ly, lx), dtype=np.float16)
    print(fp.shape)

    for i_x, x in enumerate(xs):
        print(x)
        for j_y, y in enumerate(ys):
            for t_idx, t in enumerate(thetas):
                for s_idx, s in enumerate(sigmas):
                    # Generate the 3D block
                    l = makeGaborFilter3D(x, y, t, s, tp_w, f, lx=lx, ly=ly, alpha1=alpha1, alpha2=alpha2)
                    
                    # Assign using enumerator indices (i_x, j_y) to prevent IndexError 
                    # if x or y ever exceed the bounds of lx/ly
                    fp[i_x, j_y, t_idx, s_idx] = l

    print('saving...')
    np.save(filename, fp)
    return fp


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

    # Libraries may be 6-D (with frequency) or 5-D (legacy); default to f_idx=0.
    if waveletLibrary.ndim == 6:
        lib_phase = waveletLibrary[:, :, :, 0, phase, :]
    else:
        lib_phase = waveletLibrary[:, :, :, phase, :]

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
    print(f"Loading Gabor library from {library_path} (mmap_mode='r')...", end="\n\n")
    L = np.load(library_path, mmap_mode='r') 
    
    prefix_shape = L.shape[:3] # (lx, ly, thetas)
    num_filters = int(np.prod(prefix_shape))
    T = videodata.shape[0]
    
    final_shape = prefix_shape + (T, len(sigmas))
    
    # FIX 1: Use Python's math.prod to prevent 32-bit integer overflow on Windows!
    required_bytes = math.prod(final_shape) * 4 
    
    save_path = os.path.join(folder_path, f'dwt_videodata_{phase}.npy')
    
    # --- DYNAMIC HARDWARE ROUTING ---
    if has_enough_ram(required_bytes, safety_margin=1.15):
        WT_final = np.zeros(final_shape, dtype=np.float32)
        use_mmap = False
    else:
        WT_final = np.lib.format.open_memmap(save_path, mode='w+', dtype=np.float32, shape=final_shape)
        use_mmap = True

    # FIX 2: Create a continuous view of the entire array ONCE.
    # This prevents numpy from accidentally creating massive memory copies inside the loop.
    WT_flat = WT_final.reshape(num_filters, T, len(sigmas))

    for s, ss in enumerate(sigmas):
        print(f"Processing sigma {s + 1}/{len(sigmas)}...", end="\n\n")
        
        # Pass the flattened view and the specific sigma index
        getWTfromNPY(videodata, L[:, :, :, s], phase, WT_flat=WT_flat, s_idx=s)
        
        gc.collect() 
        torch.cuda.empty_cache() 
        
    if use_mmap:
        WT_final.flush()
        del WT_final, WT_flat
        print(f"Success! Saved streamed disk array to {save_path}")
    else:
        print("Saving array to disk...", end="\n\n")
        np.save(save_path, WT_final)
        print(f"Success! Saved RAM array to {save_path}", end="\n\n")


def getTrueRF(idx, rfs, L):
    rf=rfs[idx, :, :, :]#.swapaxes(0, 1)
    # rf = skimage.transform.resize(rf, (135, 54, 8),order=5, anti_aliasing=True)
    rfv=rf.reshape(1, -1)@L[:, :, :, 2, 0, :].reshape(-1,7290)

    plt.figure()
    plt.imshow(rfv.reshape(54, 135)[5:-5, 5:-5],  vmin=-np.max(rfv), vmax=np.max(rfv) ,cmap='coolwarm')#vmin=-0.0014, vmax=0.0014,

