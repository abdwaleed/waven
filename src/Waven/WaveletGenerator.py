"""
Created on Wed Mar 25 19:31:32 2025

@author: Sophie Skriabine
@co-author: Abdelrahman Abdelrahman
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib
if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)
import itertools
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

def is_high_ram_system(threshold_gb=24):
    """
    Safely checks total system RAM. 
    Returns True if RAM >= threshold_gb, otherwise False.
    """
    try:
        import psutil
        total_ram = psutil.virtual_memory().total / (1024**3)
    except ImportError:
        # Fallback for Linux/Mac if psutil is not installed
        try:
            total_ram = (os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')) / (1024**3)
        except ValueError:
            # If all fails, assume low RAM for safety
            total_ram = 16.0 
            
    print(f"System Check: Detected {total_ram:.1f} GB Total RAM.")
    return total_ram >= threshold_gb


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


def downsample_video_binary(path, visual_coverage, analysis_coverage, shape=(54, 135), chunk_size=1000, ratios=(1, 1)):
    """
    Zero-RAM footprint downsampling using memory-mapped arrays and chunking.
    Fixed for scikit-image strict boolean anti-aliasing requirements.
    """
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
    
    xi = int((vis_cov - ana_cov)[2])
    yi = int((vis_cov - ana_cov)[0])
    
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
    
    print(f"Downsampling {total_frames} frames directly to disk...")
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
    high_ram = is_high_ram_system()
    chunk_size = 1000 if high_ram else 300
    max_workers = 4 if high_ram else 2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {path}")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
def getWTfromNPY(videodata, wavelet_slice, batch_size=100, filter_chunk_size=1000):
    """
    Optimized: Dual-chunking. Uses pinned memory and non-blocking CUDA transfers,
    and chunks BOTH the video frames and the wavelet library to prevent VRAM OOM.
    """
    
    WT = []
    
    # Pre-transpose and reshape the video data in one go on CPU
    video_flattened = videodata.reshape(videodata.shape[0], -1).T
    
    # Flatten wavelet spatial dimensions to 2D: (total_filters, num_pixels)
    # This makes it easy to chunk without breaking PyTorch's N-D broadcasting
    orig_filter_shape = wavelet_slice.shape[:-1]
    num_pixels = wavelet_slice.shape[-1]
    
    wavelet_flat = wavelet_slice.reshape(-1, num_pixels)
    total_filters = wavelet_flat.shape[0]
    
    for i in tqdm(range(0, video_flattened.shape[1], batch_size), desc="Processing Frame Batches"):
        # Make contiguous to prevent PyTorch pin_memory warnings on slices
        batch = np.ascontiguousarray(video_flattened[:, i : i + batch_size])
        
        # Pin memory for async transfer
        batch_cpu = torch.from_numpy(batch).pin_memory()
        batch_tensor = batch_cpu.to(device='cuda', dtype=torch.float16, non_blocking=True)
        
        batch_output = []
        
        # --- NEW: Chunk the filter library to cap VRAM usage ---
        for j in range(0, total_filters, filter_chunk_size):
            # Load only a safe chunk of the filters to the GPU
            l_chunk = torch.tensor(wavelet_flat[j : j + filter_chunk_size], dtype=torch.float16, device='cuda')
            
            # Matrix multiplication for this specific chunk
            out_chunk = l_chunk @ batch_tensor 
            
            # Pull back to CPU immediately
            batch_output.append(out_chunk.cpu().numpy())
            
            # Explicitly delete to free up the VRAM block for the next loop
            del l_chunk, out_chunk
        
        # Reconstruct the full filter output for this frame batch (total_filters, current_batch_size)
        full_batch_output = np.concatenate(batch_output, axis=0)
        
        # Reshape back to the original N-dimensional structure
        full_batch_output = full_batch_output.reshape(*orig_filter_shape, -1)
        
        # Apply your original axis swapping
        WT.append(full_batch_output.swapaxes(-1, -2))
        
        del batch_tensor, batch_cpu
        torch.cuda.empty_cache()
        
    return np.concatenate(WT, axis=-1)

@torch.no_grad()
def getWTfromNPY(videodata, waveletLibrary, phase, batch_size=256):
    """
    Batched GPU execution for maximum CUDA utilization. 
    Strictly backwards compatible.
    """
    import torch
    import gc
    import numpy as np

    # 1. Load only the specific phase slice to the GPU ONCE
    # Original shape expected by legacy transform: L[:, :, :, phase]
    L_gpu = torch.tensor(waveletLibrary[:, :, :, phase], dtype=torch.float32, device='cuda')
    
    num_frames = videodata.shape[0]
    WT_list = []
    
    # 2. Batch process the frames to saturate the GPU
    for i in range(0, num_frames, batch_size):
        # Extract batch and flatten each frame
        batch = videodata[i : min(i + batch_size, num_frames)]
        
        # Reshape to (pixels, batch_size) for matrix multiplication
        batch_flat = batch.reshape(batch.shape[0], -1).T 
        batch_gpu = torch.tensor(batch_flat, dtype=torch.float32, device='cuda')
        
        # 3. Massive parallel matrix multiplication: (L_dims, pixels) @ (pixels, batch_size)
        output_gpu = L_gpu @ batch_gpu
        
        # Move back to CPU immediately to free VRAM
        WT_list.append(output_gpu.cpu().numpy())
        
        del batch_gpu, output_gpu
    
    # Free the heavy library tensor
    del L_gpu
    torch.cuda.empty_cache()
    gc.collect()
    
    # 4. Concatenate and restore original axis order
    WT = np.concatenate(WT_list, axis=-1)
    
    # Original loop appended frames to axis 0. 
    # np.concatenate on axis=-1 put frames at the end, so we move them to the front.
    return np.moveaxis(WT, -1, 0)


def waveletTransform(frame,phase, L):
    output=L[:, :, :,phase]@torch.Tensor(frame.flatten()).cuda()
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletTransform3D(frame, L):
    output=L@torch.Tensor(frame.flatten()).cuda()
    # output=torch.sum(output, axis=(0, 1))
    return output.detach().cpu().numpy()


def waveletDecomposition(videodata, phase, sigmas, folder_path, library_path='/media/sophie/Expansion1/UCL/datatest/gabors_library.npy'):
    """
    Runs the wavelet decomposition.
    Optimized: Uses memory-mapping to load the library safely without OOM errors,
    and correctly passes the 3 required arguments to the batched GPU function.
    """
    import numpy as np
    import os
    import gc
    
    print(f"Loading Gabor library from {library_path} (mmap_mode='r')...")
    # mmap_mode='r' keeps the massive array on disk, only loading the chunks we need into RAM
    L = np.load(library_path, mmap_mode='r') 
    
    WT_list = []
    
    for s, ss in enumerate(sigmas):
        print(f"Processing sigma {s + 1}/{len(sigmas)}...")
        
        # Extract the 5D slice exactly as the original code expected
        l = L[:, :, :, s] 
        
        # Call the optimized getWTfromNPY with the correct 3 arguments
        wt = getWTfromNPY(videodata, l, phase)
        WT_list.append(wt)
        
        # Free up memory explicitly between sigma iterations
        gc.collect() 
        
    WT = np.array(WT_list)
    
    # Original logic: move the sigma axis from 0 to 4 (the end)
    WT = np.moveaxis(WT, 0, 4)
    
    # Save the output
    save_path = os.path.join(folder_path, f'dwt_videodata_{phase}.npy')
    np.save(save_path, WT)
    print(f"Success! Saved wavelet decomposition to {save_path}")


def getTrueRF(idx, rfs, L):
    rf=rfs[idx, :, :, :]#.swapaxes(0, 1)
    # rf = skimage.transform.resize(rf, (135, 54, 8),order=5, anti_aliasing=True)
    rfv=rf.reshape(1, -1)@L[:, :, :, 2, 0, :].reshape(-1,7290)

    plt.figure()
    plt.imshow(rfv.reshape(54, 135)[5:-5, 5:-5],  vmin=-np.max(rfv), vmax=np.max(rfv) ,cmap='coolwarm')#vmin=-0.0014, vmax=0.0014,

