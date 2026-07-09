"""Gabor filter and filter-bank construction.

This module contains only the spatial Gabor kernels and library builders.
Use :mod:`waven.wavelets.decomposition` for applying those libraries to
stimulus movies.
"""
import gc
import itertools
import math
import os
import shutil
import time

import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import gabor_kernel
from tqdm import tqdm

from ..runtime.performance import available_ram_bytes, has_enough_ram as _has_enough_ram
from ..runtime.task_control import check_cancelled, progress_message

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
        f"{available_gb:.2f} GB free -> "
        f"{'in-memory' if is_safe else 'disk streaming'}",
        end="\n\n",
    )
    return is_safe


def makeGaborFilter(i, j, angle, sigma, phase, f=0.4, lx=54, ly=135, plot=False, freq=True):
    """Function for makeGaborFilter.

    Args:
        i: Input value for this operation.
        j: Input value for this operation.
        angle: Input value for this operation.
        sigma: Input value for this operation.
        phase: Input value for this operation.
        f: Input value for this operation.
        lx: Input value for this operation.
        ly: Input value for this operation.
        plot: Input value for this operation.
        freq: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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

def makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq=True, cancel_event=None):
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

    progress_start = time.time()
    for i, x in enumerate(xs):
        check_cancelled(cancel_event)
        print(progress_message("Gabor library", i + 1, len(xs), progress_start, unit="x"))
        for j, y in enumerate(ys):
            for t_idx, t in enumerate(thetas):
                for s_idx, s in enumerate(sigmas):
                    for o_idx, o in enumerate(offsets):
                        filt = makeGaborFilter(x, y, t, s, o, f, lx=lx, ly=ly, freq=freq)
                        # Flatten and assign directly to pre-allocated memory
                        library[i, j, t_idx, s_idx, o_idx, :] = filt.flatten()

    return library


def makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies, cancel_event=None):
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
    
    progress_start = time.time()
    for i, x in enumerate(xs):
        check_cancelled(cancel_event)
        print(progress_message("Gabor library", i + 1, len(xs), progress_start, unit="x"))
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
