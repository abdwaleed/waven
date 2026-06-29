"""Receptive-field estimation, tuning analysis, and nonlinear Gabor models.

Core numerics for correlating neural responses with wavelet stimulus features,
fitting per-neuron nonlinear models, and running the high-resolution full model.
GPU-accelerated paths fall back to CPU automatically when VRAM is exhausted.
"""

import os
import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
import pickle
from scipy.stats import pearsonr
import torch
from scipy import signal
import seaborn as sns
import pandas as pd
from scipy.stats import skew
from scipy import interpolate
from scipy.sparse.linalg import svds
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.fftpack import fft, ifft
import tifffile
from numpy import exp
from scipy.spatial import cKDTree
from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
from sklearn.cluster import KMeans
import cv2 as cv
import gc
import matplotlib.gridspec as gridspec
from sklearn.decomposition import NMF
import math
import cv2
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker
import zarr
from joblib import Parallel, delayed
from scipy import signal
from sklearn.metrics import r2_score, explained_variance_score
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
from numba import njit, prange
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from scipy import ndimage
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
import os
from joblib import Parallel, delayed

from .performance import gpu_neuron_chunk_size, model_parallel_jobs

DEFAULT_MOVIE_FRAME_RATE_HZ = 30
SECONDS_PER_MINUTE = 60
DEFAULT_FRAMES_PER_MINUTE = (
    DEFAULT_MOVIE_FRAME_RATE_HZ * SECONDS_PER_MINUTE
)


def pi_formatter(x, pos):
    n = int(np.round(x / np.pi, 2))
    if n == 0:
        return "0"
    elif n == 1:
        return r"$\pi$"
    elif n == -1:
        return r"$-\pi$"
    else:
        return r"${}\pi$".format(n)


def Decay(t, tau, t0):
    ''' Decay exponential and step function '''
    return 1. / tau * np.exp(-t / tau) * 0.5 * (np.sign(t - t0) + 1.0)


def convolve_Stim(stim, time_trial1, tau=1.8):
    lambda_1 = 1 / tau
    t_response = time_trial1
    t0 = 0
    r = Decay(t_response, tau, t0)
    st = stim.reshape(stim.shape[0], -1)
    
    # Accelerated 2D convolution via FFT mode for significant speedups on large shapes
    convolved_stim = signal.convolve(st, r[:, None], mode='full', method='fft').astype('float16')
    convolved_stim = convolved_stim[:stim.shape[0], :].T
    
    return convolved_stim


def max_by_index(idx, arr):
    sub_arr = arr[idx]
    flat_idx = np.argmax(sub_arr)
    unrav = np.unravel_index(flat_idx, sub_arr.shape)
    
    # Maintained for full backward-compatibility with external callers
    where_format = tuple(np.array([dim]) for dim in unrav)
    return where_format, sub_arr.flat[flat_idx]


def _safe_chunked_cross_corr(stim_flat, resp, chunk_size=None):
    """Pearson cross-correlation between stimulus features and neural responses.

    Standardizes columns once, keeps the stimulus on the compute device for all
    neuron chunks, and falls back from CUDA to CPU when VRAM is insufficient.
    """
    n_time = stim_flat.shape[0]
    n_features = stim_flat.shape[1]
    n_neurons = resp.shape[1]

    if chunk_size is None:
        chunk_size = gpu_neuron_chunk_size(n_time, n_features)

    stim_mean = np.mean(stim_flat, axis=0, keepdims=True)
    stim_std = np.std(stim_flat, axis=0, keepdims=True, ddof=1)
    stim_std[stim_std == 0] = 1.0
    stim_norm = (stim_flat - stim_mean) / stim_std

    rfs = np.empty((n_neurons, n_features), dtype=np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cuda":
        try:
            stim_tensor = torch.from_numpy(stim_norm).to(device)
            for start in range(0, n_neurons, chunk_size):
                end = min(start + chunk_size, n_neurons)
                resp_chunk = resp[:, start:end]
                resp_mean = np.mean(resp_chunk, axis=0, keepdims=True)
                resp_std = np.std(resp_chunk, axis=0, keepdims=True, ddof=1)
                resp_std[resp_std == 0] = 1.0
                resp_norm = (resp_chunk - resp_mean) / resp_std

                resp_tensor = torch.from_numpy(resp_norm).to(device)
                chunk_corr = (resp_tensor.T @ stim_tensor) / (n_time - 1)
                rfs[start:end] = chunk_corr.cpu().numpy()
                del resp_tensor, chunk_corr

            del stim_tensor
        except RuntimeError:
            torch.cuda.empty_cache()
            device = "cpu"

    if device == "cpu":
        for start in range(0, n_neurons, chunk_size):
            end = min(start + chunk_size, n_neurons)
            resp_chunk = resp[:, start:end]
            resp_mean = np.mean(resp_chunk, axis=0, keepdims=True)
            resp_std = np.std(resp_chunk, axis=0, keepdims=True, ddof=1)
            resp_std[resp_std == 0] = 1.0
            resp_norm = (resp_chunk - resp_mean) / resp_std
            rfs[start:end] = (resp_norm.T @ stim_norm) / (n_time - 1)

    return rfs

def PearsonCorrelation(stim, resp, neuron_pos, nx, ny, plotting=True):
    """Correlate wavelet stimulus with responses and locate RF peaks per neuron."""
    stim_flat = np.abs(stim.reshape(stim.shape[0], -1))

    rfs = _safe_chunked_cross_corr(stim_flat, resp)
    print((resp.shape[1] + stim_flat.shape[1], resp.shape[1] + stim_flat.shape[1]))

    rfs[rfs >= 0.99] -= 1.0
    np.nan_to_num(rfs, copy=False)
    rfs = rfs.reshape(rfs.shape[0], ny, nx)

    # Completely vectorized multi-dimensional argmax replacing the row loop
    flat_rfs = rfs.reshape(rfs.shape[0], -1)
    flat_max_idx = np.argmax(flat_rfs, axis=1)
    ymax, xmax = np.unravel_index(flat_max_idx, (ny, nx))

    if plotting:
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=xmax, cmap='jet')
        plt.colorbar()

        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=ymax, cmap='jet')
        plt.colorbar()

    return rfs, [xmax, ymax]

def orientation_correction_for_stretches(visual_coverage, nx, ny, omax):
    xM, xm, yM, ym = visual_coverage
    x_ratio = abs(xM - xm) / nx
    y_ratio = abs(yM - ym) / ny
    delta_y = y_ratio / x_ratio
    corrected_ori = np.arctan(delta_y * np.tan(omax * np.pi / 180)) * 180 / np.pi
    
    # Vectorized boolean masking directly modifies the array in-place
    corrected_ori[corrected_ori < 0] += 180
    return corrected_ori


def PearsonCorrelationPinkNoise(stim, resp, neuron_pos, nx, ny, ns, nf, visual_coverage, screen_ratio, sigmas, frequencies, n_orientations=8, fil=[0], absolute=False, plotting=False):
    """RF correlation for pink-noise stimuli with retinotopy and tuning extraction (6D)."""
    stim_flat = stim.reshape(stim.shape[0], -1)

    rfs = _safe_chunked_cross_corr(stim_flat, resp)
    print((resp.shape[1] + stim_flat.shape[1], resp.shape[1] + stim_flat.shape[1]))

    if absolute:
        rfs = np.abs(rfs)

    rfs[rfs >= 0.99] -= 1.0
    np.nan_to_num(rfs, copy=False)
    print(rfs.shape)

    # 1. Update reshape for the 6th dimension (nf)
    rfs = rfs.reshape(rfs.shape[0], nx, ny, n_orientations, ns, nf)

    abs_rfs = np.abs(rfs)
    flat_abs_rfs = abs_rfs.reshape(abs_rfs.shape[0], -1)
    flat_max_idx = np.argmax(flat_abs_rfs, axis=1)

    maxes = flat_abs_rfs[np.arange(abs_rfs.shape[0]), flat_max_idx]
    
    # 2. Unpack 5 dimensions
    xmax, ymax, omax, smax, fmax = np.unravel_index(flat_max_idx, (nx, ny, n_orientations, ns, nf))

    # 3. Add frequency to the max outputs
    maxe = [xmax, ymax, omax, smax, fmax]
    
    xM, xm, yM, ym = visual_coverage
    degrees_per_orientation = 180.0 / n_orientations
    omax_corr = orientation_correction_for_stretches(
        visual_coverage, nx, ny, omax * degrees_per_orientation
    )
    xmax_corr = (abs(xmax) * (abs(xm - xM) / nx)) + xM
    ymax_corr = (abs(ymax - ny) * (abs(yM - ym) / ny)) + ym
    
    smax_corr = sigmas[smax.astype(int)]
    fmax_corr = frequencies[fmax.astype(int)]
    
    maxe_corr = [xmax_corr, ymax_corr, omax_corr, smax_corr, fmax_corr]
    
    if plotting:
        if np.sum(fil) == 0:
            # Added frequency to the plotting loops
            for c_arr, title, cmap, vmax in zip([xmax_corr, ymax_corr, omax_corr, smax_corr, fmax_corr], 
                                                ['azimuth (visual degree)', 'elevation (visual degree)', 'orientation (degree)', 'size (visual degree)', 'spatial frequency (cyc/deg)'],
                                                ['jet', 'jet', 'hsv', 'coolwarm', 'viridis'], [None, None, 180, None, None]):
                plt.figure()
                plt.rcParams['axes.facecolor'] = 'none'
                plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=c_arr, cmap=cmap, vmax=vmax)
                plt.colorbar()
                plt.title(title)
                plt.xlabel('position x (um)')
                plt.ylabel('position y (um)')

            # ... [Keep your screen positions plot here] ...
        else:
            fil = maxes > 0.2
            print('filtering')
            for c_arr, title in zip([xmax, ymax, omax], ['xmax', 'ymax', 'omax']):
                plt.figure()
                plt.scatter(neuron_pos[fil, 1], neuron_pos[fil, 0], s=5, c=c_arr[fil], cmap='jet_r')
                plt.colorbar()
                plt.title(title)
                
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rfs, maxe, maxe_corr, list(maxes)


def realign_Stim_mc(stim, syncEcho_flip_times, stim_times):
    # Completely vectorized searchsorted execution replacing structural loop appending
    idx = np.searchsorted(stim_times, syncEcho_flip_times, side='right') - 1
    return stim[:, :, idx]


def _vectorized_pearson_r(x, y):
    """Computes column-wise Pearson correlation between matrices x and y of shape (T, N)"""
    x_norm = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-9)
    y_norm = (y - np.mean(y, axis=0)) / (np.std(y, axis=0) + 1e-9)
    return np.mean(x_norm * y_norm, axis=0)


def repetability_trial(resps_all, neuron_pos):
    # Vectorized trial cross-comparisons bypassing cell loops
    c01 = _vectorized_pearson_r(resps_all[0], resps_all[1])
    c02 = _vectorized_pearson_r(resps_all[0], resps_all[2])
    c12 = _vectorized_pearson_r(resps_all[1], resps_all[2])
    respcorrs3 = (c01 + c02 + c12) / 3.0

    plt.figure()
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, vmax=1, c=respcorrs3, cmap='Greys')
    plt.colorbar()

    return respcorrs3


def repetability_trial2(resps_all, neuron_pos):
    S, T, N = resps_all.shape
    sum_resps = np.sum(resps_all, axis=0, keepdims=True)
    respcorrs = np.zeros(N)
    
    # Vectorized execution across all neurons concurrently per trial step
    for t in range(S):
        mean_resp = (sum_resps - resps_all[t:t+1]) / (S - 1)
        respcorrs += _vectorized_pearson_r(resps_all[t], mean_resp[0])
        
    respcorrs3 = respcorrs / S

    plt.figure()
    plt.rcParams['axes.facecolor'] = 'none'
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, vmin=0, vmax=1, c=respcorrs3, cmap='Greys')
    plt.colorbar()

    return respcorrs3


def repetability_trial3(resps_all, neuron_pos, plotting=True):
    S, T, N = resps_all.shape
    
    if S == 4:
        m1 = np.mean(resps_all[[0, 2], :, :], axis=0)
        m2 = np.mean(resps_all[[1, 3], :, :], axis=0)
    elif S == 5:
        m1 = np.mean(resps_all[[0, 2, 4], :, :], axis=0)
        m2 = np.mean(resps_all[[1, 3], :, :], axis=0)
    elif S == 2:
        m1 = resps_all[0]
        m2 = resps_all[1]
    elif S == 3:
        m1 = np.mean(resps_all[[0, 2], :, :], axis=0)
        m2 = np.mean(resps_all[[1], :, :], axis=0)
    else:
        m1 = np.zeros((T, N))
        m2 = np.zeros((T, N))

    # OPTIMIZATION: Removed the dead 'pass' loop here entirely.
    
    respcorrs3 = _vectorized_pearson_r(m1, m2)

    if plotting:
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.box(True)
        plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, vmin=0, vmax=1, c=respcorrs3, cmap='Greys')
        plt.colorbar()

    return respcorrs3

def match_cumulative_cdf(source, template):
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def plotcolorbar(vmin, vmax, cmap='coolwarm'):
    if vmax < 1:
        l = np.random.randint(vmin * 1000, vmax * 1000, 10000).reshape(100, 100) / 1000
    else:
        l = np.random.randint(vmin, vmax, 10000).reshape(100, 100)
    plt.figure()
    plt.imshow(l, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()


def plotVariability(resp):
    resps_med = np.mean(resp, axis=0)
    mins = np.min(resp, axis=0)
    maxes = np.max(resp, axis=0)
    plt.plot(resps_med, c='k')

    plt.figure()
    plt.rcParams['axes.facecolor'] = 'none'
    plt.plot(resps_med, c='k')
    x = np.arange(resps_med.shape[0])
    plt.fill_between(x, mins, maxes, color='lightgrey')
    plt.show()


def cart2pol4d(x, y, dp, dn):
    # Math runs sparse to save massive peak RAM
    xx, yy, dd, nn = np.meshgrid(x, y, dp, dn, sparse=True)
    rho = np.hypot(xx, yy) + nn
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    
    # Broadcast back to dense right before returning to guarantee 100% backward compatibility
    dd_dense = np.broadcast_to(dd, rho.shape).copy()
    nn_dense = np.broadcast_to(nn, rho.shape).copy()
    return rho, phi, dd_dense, nn_dense


def cart2pol3d(x, y, dp):
    # Math runs sparse to save peak RAM
    xx, yy, dd = np.meshgrid(x, y, dp, sparse=True)
    rho = np.hypot(xx, yy)
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    
    # Broadcast back to dense right before returning to guarantee 100% backward compatibility
    dd_dense = np.broadcast_to(dd, rho.shape).copy()
    return rho, phi, dd_dense


def cart2pol(x, y):
    # Completely safe as-is; rho and phi naturally evaluate to dense arrays
    xx, yy = np.meshgrid(x, y, sparse=True)
    rho = np.hypot(xx, yy)
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    return rho, phi


def cart2pol_noise(x, y, n):
    # Completely safe as-is; rho and phi naturally evaluate to dense arrays
    xx, yy, nn = np.meshgrid(x, y, n, sparse=True)
    rho = np.hypot(xx, yy) + nn
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    return rho, phi


def preferedDirection(videodata, spks, x, y, o, w_i, w_r, window_size=5):
    rho, phi = cart2pol(w_r, w_i)
    plt.figure()
    plt.plot(phi * 180 / np.pi)
    angle = phi * 180 / np.pi
    pref_dir = np.mean((np.mean(spks[:, :, 9275], axis=0) > 0) * angle * (abs(angle) >= o - 10))
    return angle, pref_dir


def PCcorrelation(spks, neuron_pos):
    n_cell = spks.shape[2]
    U, S, Vh = np.linalg.svd(spks.reshape(-1, n_cell), full_matrices=False)
    pc1 = U[:, 0]
    pc2 = U[:, 1]
    plt.figure()
    plt.rcParams['axes.facecolor'] = 'none'
    plt.plot(pc1)
    return pc1, pc2


def NeuronCorrelation(idx, spks, neuron_pos):
    # OPTIMIZATION: as_tensor avoids unnecessary CPU copies. Move to GPU before reshaping/transposing.
    try:
        t_spks = torch.as_tensor(spks, device='cuda')
        cc_ = torch.corrcoef(t_spks.reshape(-1, t_spks.shape[2]).T)
    except:
        print('sparsenoise sp045 ?')
        t_spks = torch.as_tensor(spks, device='cuda')
        cc_ = torch.corrcoef(t_spks)
        
    cc_f_1 = cc_[idx].detach().cpu().numpy()
    neurocorr = np.asarray(cc_f_1 >= 0.08).nonzero()[0]
    
    # OPTIMIZATION: Explicitly clean up heavy local VRAM references
    res = cc_.detach().cpu().numpy()
    del t_spks, cc_
    
    return res, neurocorr


def predictSparseNoise(resp, stim, rfs, tp, save=False):
    img = resp[tp].reshape(1, -1) @ rfs.reshape(rfs.shape[0], -1)
    img = img.reshape(8, 20)

    fig, ax = plt.subplots(8, 1)
    ax[0].imshow(img)
    for i in range(1, 8):
        ax[i].imshow(stim[tp - i + 1])
    if save:
        tifffile.imwrite('/media/sophie/Expansion1/UCL/datatest/SP045/2023-10-04/3/tp3000reconstructed.tif', img.reshape(54, 135))


def predictPinkNoise(maxes, vis_n, spks, rfs, tp, L, videodata, dt=50, save=False):
    rfs_model = np.zeros((spks.shape[-1], 27, 11, 8, 4))

    # Replaced performance-heavy loop with native advanced array indexing
    rfs_model[np.arange(spks.shape[-1]), maxes[0].astype(int), maxes[1].astype(int), maxes[2].astype(int), maxes[3].astype(int)] = 1

    try:
        rfs_model = rfs[0] * rfs_model.reshape(spks.shape[-1], 27, 11, 8, 4)
    except:
        rfs_model = np.swapaxes(rfs[0], 1, 2) * rfs_model.reshape(spks.shape[-1], 27, 11, 8, 4)

    r = np.sum(np.mean(spks[:, :, :], axis=0)[tp - dt:tp], axis=0).reshape(1, -1)
    vis = r @ rfs_model.reshape(rfs_model.shape[0], -1)
    vis = vis.reshape(27, 11, 8, 4)

    vis_t = skimage.transform.resize(vis, (135, 54, 8, 4), order=5, anti_aliasing=True)

    fig, ax = plt.subplots(8, 4)
    for i in range(8):
        ax[i, 0].imshow(vis_t[:, :, i, 0].T, cmap='coolwarm')
        ax[i, 1].imshow(vis_t[:, :, i, 1].T, cmap='coolwarm')
        ax[i, 2].imshow(vis_t[:, :, i, 2].T, cmap='coolwarm')
        ax[i, 3].imshow(vis_t[:, :, i, 3].T, cmap='coolwarm')

    vs = vis_t[:, :, :, 2].reshape(1, -1) @ L[:, :, :, 2, 0].reshape(-1, 135 * 54)
    print(np.corrcoef(vs, np.mean(videodata[tp - 2 * dt:tp - dt], axis=0).flatten()))

    if save:
        tifffile.imwrite('/media/sophie/Expansion1/UCL/datatest/SP045/2023-10-04/3/tp3520videodata.tif',
                         np.mean(videodata[tp - dt:tp], axis=0))

    vc = vis_t.reshape(1, -1) @ L[:, :, :, :, 0].reshape(-1, 135 * 54)
    if save:
        tifffile.imwrite('/media/sophie/Expansion1/UCL/datatest/SP045/2023-10-04/3/tp3520reconstructed_1wavelet.tif',
                         (vs + vc).reshape(54, 135))

    return (vs + vc).reshape(54, 135)


def compute_skewness_neurons(spks, plotting=False):
    s = np.mean(spks, axis=0)
    # Skewness applied directly to the 1D array
    skewness = skew(s)

    if plotting:
        plt.figure()
        plt.hist(skewness, bins=100)
    return list(skewness)


def lowpassfilter(sig, N=1, Wn=1.5, fs=100):
    sos = signal.butter(N, Wn, 'lowpass', fs=fs, output='sos')
    return signal.sosfilt(sos, sig)


def DirectionSelectivity(x, y, o, s, w_i_downsampled, w_r_downsampled):
    sos = signal.butter(1, 2, 'lowpass', fs=30, output='sos')

    wi_filt = signal.sosfilt(sos, w_i_downsampled[:, x, y, o, s])
    wr_filt = signal.sosfilt(sos, w_r_downsampled[:, x, y, o, s])
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = wi_filt / wr_filt
        ratio[~np.isfinite(ratio)] = 0

    tan = np.diff(ratio, append=0)

    tan_pos = tan > 0
    tan_neg = tan < 0
    return tan_pos.astype(int), tan_neg.astype(int), tan


def DirectionSelectivityPlot(idx, spks, x, y, o, s, w_i_downsampled, w_r_downsampled, plotting=False):
    sos = signal.butter(1, 2, 'lowpass', fs=30, output='sos')

    wi_filt = signal.sosfilt(sos, w_i_downsampled[:, x, y, o, s])
    wr_filt = signal.sosfilt(sos, w_r_downsampled[:, x, y, o, s])

    sine = np.diff(wi_filt, append=0)
    cosine = np.diff(wr_filt, append=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = wi_filt / wr_filt
        ratio[~np.isfinite(ratio)] = 0
        
    tan = np.diff(ratio, append=0)

    pp = np.logical_and(sine > 0, cosine > 0)
    ngp = np.logical_and(sine < 0, cosine > 0)
    ngng = np.logical_and(sine < 0, cosine < 0)
    png = np.logical_and(sine > 0, cosine < 0)
    tan_pos = tan > 0
    tan_neg = tan < 0

    if plotting:
        Y = np.mean(spks[:4, :], axis=0)[:, idx].reshape(-1, 1)
        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.plot(Y, c='k')
        plt.twinx()
        plt.plot(w_i_downsampled[:, x, y, o, s], c='b')
        plt.twinx()
        plt.plot(w_r_downsampled[:, x, y, o, s], c='r')

        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.plot(Y, c='k')
        plt.twinx()
        plt.plot(tan_pos, c='b')
        plt.fill_between(np.arange(9000), tan_pos, color='b', alpha=0.2)
        plt.plot(tan_neg, c='r')
        plt.fill_between(np.arange(9000), tan_neg, color='r', alpha=0.2)
        plt.twinx()
        plt.plot(abs(tan))
        plt.ylim(0, 10)

        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.plot(Y, c='k')
        plt.twinx()
        plt.plot(ngng, c='y')
        plt.fill_between(np.arange(9000), ngng, color='y', alpha=0.2)
        plt.plot(pp, c='r')
        plt.fill_between(np.arange(9000), pp, color='r', alpha=0.2)
        plt.plot(ngp, c='g')
        plt.fill_between(np.arange(9000), ngp, color='g', alpha=0.2)
        plt.plot(png, c='b')
        plt.fill_between(np.arange(9000), png, color='b', alpha=0.2)
        plt.twinx()

        plt.figure()
        plt.rcParams['axes.facecolor'] = 'none'
        plt.plot(Y, c='k')
        plt.twinx()
        plt.plot(pp, c='r')
        plt.plot(ngp, c='g')
        plt.plot(ngng, c='y')
        plt.plot(png, c='b')

    return tan_pos.astype(int), tan_neg.astype(int), abs(tan)


def SinCosPlot(idx, spks, x, y, o, s, w_i_downsampled, w_r_downsampled, ncut):
    # Flatten arrays directly instead of reshaping to (-1, 1) to avoid Pandas DataFrame conversion
    sin = w_i_downsampled[:, x, y, o, s].ravel()
    cos = w_r_downsampled[:, x, y, o, s].ravel()
    Y = np.mean(spks[:4, :], axis=0)[:, idx].ravel()
    Y_past = np.zeros_like(Y)
    Y_past[:-5] = Y[5:]

    # OPTIMIZATION: Replaced Memory-heavy Pandas grouping with C-optimized binned_statistic_2d
    # Axis 0 (rows) matches 'sin', Axis 1 (cols) matches 'cos'
    res_z = binned_statistic_2d(sin, cos, Y, statistic='mean', bins=ncut)
    res_z_past = binned_statistic_2d(sin, cos, Y_past, statistic='mean', bins=ncut)

    # Reversing axis 0 exactly replicates the behavior of `means = means.iloc[::-1]`
    z = res_z.statistic[::-1, :]
    z2 = res_z_past.statistic[::-1, :]

    # Calculate midpoints mathematically directly from bin edges (no interval objects needed)
    # dx (columns) were not reversed
    dx = 0.5 * (res_z.y_edge[:-1] + res_z.y_edge[1:])
    # dy (rows) were reversed
    dy = 0.5 * (res_z.x_edge[:-1] + res_z.x_edge[1:])[::-1]

    theta1 = np.arctan(dy[np.where(z == np.max(np.nan_to_num(z)))[0][0]] / dx[np.where(z == np.max(np.nan_to_num(z)))[1][0]]) * 180 / np.pi

    rho, phi = cart2pol(dx, dy)
    print(dx.shape, rho.shape)
    plt.figure()
    plt.rcParams['axes.facecolor'] = 'none'
    plt.subplot(projection="polar")
    plt.pcolormesh(phi, rho, z[:-1, :-1], cmap='coolwarm', shading='flat')
    plt.plot(phi, rho, color='k', ls='none')
    plt.grid()
    plt.title('Means of spks polar coordinate')

    theta2 = np.arctan(dy[np.where(z2 == np.max(np.nan_to_num(z2)))[0][0]] / dx[np.where(z2 == np.max(np.nan_to_num(z2)))[1][0]]) * 180 / np.pi

    rho2, phi2 = cart2pol(dx, dy)

    dtheta = theta1 - theta2
    print(dtheta)
    return rho, phi, z, theta1, dx, dy


def hanningconv(interp_grid, n):
    kern = np.hanning(n).reshape(-1, 1)
    kern = kern * kern.T
    kern /= kern.sum()
    # fftconvolve is mathematically identical but exponentially faster for larger arrays
    hanning = signal.fftconvolve(interp_grid, kern, mode='same')
    return hanning

def hanningconv3d(interp_grid, n):
    kern = np.hanning(n).reshape(-1, 1)
    kern = kern * kern.T
    kern = kern[:, :, np.newaxis] * kern.T
    kern /= kern.sum()
    # fftconvolve handles n-dimensional arrays naturally and much faster
    hanning = signal.fftconvolve(interp_grid, kern, mode='same')
    return hanning


def interpolateData3d(z, dx, dy, dp, ncut, smooth=True):
    x_grid, y_grid, p_grid = np.meshgrid(dx, dy, dp)

    mask = ~np.isnan(z)  # Simplified mask generation
    x = x_grid[mask].ravel()
    y = y_grid[mask].ravel()
    p = p_grid[mask].ravel()
    points = np.array([x, y, p]).T
    values = z[mask].ravel()

    interp_grid = griddata(points, np.nan_to_num(values), (x_grid, y_grid, p_grid), method='nearest')
    if smooth:
        interp_grid = hanningconv3d(interp_grid, ncut)
    return interp_grid


def interpolateData(z, dx, dy, ncut, smooth=True):
    x_grid, y_grid = np.meshgrid(dx, dy)

    mask = ~np.isnan(z)
    x = x_grid[mask].ravel()
    y = y_grid[mask].ravel()
    points = np.array([x, y]).T
    values = z[mask].ravel()

    interp_grid = griddata(points, np.nan_to_num(values), (x_grid, y_grid), method='nearest')
    if smooth:
        interp_grid = hanningconv(interp_grid, ncut)

    return interp_grid


def SinCosPlot2(idx, spk, w_i, w_r, dphi, noise, ncut, smoothing_size, plotting=True):
    sin = w_i.ravel()
    cos = w_r.ravel()
    dphi = dphi.ravel()
    noise = noise.ravel()
    Y = spk.ravel()

    # Stack the variables exactly in the order they were unstacked/indexed
    sample = np.column_stack([sin, dphi, noise, cos])
    
    # OPTIMIZATION: N-Dimensional pure C binning. Huge RAM save.
    res = binned_statistic_2d(sample, Y, statistic='mean', bins=ncut)
    
    # Original pandas logic unstacked feature 0 (cos) to columns, and features 1,2,3 formed the index.
    # iloc[::-1] reversed the entire MultiIndex, which is mathematically equivalent to reversing the first 3 dimensions.
    z1 = res.statistic[::-1, ::-1, ::-1, :]

    # Calculate exactly matching midpoints directly from the mathematically precise edges
    dx = 0.5 * (res.bin_edges[3][:-1] + res.bin_edges[3][1:])             # cos (not reversed)
    dy = 0.5 * (res.bin_edges[0][:-1] + res.bin_edges[0][1:])[::-1]       # sin (reversed)
    dp = 0.5 * (res.bin_edges[1][:-1] + res.bin_edges[1][1:])[::-1]       # dphi (reversed)
    dn = 0.5 * (res.bin_edges[2][:-1] + res.bin_edges[2][1:])[::-1]       # noise (reversed)

    rho4d, phi4d, ddp4d, nn = cart2pol4d(dx, dy, dp, dn)
    rho3d, phi3d, ddp3d = cart2pol3d(dx, dy, dp)
    rho = rho3d[:, :, 0]
    phi = phi3d[:, :, 0]
    print(dx.shape, rho.shape)
    
    z = np.nanmean(z1, axis=2)
    z = np.moveaxis(z, 1, 2)
    H = interpolateData3d(z, dx, dy, dp, smoothing_size)

    if plotting:
        plt.figure()
        plt.imshow(np.nanmean(z, axis=2))
        fig, ax = plt.subplots(2, int(ncut / 2) + 1, subplot_kw={'projection': "polar"})
        max_val = np.max(H)
        for i in range(ncut)[::-1]:
            hanning = H[:, :, i]
            if i < ncut / 2:
                ax[0, i].pcolormesh(phi, rho, hanning[:-1, :-1], vmin=0, vmax=max_val, cmap='coolwarm', shading='flat')
                ax[0, i].plot(phi, rho, color='k', ls='none')
                ax[0, i].grid()

            if i >= ncut / 2:
                ax[1, i - int(ncut / 2)].pcolormesh(phi, rho, hanning[:-1, :-1], vmin=0, vmax=max_val, cmap='coolwarm', shading='flat')
                ax[1, i - int(ncut / 2)].plot(phi, rho, color='k', ls='none')
                ax[1, i - int(ncut / 2)].grid()

    a = 2
    if plotting:
        if a == 2:
            zz = np.nanmean(z, axis=a)
            plt.figure()
            plt.rcParams['axes.facecolor'] = 'none'
            plt.subplot(projection="polar")
            plt.pcolormesh(phi, rho, zz[:-1, :-1], cmap='coolwarm', shading='flat')
            plt.plot(phi, rho, color='k', ls='none')
            plt.grid()
            plt.title('Means of spks polar coordinate')
            
    z = H
    zz = np.nanmean(z, axis=a)
    hanning = zz
    if plotting:
        if a == 2:
            plt.figure()
            plt.rcParams['axes.facecolor'] = 'none'
            plt.subplot(projection="polar")
            plt.pcolormesh(phi, rho, zz[:-1, :-1], cmap='coolwarm', shading='flat')
            plt.plot(phi, rho, color='k', ls='none')
            plt.grid()
            plt.title('Means of spks polar coordinate')
            
            plt.figure()
            plt.rcParams['axes.facecolor'] = 'none'
            plt.subplot(projection="polar")
            plt.pcolormesh(phi, rho, hanning[:-1, :-1], cmap='coolwarm', shading='flat')
            plt.plot(phi, rho, color='k', ls='none')
            plt.grid()
            plt.title('Means of spks polar coordinate')

    zz = np.nanmean(z, axis=(0, 1))
    kern = np.hanning(smoothing_size)
    hanningzz = ndimage.convolve1d(zz, kern, 0)
    
    if plotting:
        plt.figure()
        plt.plot(dp * 180 / np.pi, hanningzz)
        
    z1 = np.moveaxis(z1, 3, 1)
    return rho3d, phi3d, ddp3d, nn, z, hanningzz, dx, dy, dp, dn


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def UnexpectedFiring(pred, Y, videodata):
    unfire = np.logical_and(np.logical_not(pred >= 0.20), Y >= 0.5)
    plt.figure()
    plt.imshow(np.mean(videodata[np.asarray(unfire).nonzero()[0]], axis=0))
    plt.figure()
    plt.rcParams['axes.facecolor'] = 'none'
    plt.plot(Y, c='k')
    plt.plot()


def diffSinCosPlot(idx, spks, x, y, o, s, w_i_downsampled, w_r_downsampled):
    sos = signal.butter(1, 100, 'lowpass', fs=1000, output='sos')
    sin = np.diff(w_i_downsampled[:, x, y, o, s], append=0).reshape(-1, 1)
    cos = np.diff(w_r_downsampled[:, x, y, o, s], append=0).reshape(-1, 1)

    sin_sos = signal.sosfilt(sos, np.diff(w_i_downsampled[:, x, y, o, s], append=0).ravel())
    cos_sos = signal.sosfilt(sos, np.diff(w_r_downsampled[:, x, y, o, s], append=0).ravel())

    plt.figure()
    plt.plot(sin)
    plt.plot(sin_sos)
    Y = np.mean(spks[:4, :], axis=0)[:, idx].ravel()

    nCut = 30
    
    # Pure array binned stats replacing the dataframe joins
    res = binned_statistic_2d(sin_sos, cos_sos, Y, statistic='mean', bins=nCut)
    z = res.statistic[::-1, :]
    
    # Construct strictly backward-compatible edges for the heatmap
    dx = 0.5 * (res.y_edge[:-1] + res.y_edge[1:])
    dy = 0.5 * (res.x_edge[:-1] + res.x_edge[1:])[::-1]
    
    # Only build a dataframe at the very end purely to utilize sns.heatmap's native interpolation
    means_z = pd.DataFrame(z, index=pd.Index(dy, name='sinBin'), columns=pd.Index(dx, name='cosBin'))

    plt.figure()
    plt.clf()
    sns.heatmap(means_z.interpolate(method='linear', limit=2, limit_direction='both'), cmap='coolwarm')
    plt.title('Means of spks vs Features cos and sin')
    plt.tight_layout()

    plt.figure()
    sns.jointplot(x=w_i_downsampled[:, x, y, o, s].ravel(), y=w_r_downsampled[:, x, y, o, s].ravel(), kind='kde', fill=True)


def compute_sta(a, b, ran):
    with torch.no_grad():
        a_t = torch.as_tensor(a, device='cuda', dtype=torch.float32)
        b_t = torch.as_tensor(b, device='cuda', dtype=torch.float32)
        
        a_sub = a_t[ran:].reshape(1, -1)
        a_sum = torch.sum(a_sub)
        
        # Pre-allocate output on CPU to save RAM instead of list comprehension
        c = np.empty((ran, 27, 11, 8), dtype=np.float32)
        
        for i, dt1 in enumerate(range(ran)[::-1]):
            b_sub = b_t[ran - dt1 : b_t.shape[0] - dt1].reshape(b_t.shape[0] - ran, -1)
            val = torch.sum(a_sub @ b_sub, dim=0) / a_sum
            c[i] = val.cpu().numpy().reshape((27, 11, 8))
            
    return c


def spikeTrig(spk, w_i, w_r, w_c, ran):
    spk_sub = spk[ran:]
    spk_sum = np.sum(spk_sub)
    spk_t = spk_sub.T
    
    mu_sin = np.empty(ran)
    mu_cos = np.empty(ran)
    mu_complex = np.empty(ran)
    
    for i, dt1 in enumerate(range(ran)[::-1]):
        mu_sin[i] = np.sum(spk_t * w_i[ran-dt1:w_i.shape[0]-dt1]) / spk_sum
        mu_cos[i] = np.sum(spk_t * w_r[ran-dt1:w_r.shape[0]-dt1]) / spk_sum
        mu_complex[i] = np.sum(spk_t * w_c[ran-dt1:w_c.shape[0]-dt1]) / spk_sum

    return mu_sin, mu_cos, mu_complex


def compute_stc(a, b, mu_b, ran, dt1, dt2):
    with torch.no_grad():
        a_t = torch.as_tensor(a, device='cuda')
        b_t = torch.as_tensor(b, device='cuda')
        mu_b_t = torch.as_tensor(mu_b, device='cuda')
        
        a_sub = a_t[ran - dt1:a_t.shape[0] - dt1].reshape(1, -1)
        b_sub1 = b_t[ran - dt1:b_t.shape[0] - dt1] - mu_b_t[mu_b_t.shape[0] - 1 - dt1]
        b_sub2 = b_t[ran - dt2:b_t.shape[0] - dt2] - mu_b_t[mu_b_t.shape[0] - 1 - dt2]
        
        term1 = torch.matmul(a_sub, b_sub1.reshape(b_t.shape[0] - ran, -1)).T
        term2 = torch.matmul(a_sub, b_sub2.reshape(b_t.shape[0] - ran, -1))
        
        c = (term1 @ term2) / torch.sum(a_sub)
        res = c.cpu().numpy().astype('float16')
        
        # Explicit VRAM memory release to prevent fragmentation
        del a_sub, b_sub1, b_sub2, term1, term2, c
        return res


def CovspikeTrig(spk, w, mu, ran):
    ran_val = np.max(np.abs(np.array(ran)))
    Css = np.zeros((ran_val, 54*135, ran_val, 54*135), dtype='float16')
    
    with torch.no_grad():
        spk_t = torch.as_tensor(spk, device='cuda', dtype=torch.float32)
        w_t = torch.as_tensor(w, device='cuda', dtype=torch.float32)
        mu_t = torch.as_tensor(mu, device='cuda', dtype=torch.float32)
        
        for dt1 in range(ran_val):
            for dt2 in range(ran_val):
                print(dt1, dt2)
                Css[dt1, :, dt2, :] = compute_stc(spk_t, w_t, mu_t, ran_val, dt1, dt2)
        
        # Clear main tensors and force PyTorch to dump VRAM
        del spk_t, w_t, mu_t
        torch.cuda.empty_cache()
        gc.collect()
                
    return Css


@njit(parallel=True, fastmath=True)
def CovspikeTrigC(spk, w_i, w_r, mu_i, mu_r, ran):
    ran_val = np.max(np.abs(np.array(ran)))
    Css = np.zeros((ran_val, ran_val))
    
    spk_sub = spk[ran_val:]
    spk_sum = np.sum(spk_sub)
    
    # prange distributes the outer loop across available CPU cores automatically
    for dt1 in prange(ran_val):
        for dt2 in range(ran_val):
            w_i_sub = w_i[ran_val-dt1:w_i.shape[0]-dt1] - mu_i[dt1]
            w_r_sub = w_r[ran_val-dt2:w_r.shape[0]-dt2] - mu_r[dt2]
            
            dot_prod = np.sum(spk_sub * w_i_sub * w_r_sub, axis=0)
            Css[dt1, dt2] = dot_prod / spk_sum

    return Css


def getSVDPolar(idx, spk, ncut, args, plotting=False, more_smooth=False, smoothing_size=5):
    rho, phi, dd, ns, zz, hanz, dx, dy, dp, dn= args
    hanz=hanz[::-1]
    f = interpolate.LinearNDInterpolator(np.stack((rho.flatten(), phi.flatten(), dd.flatten())).T, zz.flatten().T)
    if more_smooth:
        s=ncut
        rhospace=np.linspace(0, np.max(rho), s)
        phispace=np.linspace(0, np.max(phi), s)
        dphispace = np.linspace(np.min(dd), np.max(dd), s)
        nspace = np.linspace(np.min(ns), np.max(ns), s)

        xx, yy, tt = np.meshgrid(rhospace, phispace, dphispace)
        znew=f(xx.flatten(), yy.flatten(), tt.flatten()).reshape((s, s, s))
        mask = [~np.isnan(znew)][0]
        x = xx[mask].reshape(-1)
        y = yy[mask].reshape(-1)
        t = tt[mask].reshape(-1)
        points = np.array([x, y, t]).T
        values = znew[mask].reshape(-1)

        # generate interpolated grid data
        interp_grid = griddata(points, np.nan_to_num(values), (xx, yy, tt), method='nearest')
        interp_grid=hanningconv3d(interp_grid, smoothing_size)
        if plotting:
            plt.figure()
            plt.pcolormesh(rhospace, phispace, np.nanmean(interp_grid, axis=2)[:-1, :-1],  cmap='coolwarm')
            plt.plot(rhospace, phispace, color='k', ls='none')
            plt.colorbar()
            plt.title('interp_grid')


        f = interpolate.LinearNDInterpolator(np.stack((xx.flatten(), yy.flatten(), tt.flatten())).T, interp_grid.flatten().T)

        u, s, v = svds(np.nanmean(interp_grid, axis=2), 2)
        u1, s, v1 = svds(np.nan_to_num(interp_grid.reshape(-1, ncut)), 2)
        if plotting:
            plt.figure()
            plt.plot(rhospace, abs(v[1]))
            plt.title('rho tuning curve')
            plt.figure()
            plt.plot(phispace, abs(u[:, 1]))
            plt.title('phi tuning curve')



            plt.figure()
            plt.plot(dphispace, abs(v1[1]))
            plt.title('d phi / dt tuning curve')
            plt.figure()
            plt.imshow(abs(u1[:, 1]).reshape(ncut, ncut))


        if plotting:
            return v1[1], u1[:, 1], hanz, s[1], f #
        else:
            return (dphispace, abs(v1[1])), (phispace, abs(u[:, 1])), hanz, s[1], f
    else:
        return f, hanz


def nonvis(spks, idx):
    spk=np.mean(spks[[0, 2, 4], :, idx], axis=0)
    s_non_vis = np.array([spks[i, :, idx] - spk for i in range(5)])
    return s_non_vis


def deconvolve_avg_pop(spks, idx):
    pc_mean_pop=np.mean(spks, axis=2)

    s_non_vis=nonvis(spks, idx)
    Z=[]
    for i in range(5):
        y = s_non_vis.T[:, i]
        x = pc_mean_pop.T[:, i]
        h = ifft(fft(y) / fft(x))
        z = np.convolve(x, h)[:9000].reshape(-1, 1)
        Z.append(z)
    zt = np.stack(Z, axis=1)
    zt = np.real(zt.reshape(9000, 5).T)
    zt=np.array([zt[rep] - np.mean(zt[rep]) for rep in range(5)])
    return zt


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def getNonLinearModel(idx, spks, x, y, o, s, w_i_downsampled, w_r_downsampled, ncut):
    sin_w = w_i_downsampled[6:, x, y, o, s]
    cos_w = w_r_downsampled[6:, x, y, o, s]
    rho, phi, z, theta1, dx, dy= SinCosPlot(idx, spks[:, 6:, :], x, y, o, s, w_i_downsampled[6:], w_r_downsampled[6:], ncut)
    from scipy.stats import binned_statistic

    mean_rho = binned_statistic(np.nan_to_num(rho.flatten()), np.nan_to_num(z.flatten()),
                                statistic='max',
                                bins=10)
    mean_phi = binned_statistic(np.nan_to_num(phi.flatten()), np.nan_to_num(z.flatten()),
                                statistic='max',
                                bins=10)
    plt.figure()
    plt.plot(mean_rho[1][1:], mean_rho[0])
    plt.figure()
    plt.plot(mean_phi[1][1:], mean_phi[0])

    rp = np.array([cart2pol([cos_w[i]], [sin_w[i]]) for i in range(8994)])
    rh = rp[:, 0, 0, 0]
    ph = rp[:, 1, 0, 0]
    R = np.interp(rh, mean_rho[1][1:], mean_rho[0])
    P = np.interp(ph, mean_phi[1][1:], mean_phi[0])

    resp = R * P
    return resp


from scipy.stats import binned_statistic


def getNonLinearModel2(idx, spk, w_i, w_r, dphi, noise, ncut, smoothing_size, plotting=False, more_smooth=False):

    rr, pp, dd, nn, zz, hanz, dx, dy, dp, dn= SinCosPlot2(idx, spk, w_i, w_r,dphi,noise,  ncut,smoothing_size, plotting=plotting)

    args=(rr, pp,dd, nn, zz, hanz, dx, dy, dp, dn)

    print(plotting)
    arg2 =getSVDPolar(idx, spk, ncut, args, plotting=plotting, more_smooth=more_smooth, smoothing_size=smoothing_size)#
    print(arg2)
    if plotting:
        print(plotting)
        v, u, m_dp, s, f=arg2
        return v, u, m_dp, s, f, (dp, zz)  #

    else:
        print('no plotting')
        v, u, m_dp, s, f=arg2

        return v, u, m_dp, s, f


def computeNonlin(f, rho, phi, dphi):
    nonlinres=np.nan_to_num(f(rho, phi, dphi))#*dP.reshape(-1))
    return nonlinres


def computeNonlinMultiplicative(rh, ph, mean_rho, mean_phi, dphi,dp, m_dphi, ncut):
    rhospace = np.linspace(0, np.max(rh), ncut)
    phispace = np.linspace(0, np.max(ph), ncut)

    R = np.interp(rh, rhospace, mean_rho).reshape(-1)
    P = np.interp(ph, phispace, mean_phi).reshape(-1)
    dP= np.interp(dphi, dp[::-1], m_dphi[::-1]).reshape(-1)
    nonlinres = R * P * dP

    return nonlinres


from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings


def sigmoid(X1, *args):#a, b, w1, w2, w3): # Sigmoid A With Offset
    a=args[0]
    b=args[1]
    c=args[2]
    w=np.array([args[i] for i in range(3, len(args))]).reshape(1, -1)
    x=np.dot(w, X1.T)
    return  (c / (1.0 + np.exp(-a * (x-b)))).reshape(-1)


def relu(X1,*args): # Sigmoid A With Offset
    a = args[0]
    w = np.array([args[i] for i in range(1, len(args))]).reshape(1, -1)
    x = np.dot(w, X1.T)
    return np.clip(x,a, 10000).reshape(-1)


from sklearn.metrics import r2_score, explained_variance_score

def fitnonlin(X1, y_train, func):
    def sumOfSquaredError(parameterTuple):
        warnings.filterwarnings("ignore")  # do not print warnings by genetic algorithm
        val = func(X1, *parameterTuple)
        return np.sum((y_train - val) ** 2.0)

    def generate_Initial_Parameters(nb_params):
        maxX = np.max(X1)
        minX = np.min(X1)
        maxY = np.max(y_train)
        minY = np.min(y_train)

        parameterBounds = []
        if func==sigmoid:
            print('sigmoid')
            parameterBounds.append([0, 2])  # search bounds for a
            parameterBounds.append([0, 100])  # search bounds for b
            parameterBounds.append([0, 100]) # search bounds for c
            for n in range(nb_params):
                parameterBounds.append([-1e1, 1e1])
        elif func==relu:
            print('relu')
            parameterBounds.append([-2 * maxX, 2 * maxX])  # search bounds for a
            for n in range(nb_params):
                parameterBounds.append([-1e1, 1e1])
        print(parameterBounds)
        
        # OPTIMIZED: workers=-1 utilizes all available CPU threads for the genetic algorithm
        result = differential_evolution(sumOfSquaredError, parameterBounds, seed=2, workers=-1, updating='deferred')
        return result.x

    geneticParameters = generate_Initial_Parameters(X1.shape[1])
    print('geneticParameters')
    print(geneticParameters)

    # curve fit the test data
    fittedParameters, pcov = curve_fit(func, X1, y_train, geneticParameters, method='trf')

    print('Parameters', fittedParameters)

    modelPredictions = func(X1, *fittedParameters)
    if func==sigmoid:
        dx=np.linspace(-1, 5, 100)

    absError = modelPredictions - y_train

    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)  # mean squared errors
    RMSE = np.sqrt(MSE)  # Root Mean Squared Error, RMSE
    Rsquared = 1.0 - (np.var(absError) / np.var(y_train))
    print('RMSE:', RMSE)
    print('R-squared:', Rsquared)

    return fittedParameters, pcov, modelPredictions


def PlotR2scoreAnalysis(path, neuron_pos, respcorr):
    r=np.load(path)
    pearsons=r[:, 1]
    r2=r[:, 0]
    print(np.mean(r2), np.mean(pearsons))
    print(np.mean(r2[respcorr>=0.2]), np.mean(pearsons[respcorr>=0.2]))
    print(np.median(r2), np.median(pearsons))
    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=r2, vmax=0.1, vmin=0, cmap='Greys')
    plt.colorbar()
    plt.title('R2')
    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, c=pearsons, vmax=0.3, vmin=0, cmap='Greys')
    plt.colorbar()
    plt.title('pearsons')

    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.hist(pearsons, bins=30)
    plt.hist(pearsons[respcorr>=0.2], bins=30)
    plt.title('pearsons')
    plt.tight_layout()

    plt.figure()
    plt.rcParams['axes.facecolor']='none'
    plt.hist(r2, bins=30)
    plt.hist(r2[respcorr >= 0.2], bins=30)
    plt.title('r2')
    plt.tight_layout()

    return r, np.where(r2>=0.2)


def calculate_spike_triggered_covariance(spikes, stimulus, tau):
    X, T = stimulus.shape

    # Boolean indexing to avoid massive for-loop iteration over non-spikes
    spike_indices = np.where(spikes[tau:T] != 1)[0] + tau
    num_segments = len(spike_indices)
    
    if num_segments == 0:
        return np.zeros((X * tau, X * tau))

    # Pre-allocate array to avoid np.stack RAM spike
    segments = np.empty((num_segments, X * tau), dtype=np.float32)
    
    for i, t in enumerate(spike_indices):
        segments[i] = (spikes[t] * stimulus[:, t-tau:t]).reshape(-1)

    mean_segment = np.mean(segments, axis=0)
    centered_segments = segments - mean_segment

    covariance_matrix = np.zeros((X * tau, X * tau), dtype=np.float32)
    
    # TUNE THIS: Increase slice_size to push your GPU harder. 
    # 2000-5000 is usually a great sweet spot for modern GPUs to max out cores without OOM.
    slice_size = 2000 
    
    with torch.no_grad():
        # Keep main tensor in pinned CPU memory. This allows max-speed async transfers to GPU 
        # without overflowing VRAM by trying to load the whole dataset at once.
        cent_seg_t = torch.as_tensor(centered_segments, dtype=torch.float32).pin_memory()
        
        for start_i in range(0, X * tau, slice_size):
            print(f"Processing row block: {start_i}")
            end_i = min(start_i + slice_size, X * tau)
            
            # Move only the specific chunk to GPU as fast as possible
            slice_i = cent_seg_t[:, start_i:end_i].cuda(non_blocking=True)

            for start_j in range(start_i, X * tau, slice_size):
                end_j = min(start_j + slice_size, X * tau)
                slice_j = cent_seg_t[:, start_j:end_j].cuda(non_blocking=True)

                # GPU Matrix Math
                partial_cov = slice_i.t().mm(slice_j) / (num_segments - 1)
                cov_cpu = partial_cov.cpu().numpy()

                covariance_matrix[start_i:end_i, start_j:end_j] = cov_cpu
                if start_i != start_j:
                    covariance_matrix[start_j:end_j, start_i:end_i] = cov_cpu.T
                
                # Strict RAM/VRAM management: explicitly delete the inner chunk
                del slice_j 

            # Strict RAM/VRAM management: explicitly delete the outer chunk
            del slice_i 
            # Clear the cache to prevent PyTorch from hoarding VRAM and causing a crash
            torch.cuda.empty_cache() 

    return covariance_matrix


def on_pick(event):
    artist = event.artist
    xmouse, ymouse = event.mouseevent.xdata, event.mouseevent.ydata
    x, y = artist.get_xdata(), artist.get_ydata()
    ind = event.ind
    print ('idx:', event.artist)


def gaussian(x, mu, sig):
    return (
        1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2)
    )

def sigma_func(x, a, b):
    return (1.0 / (1.0 + np.exp(-a * (x - b))))


def create_fake_cell( w_i_downsampled, w_r_downsampled, pos_angle_scale, phase_t_shift, phase=np.pi/3, thresh=0.075, ncut=30, dt=5400, plotting=False):
    x, y, o, s=pos_angle_scale
    w_i = w_i_downsampled[:dt, x, y, o, s]  
    w_r = w_r_downsampled[:dt, x, y, o, s]  

    # SUPER-VECTORIZED: Bypassing the pure python for-loop and cart2pol function entirely
    # This computes polar coords over the whole array instantly in C.
    rho = np.hypot(w_r, w_i)
    phi = np.arctan2(w_i, w_r)
    
    dphi = np.diff(phi, prepend=0)
    dphi[abs(dphi) >= 3] = np.nan
    nans, _x = nan_helper(dphi)
    dphi[nans] = np.interp(_x(nans), _x(~nans), dphi[~nans])

    rhospace = np.linspace(0, 0.2, ncut)
    phispace = np.linspace(0, 2*np.pi, ncut)
    dphispace = np.linspace(-180, 180, ncut)

    interp_grid=(((1+np.cos(phispace+phase)).reshape(-1, 1)*sigma_func(rhospace, 100, thresh).reshape(1, -1)).reshape(-1, 1)*(2.5*gaussian(dphispace, phase_t_shift, 150)).reshape(1,-1)).reshape(ncut, ncut, ncut)
    interp_grid=np.clip(interp_grid, 0, None)
    xx, yy, tt = np.meshgrid(rhospace, phispace, dphispace)

    if plotting:
        plt.figure()
        plt.pcolormesh(rhospace, phispace, np.nanmean(interp_grid, axis=2)[:-1, :-1], cmap='coolwarm')
        plt.plot(rhospace, phispace, color='k', ls='none')
        plt.colorbar()

    try:
        f = interpolate.LinearNDInterpolator(np.stack((xx.flatten(), yy.flatten(), tt.flatten())).T,
                                             interp_grid.flatten().T)
    except ValueError as e:
        print(e)

    nonlinres = computeNonlin(f, rho, phi, dphi)
    return nonlinres


def PlotSelfCorrelation(w_c_downsampled,neuron_pos, pos_ori, ns=4):
    x, y, o, s = pos_ori
    rfs = PearsonCorrelationPinkNoise(w_c_downsampled.reshape(w_c_downsampled.shape[0], -1),
                                      w_c_downsampled[:, x, y, o, s].reshape(w_c_downsampled.shape[0], -1),
                                      neuron_pos, 27, 11, ns)
    rfs_idx = rfs[0].reshape(27, 11, 8, ns)
    maxes = np.array(rfs[1])
    fig, ax = plt.subplots(8, ns)
    cc = 0
    r=skimage.transform.resize(rfs_idx, (135, 54, 8, ns), anti_aliasing=True)
    vmax = 1
    vmin = -vmax
    for i in range(8):
        for j in range(ns):
            ax[i, j].imshow(rfs_idx[:, :, i, j].T, vmin=vmin, vmax=vmax, cmap='coolwarm')


def Plot_RF(rfs_idx, ns=4, title=''):


    fig, ax = plt.subplots(8, ns)
    plt.title(title)
    cc = 0
    vmax = np.max(rfs_idx)
    vmin = -vmax
    for i in range(8):
        for j in range(ns):
            ax[i, j].imshow(rfs_idx[:, :, i, j].T, vmin=vmin, vmax=vmax, cmap='coolwarm')


def gaus(x,a,x0,sigma, offset):
    return (a*exp(-(x-x0)**2/(2*sigma**2))) + offset


def fit_gaussian_params(x_m_phi, plotting=False):
    x=x_m_phi[0]
    y=abs(x_m_phi[1])
    n = len(x)  # the number of data
    mean = sum(x * y) / n  # note this correction
    sigma = sum(y * (x - mean) ** 2) / n  # note this correction
    popt, pcov = curve_fit(gaus, x, y, p0=[1, mean, sigma, 0])
    if plotting:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x,gaus(x,*popt))
    return popt


import scipy
import scipy.cluster.hierarchy as sch


def cluster_corr(corr_array, inplace=False):
    """
    Rearranges the correlation matrix, corr_array, so that groups of highly
    correlated variables are next to eachother

    Parameters
    ----------
    corr_array : pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        a NxN correlation matrix with the columns and rows rearranged
    """
    pairwise_distances = sch.distance.pdist(corr_array)
    linkage = sch.linkage(pairwise_distances, method='complete')
    cluster_distance_threshold = pairwise_distances.max() / 2
    idx_to_cluster_array = sch.fcluster(linkage, cluster_distance_threshold,
                                        criterion='distance')
    idx = np.argsort(idx_to_cluster_array)

    if not inplace:
        corr_array = corr_array.copy()

    if isinstance(corr_array, pd.DataFrame):
        return corr_array.iloc[idx, :].T.iloc[idx, :]
    return corr_array[idx, :][:, idx], idx_to_cluster_array


def FEVE(gt, pred):
    sig=np.mean(np.var(gt, axis=0))
    absError = pred - np.mean(gt, axis=0)
    SE = np.square(absError)  # squared errors
    MSE = np.mean(SE)
    num=MSE-np.square(sig)
    denom=np.var(gt)-np.square(sig)
    feve=1-(num/denom)
    return feve


def rolling_avg(arr, win):
    import scipy.signal as sig
    kernal = np.ones(win, dtype=('float'))
    padsize = arr.shape[0] + win * 2
    mov_pad = np.zeros([padsize], dtype=('float'))
    mov_pad[win:(padsize-win)] = arr
    mov_ave = sig.fftconvolve(mov_pad, kernal) / win
    return mov_ave


def hanningconvnd(interp_grid, n):
    kern = np.hanning(n).reshape(-1, 1)
    kern = kern * kern.T
    kern=kern[:, :, np.newaxis]* kern.T
    kern /= kern.sum()  # normalize the kernel weights to sum to 1
    hanning = ndimage.convolve(interp_grid, kern)
    return hanning


from scipy.interpolate import NearestNDInterpolator,LinearNDInterpolator
from scipy.ndimage import gaussian_filter


def interpolateDatand(z, dx, dy, dp, dx_h, dy_h, dp_h, ncut, smooth=True):
    x_grid, y_grid, p_grid,xh_grid, yh_grid, ph_grid = np.meshgrid(dx, dy, dp,dx_h, dy_h, dp_h)

    # get known values to set the interpolator
    mask = [~np.isnan(z)][0]
    x = x_grid[mask].reshape(-1)
    y = y_grid[mask].reshape(-1)
    p = p_grid[mask].reshape(-1)
    xh = xh_grid[mask].reshape(-1)
    yh = yh_grid[mask].reshape(-1)
    ph = ph_grid[mask].reshape(-1)
    points = np.array([x, y, p, xh, yh, ph]).T
    values = z[mask].reshape(-1)

    # generate interpolated grid data
    interp = NearestNDInterpolator(list(zip(x, y, p, xh, yh, ph)), values)

    interp_grid = interp(x_grid, y_grid, p_grid, xh_grid, yh_grid, ph_grid)
    if smooth:
        interp_grid=gaussian_filter(interp_grid, sigma=ncut)
    interp = LinearNDInterpolator(list(zip(x_grid.reshape(-1), y_grid.reshape(-1), p_grid.reshape(-1), xh_grid.reshape(-1), yh_grid.reshape(-1), ph_grid.reshape(-1))), interp_grid)

    return interp_grid, interp


def SinCosPlot3( spk, w_i, w_r, dphi, w_i_inhib, w_r_inhib, dphi_inhib, ncut, smoothing_size, plotting=True):
    sin=w_i.reshape(-1,1)
    cos=w_r.reshape(-1,1)
    dphi=dphi.reshape(-1, 1)

    sin_h = w_i_inhib.reshape(-1, 1)
    cos_h = w_r_inhib.reshape(-1, 1)
    dphi_h = dphi_inhib.reshape(-1, 1)


    Y = spk.reshape(-1, 1)
    Y_past=np.zeros(Y.shape)
    Y_past[:-5]=Y[5:]

    nCut=ncut
    data=np.concatenate([cos, sin, dphi, sin_h, cos_h, dphi_h], axis=1)#, columns=['sin', 'cos', 'sig'])
    histo, edges=np.histogramdd(data, bins=ncut, density=False, weights=Y[:, 0])
    histo_, edges_ = np.histogramdd(data, bins=ncut)
    z=histo/histo_
    dx=(edges[0][:-1] + edges[0][1:]) / 2
    dy=(edges[1][:-1] + edges[1][1:]) / 2
    dp = (edges[2][:-1] + edges[2][1:]) / 2
    dx_h = (edges[3][:-1] + edges[3][1:]) / 2
    dy_h =(edges[4][:-1] + edges[4][1:]) / 2
    dp_h = (edges[5][:-1] + edges[5][1:]) / 2
    interp_grid, interp=interpolateDatand(z, dx, dy, dp, dx_h, dy_h, dp_h)

    return interp_grid,z, dx, dy, dp, dx_h, dy_h, dp_h


def getNonLinearModel3(idx, spk, w_i, w_r, dphi, noise, ncut, smoothing_size, plotting=False, more_smooth=False):

    interp_grid,z, dx, dy, dp, dx_h, dy_h, dp_h= SinCosPlot3(idx, spk, w_i, w_r,dphi,noise,  ncut,smoothing_size, plotting=plotting)
    x_grid, y_grid, p_grid, xh_grid, yh_grid, ph_grid = np.meshgrid(dx, dy, dp, dx_h, dy_h, dp_h)
    f = interpolate.LinearNDInterpolator(np.stack((x_grid.flatten(), y_grid.flatten(), p_grid.flatten(), xh_grid.flatten(), yh_grid.flatten(), ph_grid.flatten())).T, interp_grid.flatten().T)
    return f


def getmetrics(x, y, n, frames_per_minute=None):
    if frames_per_minute is None:
        frames_per_minute = DEFAULT_FRAMES_PER_MINUTE
    frames_per_minute = int(frames_per_minute)
    ev = explained_variance_score(
        np.mean(y.reshape(n, frames_per_minute), axis=0),
        x,
        multioutput='uniform_average',
    )
    feve = FEVE(y.reshape(n, frames_per_minute), x)
    cc = np.corrcoef(np.mean(y.reshape(n, frames_per_minute), axis=0), x)
    print(feve)
    print(ev)
    print(cc[0][1])
    return feve, ev,cc

def getpolar(cos, sin):
    rho = np.sqrt(cos ** 2 + sin ** 2)
    temp_phi = np.arctan2(sin, cos)
    shift = (2 * np.pi) * (temp_phi < 0)
    phi = np.arctan2(sin, cos) + shift
    return rho, phi

import scipy.fftpack
from scipy.signal import find_peaks


def gaussian_smooth(y, sigma = 2):
    kernel_size = 2 * int(3 * sigma) + 1
    gaussian_kernel = np.exp(-0.5 * (np.linspace(-3, 3, kernel_size) / sigma) ** 2)
    gaussian_kernel /= gaussian_kernel.sum()  # Normalisation

    # Convolution (conserve intensity)
    y_smooth = np.convolve(y, gaussian_kernel, mode='full')
    print(y.shape, y_smooth.shape, gaussian_kernel.shape)
    y_smooth=y_smooth[int((gaussian_kernel.shape[0]-1)/2)-1:-int((gaussian_kernel.shape[0]-1)/2)-1]
    print(y_smooth.shape)
    return y_smooth

def approx_Matrix(X, plotting=False):

    X=np.clip(X, 0, None)
    model1 = NMF(n_components=1, init='random', random_state=42)
    U1 = model1.fit_transform(X.reshape(X.shape[0], -1))[:, 0]

    model2 = NMF(n_components=1, init='random', random_state=42)
    U2 = model2.fit_transform(X.transpose(1, 0, 2).reshape(X.shape[1], -1))[:, 0]

    model3 = NMF(n_components=1, init='random', random_state=42)
    U3 = model3.fit_transform(X.transpose(2, 0, 1).reshape(X.shape[2], -1))[:, 0]
    if plotting:
        plt.figure()
        plt.plot(U1)
        plt.figure()
        plt.plot(U2)
        plt.figure()
        plt.plot(U3)

    U1=gaussian_smooth(U1, 1)
    U2 = gaussian_smooth(U2, 1)
    U3 = gaussian_smooth(U3, 1)

    if plotting:
        plt.figure()
        plt.plot(U1)
        plt.figure()
        plt.plot(U2)
        plt.figure()
        plt.plot(U3)
    # Approximation
    X_approx = np.einsum('i,j,k -> ijk', U1, U2, U3)

    error = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
    print(f"Erreur relative : {error:.4f}")

    return X_approx, (U1, U2, U3)

def approx_Matrix2(X, smoothing_factor=0.75, plotting=False):
    from tensorly.decomposition import non_negative_parafac

    weights, factors = non_negative_parafac(X, rank=1, init='random', normalize_factors=False)
    U1, U2, U3 = factors
    print(U1.shape, U2.shape, U3.shape)

    U1=U1.reshape(-1)
    U2=U2.reshape(-1)
    U3=U3.reshape(-1)

    if smoothing_factor!=None:
        U1 = gaussian_smooth(U1, smoothing_factor)
        U2 = gaussian_smooth(U2, smoothing_factor)
        U3 = gaussian_smooth(U3, smoothing_factor)

    X_approx = np.einsum('i,j,k -> ijk', U1, U2, U3)
    error = np.linalg.norm(X - X_approx) / np.linalg.norm(X)
    print(f"Erreur relative : {error:.4f}")

    return X_approx,  (U1, U2, U3)
    

def getPhiRho(spk, w_i, w_r, dphi, w_i_inhib, w_r_inhib, dphi_inhib, ncut=20, plotting=True, sigma=7):
    sin = w_i.reshape(-1, 1)
    cos = w_r.reshape(-1, 1)
    dphi = dphi.reshape(-1, 1)
    rho, phi = getpolar(sin, cos)

    sin_h = w_i_inhib.reshape(-1, 1)
    cos_h = w_r_inhib.reshape(-1, 1)
    dphi_h = dphi_inhib.reshape(-1, 1)
    rho_h, phi_h = getpolar(sin_h, cos_h)

    a = abs(max(rho.min(), rho.max()))
    c = abs(max(phi.min(), phi.max()))
    print('cos : ', cos.min(), cos.max())
    print('sin : ', sin.min(), sin.max())
    print('phi : ', phi.min(), phi.max())
    print('rho : ', rho.min(), rho.max())
    if a == 0:
        a = 0.3
    b = abs(max(dphi.min(), dphi.max()))
    d = abs(max(cos.min(), cos.max()))
    e = abs(max(sin.min(), sin.max()))
    d = max(d, e)
    print('dphi : ', dphi.min(), dphi.max())
    print(a, b, c, d)
    if b == 0:
        b == 1
        
    E = [np.linspace(0, a, ncut + 1), np.linspace(0, c, ncut + 1), np.linspace(-b, b, ncut + 1)]
    Ecs = [np.linspace(-d, d, ncut + 1), np.linspace(-d, d, ncut + 1)]
    
    # ---------------------------------------------------------
    # OPTIMIZATION 1: Multithreaded histogram loops
    # ---------------------------------------------------------
    n_spk = spk.shape[0]
    Hcs = np.empty((n_spk, ncut, ncut))
    Hcs_ = np.empty((n_spk, ncut, ncut))
    H = np.empty((n_spk, ncut, ncut, ncut))
    H_ = np.empty((n_spk, ncut, ncut, ncut))

    data = np.concatenate([rho, phi, dphi], axis=1)
    datacs = np.concatenate([cos, sin], axis=1)

    # Worker function to run histograms outside GIL constraints
    def compute_hist_1(i):
        Y = spk[i].reshape(-1, 1)
        _hcs, _ = np.histogramdd(datacs, bins=Ecs, density=False, weights=Y[:, 0])
        _hcs_, _ = np.histogramdd(datacs, bins=Ecs)
        _h, _ = np.histogramdd(data, bins=E, density=False, weights=Y[:, 0])
        _h_, _ = np.histogramdd(data, bins=E)
        return i, _hcs, _hcs_, _h, _h_

    # Dynamically allocate threads based on your CPU
    threads = min(32, (os.cpu_count() or 1) + 4)
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, _hcs, _hcs_, _h, _h_ in executor.map(compute_hist_1, range(n_spk)):
            Hcs[i] = _hcs
            Hcs_[i] = _hcs_
            H[i] = _h
            H_[i] = _h_

    Hcs = np.nanmean(Hcs, axis=0)
    Hcs_ = np.nanmean(Hcs_, axis=0)
    Zcs = Hcs / Hcs_
    print(Zcs.shape)
    
    H = np.nanmean(H, axis=0)
    H_ = np.nanmean(H_, axis=0)
    
    H = np.concatenate((H, H, H), axis=1)
    H_ = np.concatenate((H_, H_, H_), axis=1)
    Z = hanningconv3d(H, sigma) / hanningconv3d(H_, sigma)
    Z = Z[:, :int(Z.shape[1]/3), :]
    
    xedges = E[0]
    yedges = E[1]
    zedges = E[2]
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    zcenters = (zedges[:-1] + zedges[1:]) / 2

    mask = np.where(~np.isnan(Z))
    interp = NearestNDInterpolator(np.transpose((xcenters[mask[0]], ycenters[mask[1]], zcenters[mask[2]])), Z[mask])
    indices = np.indices(Z.shape)
    indices = np.stack([xcenters[indices[0]], ycenters[indices[1]], zcenters[indices[2]]])
    filled_data0 = interp(*indices)
    filled_data2, plot = approx_Matrix2(filled_data0, None, plotting=plotting)
    plots = [plot[0], np.append(plot[1], plot[1][0]), plot[2]]

    filled_data = np.concatenate((filled_data0, filled_data0, filled_data0), axis=1)

    d_val = zcenters[np.argmax(plot[2])]
    pp = plot[1]
    rr = plot[0]
    HMP = xcenters[np.argmin(abs(rr - (np.max(rr)/2)))]
    cv = circular_variance(np.linspace(0, 360, 20), pp.reshape(-1, 1))
    complexity = 1 - cv[1][0]

    if plotting:
        fig = plt.figure(figsize=(16, 6))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        plt.rcParams.update({
            "font.size": 8,
            "svg.fonttype": "none"
        })

        inner_gs0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0])
        ax1_0 = fig.add_subplot(inner_gs0[0], projection='3d')
        a1_1 = fig.add_subplot(inner_gs0[1])
        threshold = np.nanpercentile(Zcs.flatten(), 95)
        print('threshold ', threshold)
        m = a1_1.imshow(Zcs, vmin=0, vmax=threshold, cmap='bone_r')
        a1_1.set_xticks(np.arange(20))
        a1_1.set_xticklabels(10*(0.5 * (Ecs[0][1:] + Ecs[0][:-1])).astype(int) / 10, rotation=45)
        a1_1.set_yticks(np.arange(20))
        a1_1.set_yticklabels(10*(0.5 * (Ecs[0][1:] + Ecs[0][:-1])).astype(int) / 10)
        a1_1.set_xlabel('Sine wavelet (a.u)')
        a1_1.set_ylabel('Cosine wavelet (a.u)')
        a1_1.set_title('Firing rate histogram')
        fig.colorbar(m, ax=a1_1)

        colors = []
        for i in np.arange(8000, 9000, 1):
            c_val = np.mean(spk, axis=0)[i]
            color = plt.cm.bone_r(255 * c_val / 100)
            colors.append(color)
            ax1_0.scatter(rho[i - 1:i + 1], phi[i - 1:i + 1], dphi[i - 1:i + 1], s=10, color=color)
            ax1_0.set_xlabel('Amplitude (a.u)')
            ax1_0.set_ylabel('Phase (rad)')
            ax1_0.set_zlabel('Drift (rad/s)')
            ax1_0.set_title('Firing rate trajectory')
        fig.colorbar(m, ax=ax1_0)

        inner_gs1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 0])
        ax3_0 = fig.add_subplot(inner_gs1[0])
        ax3_1 = fig.add_subplot(inner_gs1[1])
        ax3_2 = fig.add_subplot(inner_gs1[2])
        ax3_0.plot(xcenters, np.nanmean(filled_data0, axis=(2, 1)))
        ax3_0.set_title('rho')
        ax3_1.plot(ycenters, np.nanmean(filled_data0, axis=(2, 0)))
        ax3_1.set_title('phi')
        ax3_1.set_ylim(bottom=0)
        ax3_2.plot(zcenters, np.nanmean(filled_data0, axis=(0, 1)))
        ax3_2.set_title('dphi')

        inner_gs2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1])
        ax4_0 = fig.add_subplot(inner_gs2[0])
        ax4_1 = fig.add_subplot(inner_gs2[1])
        ax4_2 = fig.add_subplot(inner_gs2[2])
        m1 = ax4_0.imshow(np.nanmean(filled_data0, axis=2).T, cmap='bone_r')
        ax4_0.set_xticks(range(20))
        ax4_0.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax4_0.set_yticks(range(20))
        ax4_0.set_yticklabels((ycenters * 100).astype(int) / 100)
        ax4_0.set_xlabel('rho')
        ax4_0.set_ylabel('phi')
        fig.colorbar(m1, ax=ax4_0)

        m2 = ax4_1.imshow(np.nanmean(filled_data0, axis=1).T, cmap='bone_r')
        ax4_1.set_xticks(range(20))
        ax4_1.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax4_1.set_yticks(range(20))
        ax4_1.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax4_1.set_xlabel('rho')
        ax4_1.set_ylabel('dphi')
        fig.colorbar(m2, ax=ax4_1)

        m3 = ax4_2.imshow(np.nanmean(filled_data0, axis=0).T, cmap='bone_r')
        ax4_2.set_xticks(range(20))
        ax4_2.set_xticklabels((ycenters * 100).astype(int) / 100, rotation=90)
        ax4_2.set_yticks(range(20))
        ax4_2.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax4_2.set_xlabel('phi')
        ax4_2.set_ylabel('dphi')
        fig.colorbar(m3, ax=ax4_2)

        inner_gs3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 1])
        ax5_0 = fig.add_subplot(inner_gs3[0])
        ax5_1 = fig.add_subplot(inner_gs3[1])
        ax5_2 = fig.add_subplot(inner_gs3[2])
        m4 = ax5_0.imshow(np.nanmean(Z, axis=2).T, cmap='coolwarm')
        ax5_0.set_xticks(range(20))
        ax5_0.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax5_0.set_yticks(range(20))
        ax5_0.set_yticklabels((ycenters * 100).astype(int) / 100)
        ax5_0.set_xlabel('rho')
        ax5_0.set_ylabel('phi')
        fig.colorbar(m4, ax=ax5_0)

        m5 = ax5_1.imshow(np.nanmean(Z, axis=1).T, cmap='coolwarm')
        ax5_1.set_xticks(range(20))
        ax5_1.set_xticklabels((xcenters * 100).astype(int) / 100, rotation=90)
        ax5_1.set_yticks(range(20))
        ax5_1.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax5_1.set_xlabel('rho')
        ax5_1.set_ylabel('dphi')
        fig.colorbar(m5, ax=ax5_1)

        m6 = ax5_2.imshow(np.nanmean(Z, axis=0).T, cmap='coolwarm')
        ax5_2.set_xticks(range(20))
        ax5_2.set_xticklabels((ycenters * 100).astype(int) / 100, rotation=90)
        ax5_2.set_yticks(range(20))
        ax5_2.set_yticklabels((zcenters * 100).astype(int) / 100)
        ax5_2.set_xlabel('phi')
        ax5_2.set_ylabel('dphi')
        fig.colorbar(m6, ax=ax5_2)

        plt.tight_layout()
        plt.show()

    a = abs(max(rho_h.min(), rho_h.max()))
    c = abs(max(phi_h.min(), phi_h.max()))
    if a == 0:
        a = 0.3
    b = abs(max(dphi_h.min(), dphi_h.max()))
    if b == 0:
        b = 1
    E = [np.linspace(0, a, ncut + 1), np.linspace(0, c, ncut + 1), np.linspace(-b, b, ncut + 1)]
    
    # ---------------------------------------------------------
    # OPTIMIZATION 2: Multithreaded histogram loop 2
    # ---------------------------------------------------------
    H_inhib = np.empty((n_spk, ncut, ncut, ncut))
    H_inhib_ = np.empty((n_spk, ncut, ncut, ncut))
    data_h = np.concatenate([rho_h, phi_h, dphi_h], axis=1)

    def compute_hist_2(i):
        Y = spk[i].reshape(-1, 1)
        _h_inh, _ = np.histogramdd(data_h, bins=E, density=False, weights=Y[:, 0])
        _h_inh_, _ = np.histogramdd(data_h, bins=E)
        return i, _h_inh, _h_inh_

    with ThreadPoolExecutor(max_workers=threads) as executor:
        for i, _h_inh, _h_inh_ in executor.map(compute_hist_2, range(n_spk)):
            H_inhib[i] = _h_inh
            H_inhib_[i] = _h_inh_

    H = np.mean(H_inhib, axis=0)
    H_ = np.mean(H_inhib_, axis=0)
    
    Z = hanningconv3d(H, 3) / hanningconv3d(H_, 3)
    xedges = E[0]
    yedges = E[1]
    zedges = E[2]
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2
    zcenters = (zedges[:-1] + zedges[1:]) / 2

    mask = np.where(~np.isnan(Z))
    interp_h = NearestNDInterpolator(np.transpose((xcenters[mask[0]], ycenters[mask[1]], zcenters[mask[2]])), Z[mask])
    indices = np.indices(Z.shape)
    indices = np.stack([xcenters[indices[0]], ycenters[indices[1]], zcenters[indices[2]]])
    filled_data = interp_h(*indices)
    filled_data, plot_h = approx_Matrix2(filled_data, plotting=plotting)

    d_h = zcenters[np.argmax(plot[2])] 

    pp = plot_h[1]
    rr = plot_h[0]
    HMP_f = xcenters[np.argmin(abs(rr - (np.max(rr)/2)))]
    cv = circular_variance(np.linspace(0, 360, 20), pp.reshape(-1, 1))
    complexity_f = 1 - cv[1][0]

    print('linearity: ' , complexity, complexity_f)

    return interp, interp_h, d_val, d_h, complexity, complexity_f, HMP, HMP_f, [plots, plot_h]


def GetNeuronVisresponse(idx, w_i, w_r, w_i_inhib, w_r_inhib, dphi, dphi_inhib,
                         spks, n_min, double_wavelet_model, dt1=9000,
                         train_idx=[0, 2, 4], test_idx=[1, 3],
                         lastmin=False, func=relu, sigma=7, plotting=False,
                         frames_per_minute=None) :
    if frames_per_minute is None:
        frames_per_minute = DEFAULT_FRAMES_PER_MINUTE
    frames_per_minute = int(frames_per_minute)
    spk = spks[:, :n_min * frames_per_minute, idx]
    if lastmin:
        dt1 = n_min * frames_per_minute
    y_train = spks[train_idx, :dt1, idx]
    y_test = spks[test_idx, :dt1, idx]
    rho, phi = getpolar(w_i, w_r)  # [:dt1]
    f, f_h, d, d_h, c, c_h, hmp, hmp_h, plot= getPhiRho(y_train[:, :dt1], w_i[:dt1], w_r[:dt1], dphi[:dt1], w_i_inhib[:dt1], w_r_inhib[:dt1], dphi_inhib[:dt1], plotting=plotting, sigma=sigma)




    rho_h, phi_h = getpolar( w_i_inhib, w_r_inhib)#[:dt1]
    # --- Training Data Preparation ---
    pred = f(rho, phi, dphi)
    pred_h = f_h(rho_h, phi_h, dphi_inhib.reshape(-1, 1))

    # Vectorized tiling saves memory and CPU time compared to list comprehensions
    base_X1 = np.concatenate((pred[:dt1], pred_h[:dt1]), axis=1)
    X1 = np.tile(base_X1, (len(train_idx), 1))
    X1 = np.nan_to_num(X1)

    # Ravel flattens the array using a memory view (zero RAM cost if contiguous)
    y_train = spks[train_idx, :dt1, idx].ravel()
    y_test = spks[test_idx, :dt1, idx].ravel()
    
    fittedParameters, pcov, res = fitnonlin(X1, y_train, func)

    fittedParameters, pcov, res = fitnonlin(X1, y_train, func)

    print('prediction on training set : ')
    res1 = res  # + (w_pc*pcs_test[:, 0])
    res2 = np.mean(res1.reshape(len(train_idx), dt1), axis=0)
    ev = explained_variance_score(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2, multioutput='uniform_average')
    feve = FEVE(y_train.reshape(len(train_idx), dt1), res2)
    cc_train=np.corrcoef(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2)[0][1]
    print(feve)
    print(ev)
    print(cc_train)

    # --- Test Data Preparation ---
    print('prediction acress repeats : ')
    # Reuse the same tiling logic for the test indices
    X1 = np.tile(base_X1, (len(test_idx), 1))
    X1 = np.nan_to_num(X1)
    
    print(X1.shape)
    unrectified = np.mean(X1[:, 0].reshape(len(test_idx), dt1), axis=0)
    unrectified2 = np.mean(X1[:, 1].reshape(len(test_idx), dt1), axis=0)
    res = func(X1, *fittedParameters)
    print(res.shape,unrectified.shape)

    res1 = res  # + (w_pc*pcs_test[:, 0])
    w = 0
    if double_wavelet_model:
        res2 = np.mean(res1.reshape(len(test_idx), dt1), axis=0)
    else:
        print('single wavelet')
        res21=unrectified
        cc1 = np.corrcoef(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res21)[0,1]
        print(cc1)
        res22 = unrectified2
        cc2 = np.corrcoef(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res22)[0,1]
        print(cc2)
        if cc1>=cc2:
            print('1st')
            res2=unrectified
        else:
            print('2nd')
            w = 1
            res2=unrectified2
    ev = explained_variance_score(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res2, multioutput='uniform_average')
    feve = FEVE(y_test.reshape(len(test_idx), dt1), res2)
    cc = np.corrcoef(np.mean(y_test.reshape(len(test_idx), dt1), axis=0), res2)
    print(feve)
    print(ev)
    print(cc)

    print('prediction last minute : ')
    tp_test = dt1 + frames_per_minute
    print(tp_test, dt1)
    X1 = np.concatenate([(np.concatenate(
        (pred[tp_test:tp_test + frames_per_minute],
         pred_h[tp_test:tp_test + frames_per_minute]), axis=1))
        for rep in test_idx])

    X1 = np.nan_to_num(X1)
    print(pred.shape)
    y_test = np.concatenate([
        spks[r, tp_test:tp_test + frames_per_minute, idx]
        for r in train_idx
    ])
    res = func(X1, *fittedParameters)
    res1 = res  # + (w_pc*pcs_test[:, 0])
    reslastmin = np.mean(res1.reshape(len(test_idx), frames_per_minute), axis=0)
    print(np.mean(y_test.reshape(len(test_idx),  dt1-tp_test), axis=0).shape, reslastmin.shape)
    evlastmin= explained_variance_score(np.mean(y_test.reshape(len(test_idx),  dt1-tp_test), axis=0), reslastmin, multioutput='uniform_average')
    fevelastmin = FEVE(y_test.reshape(len(test_idx), dt1-tp_test), reslastmin)
    cclastmin=np.corrcoef(np.mean(y_test.reshape(len(test_idx),  dt1-tp_test), axis=0), reslastmin)[0,1]
    print(feve)
    print(ev)
    print(cclastmin)
    return res2, [feve, ev, cc, cc_train, cclastmin], fittedParameters, [d, c, hmp, d_h, c_h, hmp_h], plot, unrectified, w, f


def PredictNeuronsTest(wt_test, spks, idx, ncut, dt1=9000, func=relu):
    test_idx = [0, 2]
    train_idx=[1, 3]
    spk = spks[:, :, idx]
    H = []
    H_ = []
    for i in [0, 2]:
        Y = spk[i].reshape(-1, 1)
        a = abs(max(wt_test.min(), wt_test.max()))
        histo, edges = np.histogram(wt_test, bins=np.linspace(wt_test.min(), wt_test.max(), ncut + 1), density=False, weights=Y[:, 0])
        histo_, edges_ = np.histogram(wt_test, bins=np.linspace(wt_test.min(), wt_test.max(), ncut + 1))
        H.append(histo)
        H_.append(histo_)

    H = np.mean(np.array(H), axis=0)
    H_ = np.mean(np.array(H_), axis=0)
    Z = H / H_

    xedges = np.linspace(wt_test.min(), wt_test.max(), ncut + 1)
    xcenters = (xedges[:-1] + xedges[1:]) / 2

    plt.figure()
    plt.plot(xcenters,Z)
    plt.plot(xcenters,H)
    plt.plot(xcenters,H_)
    from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator, interp1d
    mask = np.where(~np.isnan(Z))


    interp = interp1d(np.transpose(xcenters[mask[0]]), Z[mask], kind='nearest', fill_value=0, bounds_error=False)

    pred = interp(wt_test[:dt1])
    pred_h = np.zeros(pred.shape).reshape(-1, 1)
    pred = pred.reshape(-1, 1)
    X1 = np.concatenate([(np.concatenate(
        (pred, pred_h), axis=1))
        for rep in train_idx])

    X1 = np.nan_to_num(X1)

    y_train = np.concatenate([spks[r, :dt1, idx] for r in train_idx])
    y_test = np.concatenate([spks[r, :dt1, idx] for r in test_idx])
    fittedParameters, pcov, res = fitnonlin(X1, y_train, func)

    print('prediction on training set : ')
    res1 = res  # + (w_pc*pcs_test[:, 0])
    res2 = np.mean(res1.reshape(len(train_idx), dt1), axis=0)
    ev = explained_variance_score(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2,
                                  multioutput='uniform_average')
    feve = FEVE(y_train.reshape(len(train_idx), dt1), res2)
    print(feve)
    print(ev)
    print(np.corrcoef(np.mean(y_train.reshape(len(train_idx), dt1), axis=0), res2))

    plt.figure()
    plt.plot(res)
    plt.plot(np.mean(spk, axis=0))


def PlotTuningCurve(rfs, idx, visual_coverage, sigmas, screen_ratio, frequencies, show=True):
    xM, xm, yM, ym = visual_coverage
    
    # 1. Extract all 5 max indices clearly (to make the code easier to read)
    maxes = np.array(rfs[1])
    x = maxes[0, idx]
    y = maxes[1, idx]
    o = maxes[2, idx]
    s = maxes[3, idx]
    f = maxes[4, idx]  # NEW: The frequency index

    # 2. Slice the 6D array! We freeze the 6th dimension at 'f' for the spatial/ori plots
    cc_f_1_xy = rfs[0][idx, :, :, o, s, f]
    cc_f_1_o = rfs[0][idx, x, y, :, :, f]
    
    # 3. NEW: Extract the 1D frequency tuning curve (varying the 6th dimension)
    f_tuning = rfs[0][idx, x, y, o, s, :]

    # --- Everything below here is your original SVD logic ---
    u, s__, v = svds(cc_f_1_xy, 2)
    ori_tun = np.append(cc_f_1_o[:, s], cc_f_1_o[0, s])
    i = 1
    if v[1][np.argmax(abs(v[1]))] < 0:
        i = -1
        
    if show:
        # Changed 1, 5 to 1, 6 and made the figure slightly wider
        fig, ax = plt.subplots(1, 6, figsize=(18, 1.5))
        m = ax[0].imshow(cc_f_1_xy.T, cmap='coolwarm')
        fig.colorbar(m)
        ax[0].set_xticks([0, cc_f_1_xy.shape[0]], [xM, xm])
        ax[0].set_yticks([0, cc_f_1_xy.shape[1]], [yM, ym])
        ax[0].set_title('2D correlation')
        
        ax[1].plot(i * v[1][::-1], c='k')
        ax[1].set_xticks([0, cc_f_1_xy.shape[1]], [ym, yM])
        ax[1].set_title('Elevation (deg)')
        
        ax[2].plot(i * u[:, 1], c='k')
        ax[2].set_xticks([0, cc_f_1_xy.shape[0]], [xM, xm])
        ax[2].set_title('Azimuth (deg)')
        
        ax[3].plot(ori_tun, 'o-', c='k')
        ax[3].set_xticks([0, 4, 8], [0, 90, 180])
        ax[3].set_title('Orientation (deg)')
        
        ax[4].plot(cc_f_1_o[o, :], 'o-', c='k')
        ax[4].set_xticks(np.arange(len(sigmas)), sigmas)
        ax[4].set_title('Size (deg)')

        # NEW: Plot the frequency tuning curve
        ax[5].plot(f_tuning, 'o-', c='k')
        ax[5].set_xticks(np.arange(len(frequencies)), [round(freq, 2) for freq in frequencies])
        ax[5].set_title('Spatial Freq')

    # Return your original list, but append f_tuning at the end
    return [cc_f_1_xy.T, i * v[1][::-1], i * u[:, 1], ori_tun, cc_f_1_o[o, :], f_tuning]



from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels



def circular_variance(angles, responses):
    '''
    Compute the preferred orientation and circular variance of cells
    as per Ringach et al. 2002

    Args:
        angles (ndarray): Angles used in experiment, in degrees
        responses (ndarray): (n_angles, n_cells) responses of all cells to all angles

    Returns:
        preferred_angles: 0-180 degrees, favorite angle of each cell
        circular_variance: 0 is very selective, 1 is not selective at all

    '''
    # responses should be of shape n_angles, n_cells
    # angles is of shape n_angles IN DEGREES
    angles_radians = np.deg2rad(angles)[:, np.newaxis]

    numerator = (responses * np.exp(angles_radians * 2j)).sum(axis=0)
    denominator = responses.sum(axis=0)
    resultant = numerator / denominator

    circular_variance = 1 - np.abs(resultant)

    preferred_angles = np.rad2deg(np.angle(resultant))
    preferred_angles = preferred_angles / 2 + 90
    preferred_angles = np.mod(preferred_angles + 90, 180)

    return preferred_angles, circular_variance


def compute_signal_related_variance(resp_a, resp_b, mean_center=True):
    '''
    compute the fraction of signal-related variance for each neuron,
    as per Stringer et al Nature 2019. Cross-validated by splitting
    responses into two halves. Note, this only is "correct" if resp_a
    and resp_b are *not* averages of many trials.

    Args:
        resp_a (ndarray): n_stimuli, n_cells
        resp_b (ndarray): n_stimuli, n_cells

    Returns:
        fraction_of_stimulus_variance: 0-1, 0 is non-stimulus-caring, 1 is only-stimulus-caring neurons
        stim_to_noise_ratio: ratio of the stim-related variance to all other variance
    '''
    if len(resp_a.shape) > 2:
        # if the stimulus is multi-dimensional, flatten across all stimuli
        resp_a = resp_a.reshape(-1, resp_a.shape[-1])
        resp_b = resp_b.reshape(-1, resp_b.shape[-1])
    ns, nc = resp_a.shape
    if mean_center:
        # mean-center the activity of each cell
        resp_a = resp_a - resp_a.mean(axis=0)
        resp_b = resp_b - resp_b.mean(axis=0)

    # compute the cross-trial stimulus covariance of each cell
    # dot-product each cell's (n_stim, ) vector from one half
    # with its own (n_stim, ) vector on the other half

    covariance = (resp_a * resp_b).sum(axis=0) / ns

    # compute the variance of each cell across both halves
    resp_a_variance = (resp_a ** 2).sum(axis=0) / ns
    resp_b_variance = (resp_b ** 2).sum(axis=0) / ns
    total_variance = (resp_a_variance + resp_b_variance) / 2

    # compute the fraction of the total variance that is
    # captured in the covariance
    fraction_of_stimulus_variance = covariance / total_variance

    # if you want, you can compute SNR as well:
    stim_to_noise_ratio = fraction_of_stimulus_variance / (1 - fraction_of_stimulus_variance)

    return fraction_of_stimulus_variance, stim_to_noise_ratio

def split_trials(n_stim, n_rep, n_split = 2):
    # n_stim, n_rep = respmat.shape[:2]
    n_trial_split = n_rep // n_split
    trial_idxs = np.zeros((n_split, n_stim, n_rep), bool)

    for st_idx in range(n_stim):
        trial_order = np.random.permutation(n_rep)
        for split_idx in range(n_split):
            split_trials = trial_order[split_idx*n_trial_split : (split_idx+1) * n_trial_split]
            trial_idxs[split_idx, st_idx][split_trials] = 1

    return trial_idxs.astype(bool)

def stimresp_matrix(stimuli, responses, n_responses_per_stim = None):
        '''
        Make a stim-response matrix from a sequence of stimuli and responses

        Args:
            stimuli (ndarray): IDs of stimuli presented in sequence, of size n_trials
            responses (ndarray): Responses to each trial, of size n_trials, n_cells, n_response_window
            n_responses_per_stim (int, optional): Ignore stimuli with less repeats than this number

        Returns:
            respmat: n_unique_stim, n_repeats, n_cells, n_response_window
        '''
        unique_stim = np.unique(stimuli)
        if n_responses_per_stim is None:
            n_responses_per_stim = (stimuli  == unique_stim[0]).sum()
        n_unique_stim = len(unique_stim)
        n_cells = responses.shape[1]
        stim_ids = []
        respmat = []
        for idx, stim_id in enumerate(unique_stim):
            idxs = np.where(stimuli == stim_id)[0][:n_responses_per_stim]
            if len(idxs) < n_responses_per_stim:
                print("Stim %d only has %d repeats" % (stim_id, len(idxs)))
                continue
            respmat.append(responses[idxs])
            stim_ids.append(stim_id)
        return np.array(respmat), stim_ids


def lowess(x, y, f=1./3.):
    """
    Basic LOWESS smoother with uncertainty.
    Note:
        - Not robust (so no iteration) and
             only normally distributed errors.
        - No higher order polynomials d=1
            so linear smoother.
    """
    # get some paras
    xwidth = f*(x.max()-x.min()) # effective width after reduction factor
    N = len(x) # number of obs
    # Don't assume the data is sorted
    order = np.argsort(x)
    # storage
    y_sm = np.zeros_like(y)
    y_stderr = np.zeros_like(y)
    # define the weigthing function -- clipping too!
    tricube = lambda d : np.clip((1- np.abs(d)**3)**3, 0, 1)
    # run the regression for each observation i
    for i in range(N):
        dist = np.abs((x[order][i]-x[order]))/xwidth
        w = tricube(dist)
        # form linear system with the weights
        A = np.stack([w, x[order]*w]).T
        b = w * y[order]
        ATA = A.T.dot(A)
        ATb = A.T.dot(b)
        # solve the syste
        sol = np.linalg.solve(ATA, ATb)
        # predict for the observation only
        yest = A[i].dot(sol)# equiv of A.dot(yest) just for k
        place = order[i]
        y_sm[place]=yest
        sigma2 = (np.sum((A.dot(sol) -y [order])**2)/N )
        # Calculate the standard error
        y_stderr[place] = np.sqrt(sigma2 *
                                A[i].dot(np.linalg.inv(ATA)
                                                    ).dot(A[i]))
    return y_sm, y_stderr

def getHVA(signMap, neuron_pos, thresh=0.3, sign=1):
    if sign==1:
        signMap_binary=signMap>0
    elif sign==0:
        signMap_binary = signMap < 0
    kernel = np.ones((100, 100), np.uint8)
    opening = cv.morphologyEx(signMap_binary.astype('uint8'), cv.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
    ret, sure_fg = cv.threshold(dist_transform, thresh * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0

    sm=signMap*255*signMap_binary
    markers2 = cv.watershed(cv2.merge((sm.astype('uint8'),sm.astype('uint8'),sm.astype('uint8'))), markers)
    markers2_neurons=np.array([markers2[np.maximum(0, int(neuron_pos[i, 1])-1), np.maximum(0, int(neuron_pos[i, 0])-1)] for i in range(neuron_pos.shape[0])])#np.zeros_like(maxes[0, :])
    return markers2, markers2_neurons

def filter_nan_gaussian_conserving2(arr, sigma, mode='reflect'):
    """Apply a gaussian filter to an array with nans.

    Intensity is only shifted between not-nan pixels and is hence conserved.
    The intensity redistribution with respect to each single point
    is done by the weights of available pixels according
    to a gaussian distribution.
    All nans in arr, stay nans in gauss.
    """
    nan_msk = np.isnan(arr)

    loss = np.zeros(arr.shape)
    loss[nan_msk] = 1
    loss = ndimage.gaussian_filter(
            loss, sigma=sigma, mode=mode, cval=1)

    gauss = arr / (1-loss)
    gauss[nan_msk] = 0
    gauss = ndimage.gaussian_filter(
            gauss, sigma=sigma, mode=mode, cval=0)
    gauss[nan_msk] = np.nan

    return gauss

def visualSignMap(phasemap1, phasemap2):
    """
    calculate visual sign map from two orthogonally oriented phase maps
    """

    if phasemap1.shape != phasemap2.shape:
        raise LookupError("'phasemap1' and 'phasemap2' should have same size.")

    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    graddir1 = np.zeros(np.shape(gradmap1[0]))

    graddir2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i, j] = math.atan2(gradmap1[1][i, j], gradmap1[0][i, j])
            graddir2[i, j] = math.atan2(gradmap2[1][i, j], gradmap2[0][i, j])

    vdiff = np.multiply(np.exp(1j * graddir1), np.exp(-1j * graddir2))

    areamap = np.sin(np.angle(vdiff))

    return areamap

def getSignMap(neuron_pos, maxes, plotting=False):
    import scipy as sp
    x_pos=np.arange(0, np.max(neuron_pos[:, 0]))
    y_pos=np.arange(0, np.max(neuron_pos[:, 1]))
    grid=np.meshgrid(x_pos, y_pos)
    grid=np.array(grid).reshape(2, -1).T
    tree_A = cKDTree(neuron_pos)
    tree_B = cKDTree(grid)
    neighbourhood = tree_B.query_ball_tree(tree_A, 100)
    newx = np.ones_like(grid[:, 0]) * np.NaN
    newy = np.ones_like(grid[:, 1]) * np.NaN
    
    # Removed the wrapping 'for i in range(len(neighbourhood)):' here
    # Direct boolean evaluation is vastly faster than try/except
    for i, n in enumerate(neighbourhood):
        if n:  # If the neighborhood list is not empty
            newx[i] = np.nanmedian(maxes[0, n])
            newy[i] = np.nanmedian(maxes[1, n])
        else:
            print('no neighbour')

    newx2d = newx.reshape(y_pos.shape[0], x_pos.shape[0])
    newy2d = newy.reshape(y_pos.shape[0], x_pos.shape[0])

    newx2d_blur = filter_nan_gaussian_conserving2(newx2d, sigma=50)
    newy2d_blur = filter_nan_gaussian_conserving2(newy2d, sigma=50)

    signMap = visualSignMap(newx2d_blur, newy2d_blur)
    signMap_blur = filter_nan_gaussian_conserving2(signMap, sigma=15)
    sign_map_neurons=np.array([signMap_blur[np.maximum(0, int(neuron_pos[i, 1])-1), np.maximum(0, int(neuron_pos[i, 0])-1)] for i in range(neuron_pos.shape[0])])#np.zeros_like(maxes[0, :])

    if plotting:
        plt.figure()
        plt.imshow(signMap, cmap='coolwarm')

        plt.figure()
        plt.imshow(signMap_blur, vmin=-np.max(signMap_blur), vmax=np.max(signMap_blur), cmap='coolwarm')
        plt.colorbar()

        plt.figure()
        plt.imshow(newx2d, cmap='jet')

        plt.figure()
        plt.imshow(newy2d, cmap='jet')

    return signMap, sign_map_neurons

from skimage.color import lab2rgb

def TwoDimColorMap(X, Y, plotting=False):
    az = np.arange(0, 6)  # azimuths (assumes one screen and a half)
    el = np.arange(-1.5, 1.5)
    # I chose squares of size 8 just to illustrate the point:
    # to map neurons I would use 1 deg squares
    # to map neuropil I would use 10 or 15 deg squares

    azMean = np.mean(az)
    elMean = np.mean(el)
    azRange = np.ptp(az)
    elRange = np.ptp(el)

    azMat, elMat = np.meshgrid(az, el)

    # scale them to be L,a,b coordinates

    aMat = 2 * (azMat - azMean) / azRange * 100  # red-green
    bMat = 2 * (elMat - elMean) / elRange * 100  # blue-yellow

    Lmat = 65 * np.ones(np.shape(azMat))  # brightness (from 0 to 100)
    # where zero should be black but weirdly it is not
    # I would use transparency to encode strength of responses

    # convert them to RGB

    rgbImage = lab2rgb(np.dstack((Lmat, aMat, bMat)))
    rgbImage[rgbImage < 0] = 0
    rgbImage=skimage.transform.resize(rgbImage, (11, 27),mode='edge', order=0, anti_aliasing=True, preserve_range=True)

    if plotting:
        plt.figure()
        plt.imshow(rgbImage)
        plt.plot([0, 26], [5, 5], 'k--')
        plt.plot([0, 0], [0, 10], 'k--')
        plt.axis('image')
        plt.xlabel('Azimuth')
        plt.ylabel('Elevation')
        plt.title('Two-dimensional colormap of the screen')

    col=rgbImage[Y, X]
    return col, rgbImage


def rescale_to_minus_a_plus_a(arr, a=1.0):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr)  # éviter division par zéro
    arr_scaled = 2 * a * (arr - arr_min) / (arr_max - arr_min) - a
    return arr_scaled


def signaltonoiseScipy(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def _process_single_neuron(idx, maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1, n_min, double_wavelet_model, train_idx, test_idx, plotting, frames_per_minute):
    x=maxes0[0, idx]
    y=maxes0[1, idx]
    o=maxes0[2, idx]
    s=maxes0[3, idx]
    x1 = maxes1[0, idx]
    y1 = maxes1[1, idx]
    o1 = maxes1[2, idx]
    s1 = maxes1[3, idx]
    print(idx, (x, y, o, s), (x1, y1, o1, s1))

    w_i = wavelets_i[:, x, y, o, s ].reshape(-1, 1)  # +w_i_downsampled[:, x1, y1, o1, s1]
    w_r = wavelets_r[:, x, y, o, s].reshape(-1, 1)  # +w_r_downsampled[:, x1, y1, o1, s1]

    w_i_inhib = wavelets_i[:, x1, y1, o1, s1].reshape(-1, 1)
    w_r_inhib = wavelets_r[:, x1, y1, o1, s1].reshape(-1, 1)
    
    w_r = rescale_to_minus_a_plus_a(w_r, a=abs(w_i).max())
    w_r_inhib = rescale_to_minus_a_plus_a(w_r_inhib, a=abs(w_i_inhib).max())

    # Vectorized polar coordinate calculation with ravel() for zero unnecessary RAM
    rho = np.hypot(w_r.ravel(), w_i.ravel())
    phi = np.arctan2(w_i.ravel(), w_r.ravel())
    phi = np.unwrap(phi)
    dphi = np.diff(phi, prepend=0)
    dphi[abs(dphi) >= 3] = np.nan
    nans, x_val = nan_helper(dphi)
    dphi[nans] = np.interp(x_val(nans), x_val(~nans), dphi[~nans])

    rho_inhib = np.hypot(w_r_inhib.ravel(), w_i_inhib.ravel())
    phi_inhib = np.arctan2(w_i_inhib.ravel(), w_r_inhib.ravel())
    dphi_inhib = np.diff(phi_inhib, prepend=0)
    dphi_inhib[abs(dphi_inhib) >= 3] = np.nan
    nans, x_val = nan_helper(dphi_inhib)
    dphi_inhib[nans] = np.interp(x_val(nans), x_val(~nans), dphi_inhib[~nans])

    if not double_wavelet_model:
        w_i_inhib=np.zeros(w_i.shape)
        w_r_inhib = np.zeros(w_i.shape)
        dphi_inhib=np.zeros(dphi.shape)

    if plotting:
        spk = spks[:, :, idx]

        fig, ax = plt.subplots(6, 1)
        ax[0].plot(np.mean(spk, axis=0)[8500:9000])
        ax[1].plot(w_r[8500:9000])
        ax[2].plot(w_i[8500:9000])
        ax[3].plot(rho[8500:9000])
        ax[4].plot(phi[8500:9000])
        ax[5].plot(dphi[8500:9000])

        colors = []
        ax = plt.figure().add_subplot(projection='3d')
        for i in np.arange(1000, 3000, 1):
            c = np.mean(spk, axis=0)[i]
            color = plt.cm.coolwarm(255 * c / 100)
            colors.append(color)
            ax.scatter(rho[i - 1:i + 1], phi[i - 1:i + 1], dphi[i - 1:i + 1], s=10,
                       color=color)  # dphi[1000:1100])#, color=colors)

    vis_resp, a, nonlinparams, rhophiparams, plots, unrectified, w, interp = GetNeuronVisresponse(idx, w_i, w_r, w_i_inhib,
                                                                                          w_r_inhib,
                                                                                          dphi.reshape(-1, 1),
                                                                                          dphi_inhib.reshape(-1, 1),
                                                                                          spks, dt1=dt1,
                                                                                          n_min=n_min,
                                                                                          train_idx=train_idx,
                                                                                          test_idx=test_idx,
                                                                                          double_wavelet_model=False,
                                                                                          lastmin=True, func=relu, sigma=15,
                                                                                          plotting=False,
                                                                                          frames_per_minute=frames_per_minute)
    print(rhophiparams)
    return vis_resp, nonlinparams, rhophiparams, a, interp

def run_Model(maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1=9000,
              n_min=5, double_wavelet_model=True, train_idx=[0, 2],
              test_idx=[1, 3], plotting=False, frames_per_minute=None):
    """Fit the fast nonlinear Gabor-wavelet model for every neuron in parallel."""
    if frames_per_minute is None:
        frames_per_minute = DEFAULT_FRAMES_PER_MINUTE
    frames_per_minute = int(frames_per_minute)

    num_neurons = spks.shape[2]
    n_jobs = model_parallel_jobs()

    parallel_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_process_single_neuron)(
            idx, maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1, n_min,
            double_wavelet_model, train_idx, test_idx, plotting, frames_per_minute
        ) for idx in range(num_neurons)
    )

    # Memory Optimization: Pre-allocate standard lists
    Predictions = []
    nonlinParams = []
    RhoPhiParams = []
    Metrics = []
    interpolators = []

    # Append sequentially to avoid zip(*results) duplicating the full result set in RAM.
    for res in parallel_results:
        Predictions.append(res[0])
        nonlinParams.append(res[1])
        RhoPhiParams.append(res[2])
        Metrics.append([res[3][0], res[3][1], res[3][2][0][1]])
        interpolators.append(res[4])

    del parallel_results

    # Convert to NumPy arrays and return, strictly backwards-compatible
    Predictions = np.array(Predictions)
    nonlinParams = np.array(nonlinParams)
    RhoPhiParams = np.array(RhoPhiParams)
    Metrics = np.array(Metrics)
    
    return Predictions, nonlinParams, RhoPhiParams, Metrics, interpolators


def run_Full_Model( maxes1, maxes0, spks,idxs,thetas,sigmas, frequencies,visual_coverage, neuron_pos,
                    wavelet_path='/media/sophie/Expansion1/UCL/utils/2screens/10/',
                    savepath = '/home/sophie/Pictures/img zebra/supp/supp/', n_min=5, tt=[10, 18000],
                    memmapping=True, train_idx=[0, 2],test_idx=[1, 3],double_wavelet_model=False, lastmin=False,
                    plotting=False, frames_per_minute=None,
                    hz=DEFAULT_MOVIE_FRAME_RATE_HZ ):
    if frames_per_minute is None:
        frames_per_minute = int(hz) * SECONDS_PER_MINUTE
    frames_per_minute = int(frames_per_minute)
    hz = int(hz)
    Predictions = []
    Metrics = []
    Params = []
    nonlinParams = []
    RhoPhiParams = []
    OS = []
    xM, xm, yM, ym = visual_coverage
    interpolators = []

    if memmapping:
        print('looking for wavelets zarr folder in : ', wavelet_path)
        # ensure combined zarr exists for the two arrays (real/imag)
        # if .zarr exist use them, else try to combine .npy files to zarr on disk-efficient way
        try:
            wavelets_i = zarr.open(wavelet_path + 'dwt_videodata2_i.zarr', mode='r')
            wavelets_r = zarr.open(wavelet_path + 'dwt_videodata2_r.zarr', mode='r')
        except Exception:
            print('combining npy parts into zarr for imag using wavelets_io.py file ?')
        print(wavelets_i.shape)

        # do not load entire arrays to memory; access only slices when needed
    else:
        from .LoadPinkNoise import load_stimulus_simple_cell2

        wavelets = load_stimulus_simple_cell2(wavelet_path, tt=tt, downsampling=downsampling)
        wavelets_r = wavelets[0]
        wavelets_i = wavelets[1]
        del wavelets

    def findBestPos_profiled(x, y, o, s, nmin=5, plotting=False):
        x0 = np.maximum(x * 5, 5)
        y0 = np.maximum(y * 5, 5)
        x0 = np.minimum(x0, 130)
        y0 = np.minimum(y0, 49)

        spk = np.mean(spks[:5, :nmin * frames_per_minute, idx], axis=0).reshape(-1, 1)
        spk_train = np.mean(spks[[0, 2], :nmin * frames_per_minute, idx], axis=0).reshape(-1, 1)

        x = x0
        y = y0
        div = True
        ttt = 0
        r = 1
        w = 10

        xt = max(0, x0 - w)
        yt = max(0, y0 - w)

        x = w
        y = w

        wavelets_i_ = wavelets_i[tt[0]:nmin * frames_per_minute, xt:xt + (2 * w),
        yt:yt + (2 * w), :, :, :]
        wavelets_i_ = np.array(wavelets_i_)
        wavelets_r_ = wavelets_r[tt[0]:nmin * frames_per_minute, xt:xt + (2 * w),
        yt:yt + (2 * w), :, :, :]
        wavelets_r_ = np.array(wavelets_r_)

        # PRE-ALLOCATE directly to device to save PCIe bandwidth
        spk_train_tensor = torch.as_tensor(spk_train.T, device='cuda', dtype=torch.float32)

        while div:
            print(x, y)
            wavelets_complex = np.power(wavelets_r_[:, x, y], 2) + np.power(wavelets_i_[:, x, y], 2)
            
            # Use as_tensor to prevent intermediate CPU copies
            wavelets_tensor = torch.as_tensor(
                wavelets_complex.reshape(nmin * frames_per_minute, -1).T, 
                device='cuda', 
                dtype=torch.float32
            )
            
            concat_tensor = torch.cat((wavelets_tensor, spk_train_tensor), dim=0)
            
            # No need to detach if we aren't tracking gradients
            cc_f_1 = torch.corrcoef(concat_tensor).cpu().numpy()[-1:, :-1]
            cc_f_1 = cc_f_1.reshape(8, 5, 4)
            
            if plotting:
                fig, ax = plt.subplots(8)
                for i in range(8):
                    ax[i].imshow(cc_f_1[i].T, vmin=-np.max(abs(cc_f_1)), vmax=np.max(abs(cc_f_1)), cmap='coolwarm')
            
            m = np.where(abs(cc_f_1) == np.max(abs(cc_f_1)))
            o, s, f = m[0][0], m[1][0], m[2][0]

            wc = np.sqrt(np.power(wavelets_r_[:, :, :, o, s, f], 2) + np.power(
                wavelets_i_[:, :, :, o, s, f], 2))
            if r == 1:
                wi = wavelets_i_[:, :, :, o, s, f]
                wr = wavelets_r_[:, :, :, o, s, f]
                wsin = np.max(wi)
                wcos = np.max(wr)
                m = np.argmax([wcos, wsin, np.max(wc)])
                print(m, [wcos, wsin, np.max(wc)])
                if m == 0:
                    ww = wr
                elif m == 1:
                    ww = wi
                else:
                    ww = wc
                r = 0

            if ww.shape != (2 * w, 2 * w):
                print('padding')
                w_temp = np.zeros((ww.shape[0], 2 * w, 2 * w))
                w_temp[:, :ww.shape[1], :ww.shape[2]] = ww
                ww = w_temp

            ww_tensor = torch.as_tensor(
                ww.reshape(nmin * frames_per_minute, -1).T, 
                device='cuda', 
                dtype=torch.float32
            )
            
            concat_xy = torch.cat((ww_tensor, spk_train_tensor), dim=0)
            cc_f_1_xy = torch.corrcoef(concat_xy).cpu().numpy()[-1:, :-1]
            cc_f_1_xy = np.nan_to_num(cc_f_1_xy)
            cc_f_1_xy = cc_f_1_xy.reshape(2 * w, 2 * w)

            m = np.where(abs(cc_f_1_xy) == np.max(abs(cc_f_1_xy)))
            xt, yt = m[0][0], m[1][0]
            x1, y1 = np.maximum(0, m[0][0] - w + x0), np.maximum(0, m[1][0] - w + y0)
            
            if plotting:
                plt.figure()
                plt.imshow(cc_f_1_xy.T, vmax=np.max(abs(cc_f_1_xy)), cmap='coolwarm')

            print(xt, yt)
            print(o, s, f)
            print(x1, y1)
            ttt = ttt + 1
            if xt == x:
                if yt == y:
                    print('converged')
                    div = False
            elif ttt == 10:
                div = False
            x = xt
            y = yt

        print(x1, y1, o, s, f)
        # We allow PyTorch's caching allocator to manage RAM naturally here
        return (x1, y1, o, s, f)
    
    
    def findBestPos(x, y, o, s, nmin=5, plotting=False):
        x0 = np.maximum(x * 5, 5)
        y0 = np.maximum(y * 5, 5)
        x0 = np.minimum(x0, 130)
        y0 = np.minimum(y0, 49)
        spk_train = np.mean(spks[[0, 2], :nmin * frames_per_minute, idx], axis=0).reshape(-1, 1)
        x = x0
        y = y0
        div = True
        ttt = 0
        r = 1
        w = 10

        # PRE-ALLOCATE directly to device
        spk_train_tensor = torch.as_tensor(spk_train.T, device='cuda', dtype=torch.float32)

        while div:
            print(x, y)
            wavelets_complex = np.power(wavelets_r_[:, x, y], 2) + np.power(wavelets_i_[:, x, y], 2)
            
            wavelets_tensor = torch.as_tensor(
                wavelets_complex.reshape(nmin * frames_per_minute, -1).T, 
                device='cuda', 
                dtype=torch.float32
            )
            
            concat_tensor = torch.cat((wavelets_tensor, spk_train_tensor), dim=0)
            
            cc_f_1 = torch.corrcoef(concat_tensor).cpu().numpy()[-1:, :-1]
            cc_f_1 = cc_f_1.reshape(8, 5, 4)
            
            if plotting:
                fig, ax = plt.subplots(8)
                for i in range(8):
                    ax[i].imshow(cc_f_1[i].T, vmin=-np.max(abs(cc_f_1)), vmax=np.max(abs(cc_f_1)), cmap='coolwarm')
            
            m = np.where(abs(cc_f_1) == np.max(abs(cc_f_1)))
            o, s, f = m[0][0], m[1][0], m[2][0]

            wc = np.sqrt(np.power(wavelets_r[:nmin * frames_per_minute, np.maximum(0, x0 - w):np.maximum(0, x0 - w) + (2 * w),
            np.maximum(0, y0 - w):np.maximum(0, y0 - w) + (2 * w), o, s, f], 2) + np.power(
                wavelets_i[:nmin * frames_per_minute, np.maximum(0, x0 - w):np.maximum(0, x0 - w) + (2 * w),
                np.maximum(0, y0 - w):np.maximum(0, y0 - w) + (2 * w), o, s, f], 2))
            
            if r == 1:
                wi = wavelets_i[:nmin * frames_per_minute, np.maximum(0, x0 - w):np.maximum(0, x0 - w) + (2 * w),
                np.maximum(0, y0 - w):np.maximum(0, y0 - w) + (2 * w), o, s, f]
                wr = wavelets_r[:nmin * frames_per_minute, np.maximum(0, x0 - w):np.maximum(0, x0 - w) + (2 * w),
                np.maximum(0, y0 - w):np.maximum(0, y0 - w) + (2 * w), o, s, f]
                wsin = np.max(wi)
                wcos = np.max(wr)
                m = np.argmax([wcos, wsin, np.max(wc)])
                print(m, [wcos, wsin, np.max(wc)])
                if m == 0:
                    ww = wr
                elif m == 1:
                    ww = wi
                else:
                    ww = wc
                r = 0

            if ww.shape != (2 * w, 2 * w):
                print('padding')
                w_temp = np.zeros((ww.shape[0], 2 * w, 2 * w))
                w_temp[:, :ww.shape[1], :ww.shape[2]] = ww
                ww = w_temp

            ww_tensor = torch.as_tensor(
                ww.reshape(nmin * frames_per_minute, -1).T, 
                device='cuda', 
                dtype=torch.float32
            )
            concat_xy = torch.cat((ww_tensor, spk_train_tensor), dim=0)
            
            cc_f_1_xy = torch.corrcoef(concat_xy).cpu().numpy()[-1:, :-1]
            cc_f_1_xy = np.nan_to_num(cc_f_1_xy)
            cc_f_1_xy = cc_f_1_xy.reshape(2 * w, 2 * w)

            m = np.where(abs(cc_f_1_xy) == np.max(abs(cc_f_1_xy)))
            x1, y1 = np.maximum(0, m[0][0] - w + x0), np.maximum(0, m[1][0] - w + y0)
            
            if plotting:
                plt.figure()
                plt.imshow(cc_f_1_xy.T, vmax=np.max(abs(cc_f_1_xy)), cmap='coolwarm')
            
            print(x, y)
            print(o, s, f)
            print(x1, y1)
            ttt = ttt + 1
            if x1 == x:
                if y1 == y:
                    print('converged')
                    div = False
            elif ttt == 10:
                div = False
            x = x1
            y = y1

        print(x, y, o, s, f)
        return (x, y, o, s, f)
    
    
    if idxs==None:
        list_neurons= range(neuron_pos.shape[0])
    else:
        list_neurons=idxs
    for idx in list_neurons:  # np.asarray(neuron_pos[:, 1]>600).nonzero()[0]:[1024, 732, 1789, 3279, 614]:#
        torch.cuda.empty_cache()
        print(idx)
        x, y, o, s = maxes1[:, idx]

        print(x, y, o, s)
        x1, y1, o1, s1 = maxes0[:, idx]
        print(x1, y1, o1, s1)

        if memmapping:
            (x, y, o, s, f) = findBestPos_profiled(int(np.round(x)), int(np.round(y)), int(np.round(o)),
                                                   int(np.round(s)))

            (x1, y1, o1, s1, f1) = findBestPos_profiled(int(np.round(x1)), int(np.round(y1)), int(np.round(o1)),
                                                        int(np.round(s1)))
        else:
            (x, y, o, s, f) = findBestPos(int(np.round(x)), int(np.round(y)), int(np.round(o)),
                                          int(np.round(s)), nmin=n_min)

            (x1, y1, o1, s1, f1) = findBestPos(int(np.round(x1)), int(np.round(y1)), int(np.round(o1)),
                                               int(np.round(s1)), nmin=n_min)

        if memmapping:
            wavelets_i_ = wavelets_i[tt[0]:n_min * frames_per_minute, x, y, :, :, :]
            wavelets_r_ = wavelets_r[tt[0]:n_min * frames_per_minute, x, y, :, :, :]
            wc = np.sqrt(np.power(wavelets_r_, 2) + np.power(
                wavelets_i_, 2))
        else:
            wc = np.sqrt(np.power(wavelets_r[:n_min * frames_per_minute, x, y, :, :, :], 2) + np.power(
                wavelets_i[:n_min * frames_per_minute, x, y, :, :, :], 2))

        wc_tensor = torch.as_tensor(wc.reshape(n_min * frames_per_minute, -1).T, device='cuda', dtype=torch.float32)
        spks_tensor = torch.as_tensor(np.mean(spks[:, :n_min * frames_per_minute, idx], axis=0).reshape(1, -1), device='cuda', dtype=torch.float32)
        
        cc_f_1_o = torch.corrcoef(torch.cat((wc_tensor, spks_tensor), dim=0)).cpu().numpy()[-1:, :-1]
        cc_f_1_o = cc_f_1_o.reshape(8, 5, 4)
        pp = cc_f_1_o[:, s, f]
        ori_selectivity = signaltonoiseScipy(pp)  # abs(np.max(pp)) / abs(np.mean(pp))

        if not plotting:
            OS.append(ori_selectivity)
            Params.append([[x, y, o, s, f], [x1, y1, o1, s1, f1]])

        if plotting:
            plt.rcParams.update({
                "font.size": 8,
                "svg.fonttype": "none"  # Texte éditable dans Inkscape
            })

            if memmapping:
                wavelets_i_ = wavelets_i[tt[0]:n_min * frames_per_minute, x, y, :, :, :]
                wavelets_r_ = wavelets_r[tt[0]:n_min * frames_per_minute, x, y, :, :, :]
                wc = np.sqrt(np.power(wavelets_r_, 2) + np.power(wavelets_i_, 2))
            else:
                wc = np.sqrt(np.power(wavelets_r[:n_min * frames_per_minute, x, y, :, :, :], 2) + np.power(
                    wavelets_i[:n_min * frames_per_minute, x, y, :, :, :], 2))

            wc_tensor_plot = torch.as_tensor(wc.reshape(n_min * frames_per_minute, -1).T, device='cuda', dtype=torch.float32)
            
            cc_f_1_xy = torch.corrcoef(torch.cat((wc_tensor_plot, spks_tensor), dim=0)).cpu().numpy()[-1:, :-1]
            cc_f_1_xy = cc_f_1_xy.reshape(135, 54)

            elev_val = np.arange(cc_f_1_xy.shape[1])
            azi_val = np.arange(cc_f_1_xy.shape[0])
            azi_val = (abs(azi_val - cc_f_1_xy.shape[0]) * (abs(xM - xm) / cc_f_1_xy.shape[0])) + xm
            elev_val = (abs(elev_val - cc_f_1_xy.shape[1]) * (abs(yM - ym) / cc_f_1_xy.shape[1])) + ym

            fig = plt.figure(figsize=(12, 1.2))  # en pouces
            fig.suptitle(str(idx), fontsize=16, ha='left', va='top')
            gs = GridSpec(1, 9, figure=fig)  # grille 1x8
            ax = []
            markersize = 4
            # 5 premiers subplots avec sharey
            for i in range(5):
                axe = fig.add_subplot(gs[0, i], sharey=ax[0] if ax else None)
                if i > 0:
                    axe.tick_params(labelleft=False)  # cache seulement les valeurs
                ax.append(axe)

            # 3 derniers subplots sans sharey
            for i in range(5, 9):
                axe = fig.add_subplot(gs[0, i])
                ax.append(axe)
            i = 1
            ax[0].plot(azi_val[::1], i * cc_f_1_xy[:, y][::-1], c='k')
            ax[0].set_xticks([-45, 45, 135], [135, 45, -45])
            ax[0].spines["top"].set_visible(False)
            ax[0].spines["right"].set_visible(False)
            ax[0].set_ylim(bottom=-np.max(abs(cc_f_1_xy)), top=np.max(abs(cc_f_1_xy)))
            ax[0].set_title('Azimuth (deg)')

            ax[2].plot(np.append(cc_f_1_o[:, s, f], cc_f_1_o[0, s, f]), 'o-', c='k', markersize=markersize)
            ax[2].set_xticks([0, 4, 8], [0, 90, 180])
            ax[2].spines["top"].set_visible(False)
            ax[2].spines["right"].set_visible(False)
            ax[2].set_title('Orientation (deg)')

            ax[1].plot(elev_val, i * cc_f_1_xy[x, :], c='k')
            ax[1].set_xticks([-30, 0, 30])
            ax[1].spines["top"].set_visible(False)
            ax[1].spines["right"].set_visible(False)
            ax[1].set_title('Elevation (deg)')

            mm = max(cc_f_1_o.min(), cc_f_1_o.max(), key=abs)

            ax[3].plot(sigmas, cc_f_1_o[o, :, f], 'o-', c='k', markersize=markersize)
            ax[3].set_xticks([0, 5, 10, 15, 20])
            ax[3].spines["top"].set_visible(False)
            ax[3].spines["right"].set_visible(False)
            ax[3].set_title('Size (deg)')

            ax[4].plot(frequencies, cc_f_1_o[o, s, :], 'o-', c='k', markersize=markersize)
            ax[4].set_xticks(
                np.floor(frequencies * np.array([100, 100, 100, 100])).astype(int) / np.array([100, 100, 100, 100]))
            ax[4].tick_params(axis='x', labelrotation=45)
            ax[4].spines["top"].set_visible(False)
            ax[4].spines["right"].set_visible(False)
            ax[4].set_title('Frequency (cdp)')

        if memmapping:
            # Slicing the existing zarr references instead of re-opening them from disk
            w_i = np.array(wavelets_i[tt[0]:tt[1], x, y, o, s, f]).reshape(-1, 1)
            w_r = np.array(wavelets_r[tt[0]:tt[1], x, y, o, s, f]).reshape(-1, 1)

            w_i_inhib = np.array(wavelets_i[tt[0]:tt[1], x1, y1, o1, s1, f1]).reshape(-1, 1)
            w_r_inhib = np.array(wavelets_r[tt[0]:tt[1], x1, y1, o1, s1, f1]).reshape(-1, 1)
        else:
            w_i = wavelets_i[:, x, y, o, s, f].reshape(-1, 1)  # +w_i_downsampled[:, x1, y1, o1, s1]
            w_r = wavelets_r[:, x, y, o, s, f].reshape(-1, 1)  # +w_r_downsampled[:, x1, y1, o1, s1]

            w_i_inhib = wavelets_i[:, x1, y1, o1, s1, f1].reshape(-1, 1)
            w_r_inhib = wavelets_r[:, x1, y1, o1, s1, f1].reshape(-1, 1)
        
        w_r = rescale_to_minus_a_plus_a(w_r, a=abs(w_i).max())
        w_r_inhib = rescale_to_minus_a_plus_a(w_r_inhib, a=abs(w_i_inhib).max())

        # Vectorized polar coordinate calculation (100x faster, zero unnecessary RAM)
        rho = np.hypot(w_r.flatten(), w_i.flatten())
        phi = np.arctan2(w_i.flatten(), w_r.flatten())
        dphi = np.diff(np.unwrap(phi), prepend=0) * hz
        dphi = np.clip(dphi, -2 * np.pi, 2 * np.pi)
        nans, dx = nan_helper(dphi)
        dphi[nans] = np.interp(dx(nans), dx(~nans), dphi[~nans])

        # Vectorized polar coordinate calculation (100x faster, zero unnecessary RAM)
        rho_inhib = np.hypot(w_r.flatten(), w_i.flatten())
        phi_inhib = np.arctan2(w_i.flatten(), w_r.flatten())
        phi_inhib = np.unwrap(phi_inhib)
        dphi_inhib = np.diff(np.unwrap(phi_inhib), prepend=0) * hz
        dphi_inhib = np.clip(dphi_inhib, -2 * np.pi, 2 * np.pi)
        nans, dx = nan_helper(dphi_inhib)
        dphi_inhib[nans] = np.interp(dx(nans), dx(~nans), dphi_inhib[~nans])

        dphi_ortho = np.zeros(dphi_inhib.shape)
        vis_resp, a, nonlinparams, rhophiparams, plots, unrectified, w,interp = GetNeuronVisresponse(idx, w_i, w_r,
                                                                                              w_i_inhib,
                                                                                              w_r_inhib,
                                                                                              dphi.reshape(-1, 1),
                                                                                              dphi_inhib.reshape(-1,
                                                                                                                 1),
                                                                                              spks, dt1=n_min * frames_per_minute,
                                                                                              n_min=n_min,
                                                                                              train_idx=train_idx,
                                                                                              test_idx=test_idx,
                                                                                              double_wavelet_model=double_wavelet_model,
                                                                                              lastmin=lastmin,
                                                                                              func=relu, sigma=15,
                                                                                              plotting=False,
                                                                                              frames_per_minute=frames_per_minute)

        if plotting:
            ncut = 20

            a = abs(max(rho.min(), rho.max()))
            a_x = np.linspace(0, a, ncut)
            plot = plots[0]
            ax[5].plot(a_x, plot[0], c='k')
            ax[5].set_ylim(bottom=0, top=np.max(plot[0]))
            ax[5].spines["top"].set_visible(False)
            ax[5].spines["right"].set_visible(False)
            ax[5].set_xticks([0, a / 2, a])
            ax[5].set_title('Amplitude (a.u.)')

            b_x = np.linspace(0, 2 * np.pi, ncut + 1)
            ax[6].plot(b_x, plot[1], c='k')
            ax[6].set_ylim(bottom=0, top=np.max(plot[1]))
            ax[6].spines["top"].set_visible(False)
            ax[6].spines["right"].set_visible(False)
            ax[6].set_xticks([0, np.pi, 2 * np.pi])
            ax[6].set_xticklabels([0, r'$\pi$', r'$2\pi$'])
            ax[6].set_title('Phase (rad)')

            c_x = np.linspace(-1, 1, ncut)
            ax[7].plot(c_x, plot[2], c='k')
            ax[7].set_ylim(bottom=0, top=np.max(plot[2]))
            ax[7].spines["top"].set_visible(False)
            ax[7].spines["right"].set_visible(False)
            ax[7].set_xticks([-1, 0, 1])
            ax[7].set_title('Drift (a.u.)')

            fig.savefig(savepath + str(idx) + "new.svg", format="svg", bbox_inches="tight")

            pref_phase = b_x[np.argmax(plot[1])]
            pref_ori = thetas[o] * 180 / np.pi
            pref_size = sigmas[s]

            plt.title('Receptive field ' + str(idx))

        if plotting:
            star = 4500
            stop = star + 500
            spk = spks[:, :, idx]
            fig, ax = plt.subplots(7, 1)
            plt.rcParams.update({
                "font.size": 8,
                "svg.fonttype": "none"  # Texte éditable dans Inkscape
            })
            ax[2].plot(np.mean(spk, axis=0)[star:stop], c='k')
            ax[3].plot(rho[star:stop], c='k')
            ax3 = ax[3].twinx()
            ax3.plot(np.mean(spk, axis=0)[star:stop], c='k', linestyle='dotted')
            ax[4].plot(np.unwrap(phi[star:stop]), c='k')
            ax[4].yaxis.set_major_locator(ticker.MultipleLocator(base=2 * np.pi))
            ax[4].yaxis.set_major_formatter(FuncFormatter(pi_formatter))
            ax4 = ax[4].twinx()
            ax4.plot(np.mean(spk, axis=0)[star:stop], c='k', linestyle='dotted')
            dphi_smooth = rolling_avg(dphi, 10)[15:-15]
            ax[5].plot(dphi_smooth[star:stop])
            ax[5].yaxis.set_major_locator(ticker.MultipleLocator(base=2 * np.pi))
            ax[5].yaxis.set_major_formatter(FuncFormatter(pi_formatter))
            ax[6].plot(vis_resp[star:stop], c='r')
            ax6 = ax[6].twinx()
            ax6.plot(np.mean(spk, axis=0)[star:stop], c='k', linestyle='--')
        print(rhophiparams)
        Predictions.append(vis_resp)
        nonlinParams.append(nonlinparams)
        rhophiparams.append(ori_selectivity)
        RhoPhiParams.append(rhophiparams)

        Metrics.append(a)
        interpolators.append(interp)

    M = [[m[0], m[1], m[2][0][1], m[3], m[4]] for m in Metrics]

    folder_name = 'model_results'
    full_path = os.path.join(savepath, folder_name)
    os.makedirs(full_path, exist_ok=True)

    np.save(os.path.join(full_path , 'RPdp_predictions_8_noneigh_c_smoothpos_' + str(n_min) + '.npy'), Predictions)
    np.save(os.path.join(full_path , 'RPdp_nonlinparams_8_noneigh_c_smoothpos_' + str(n_min) + '.npy'), nonlinParams)
    np.save(os.path.join(full_path ,  'RPdp_rhophiparams_8_noneigh_c_smoothpos_' + str(n_min) + '.npy'), RhoPhiParams)
    np.save(os.path.join(full_path ,  'RPdp_metrics_8_noneigh_c_smoothpos_' + str(n_min) + '.npy'), M)
    np.save(os.path.join(full_path ,  'RPdp_os_8_noneigh_c_smoothpos_' + str(n_min) + '.npy'), OS)
    np.save(os.path.join(full_path , 'RPdp_params_8_noneigh_c_smoothpos_' + str(n_min) + '.npy'), np.array(Params))
    with open(os.path.join(full_path ,  "interpolators_" + str(n_min) + '.pkl'), "wb") as f:
        pickle.dump(interpolators, f)
    return Predictions, Params, nonlinParams, RhoPhiParams, Metrics, OS, interpolators


def plotNeuralRaster(spks):
    test = spks.reshape(-1, spks.shape[2])
    k = 3
    res = svds(test, k, which='LM')
    u, s, v = res

    n_sort_ind = np.argsort(v[0])

    sorted_neurons = test[:, n_sort_ind[np.arange(0,  spks.shape[2],  1)]]
    plt.figure()
    plt.imshow(sorted_neurons.T, vmin=0, vmax=0.8, cmap='Greys')