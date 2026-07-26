"""Receptive-field correlation and low-level signal utilities."""
from .common import *
from .rf_correlation import streaming_cross_correlation as _safe_chunked_cross_corr
from ..runtime.performance import resolve_compute_device
from ..runtime.task_control import check_cancelled

def wavelet_feature_dims(wavelets_r, sigmas=None, frequencies=None):
    """Infer spatial and feature dimensions from a full-model wavelet tensor."""
    if wavelets_r.ndim == 6:
        _, nx, ny, n_orientations, n_sigmas, n_frequencies = wavelets_r.shape
    elif wavelets_r.ndim == 5:
        _, nx, ny, n_orientations, n_sigmas = wavelets_r.shape
        n_frequencies = len(frequencies) if frequencies is not None else 1
    else:
        raise ValueError(
            f"Expected wavelet tensor with 5 or 6 dimensions, got shape {wavelets_r.shape}"
        )
    if sigmas is not None and len(sigmas) != n_sigmas:
        raise ValueError(
            f"Wavelet sigma axis ({n_sigmas}) does not match sigmas length ({len(sigmas)})"
        )
    if frequencies is not None and len(frequencies) != n_frequencies:
        raise ValueError(
            "Wavelet frequency axis "
            f"({n_frequencies}) does not match frequencies length ({len(frequencies)})"
        )
    return nx, ny, n_orientations, n_sigmas, n_frequencies


def coarse_indices_to_full(x, y, nx, ny, margin=5):
    """Map coarse RF indices to full-resolution wavelet coordinates."""
    coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
    scale_x, scale_y = coarse_to_full_scale(nx, ny)
    x0 = np.minimum(np.maximum(x * scale_x, margin), nx - margin)
    y0 = np.minimum(np.maximum(y * scale_y, margin), ny - margin)
    return int(np.round(x0)), int(np.round(y0)), coarse_nx, coarse_ny


def pi_formatter(x, pos):
    """Function for pi formatter.

    Args:
        x: Input value for this operation.
        pos: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for convolve Stim.

    Args:
        stim: Input value for this operation.
        time_trial1: Input value for this operation.
        tau: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for max by index.

    Args:
        idx: Input value for this operation.
        arr: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    sub_arr = arr[idx]
    flat_idx = np.argmax(sub_arr)
    unrav = np.unravel_index(flat_idx, sub_arr.shape)
    
    # Maintained for full backward-compatibility with external callers
    where_format = tuple(np.array([dim]) for dim in unrav)
    return where_format, sub_arr.flat[flat_idx]


def PearsonCorrelation(stim, resp, neuron_pos, nx, ny, plotting=True):
    """Correlate wavelet stimulus with responses and locate RF peaks per neuron."""
    stim_flat = np.abs(stim.reshape(stim.shape[0], -1))

    rfs = _safe_chunked_cross_corr(stim_flat, resp)

    # Apply the legacy self-correlation correction a row at a time so its
    # boolean mask cannot double peak memory for a large RF tensor.
    for row in rfs:
        row[row >= 0.99] -= 1.0
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
    """Function for orientation correction for stretches.

    Args:
        visual_coverage: Input value for this operation.
        nx: Input value for this operation.
        ny: Input value for this operation.
        omax: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    xM, xm, yM, ym = visual_coverage
    x_ratio = abs(xM - xm) / nx
    y_ratio = abs(yM - ym) / ny
    delta_y = y_ratio / x_ratio
    corrected_ori = np.arctan(delta_y * np.tan(omax * np.pi / 180)) * 180 / np.pi
    
    # Vectorized boolean masking directly modifies the array in-place
    corrected_ori[corrected_ori < 0] += 180
    return corrected_ori


def PearsonCorrelationPinkNoise(
    stim, resp, neuron_pos, nx, ny, ns, nf, visual_coverage, screen_ratio,
    sigmas, frequencies, n_orientations=8, fil=[0], absolute=False,
    plotting=False, n_time=None, rf_output_path=None, paired_frequencies=None,
    cancel_event=None,
):
    """Compute a chunked RF-correlation tensor and preferred feature indices.

    Args:
        stim: Disk-backed or dense wavelet power with shape ``(frames, x, y,
            orientations, sigmas[, frequencies])``.
        resp: Trial-averaged neural responses with shape ``(frames, neurons)``.
        neuron_pos: `(neurons, 2+)` positions retained for optional legacy plots.
        nx: Number of x features in ``stim``.
        ny: Number of y features in ``stim``.
        ns: Number of sigma features in ``stim``.
        nf: Number of frequency features; use one for coarse coupled wavelets.
        visual_coverage: Visual-degree bounds in left/right/top/bottom order.
        screen_ratio: Visual degrees per x analysis pixel.
        sigmas: Sigma values converted for visual-degree display.
        frequencies: Frequency values associated with the final feature axis.
        paired_frequencies: Optional one-frequency-per-sigma values for a
            coupled coarse bank. These preserve the compact 5-D cache while
            reporting the scientifically correct frequency for each preferred
            sigma feature.
        n_orientations: Number of orientation bins over 180 degrees.
        fil: Legacy filtering selector used only by optional plotting.
        absolute: Select preferred features by absolute correlation after output.
        plotting: Enable legacy Matplotlib diagnostic figures.
        n_time: Optional shared frame limit; avoids loading beyond aligned data.
        rf_output_path: Optional NPY or Zarr path for disk-backed correlation
            output. The selected format remains sliceable for plotting without
            allocating the complete neuron-by-feature tensor in RAM.

    Returns:
        tuple: RF tensor with shape ``(neurons, x, y, orientations, sigmas,
        frequencies)``, preferred integer indices, preferred values in visual
        units, and peak correlation magnitudes.
    """
    # Keep Zarr/memmap inputs structured.  Flattening a disk-backed wavelet
    # tensor forces a full in-memory allocation before correlation starts.
    rfs = _safe_chunked_cross_corr(
        stim,
        resp,
        n_time=n_time,
        output_path=rf_output_path,
        # A Zarr RF result must be created in its final feature layout.  Zarr
        # has no inexpensive reshape view, whereas NPY memmaps retain their
        # established flattened write layout and are reshaped below.
        structured_output=True,
        output_feature_shape=(nx, ny, n_orientations, ns, nf),
        cancel_event=cancel_event,
    )
    expected_shape = (rfs.shape[0], nx, ny, n_orientations, ns, nf)
    is_structured_output = tuple(rfs.shape) == tuple(expected_shape)

    # Process one neuron at a time. This preserves the disk-bound contract for
    # both NPY memmaps and Zarr arrays: NumPy ufuncs/iteration on a Zarr object
    # could otherwise materialize or modify a detached full result array.
    flat_max_idx = np.empty(rfs.shape[0], dtype=np.int64)
    maxes = np.empty(rfs.shape[0], dtype=np.float32)
    output_chunks = getattr(rfs, "chunks", None)
    neuron_block = int(output_chunks[0]) if output_chunks else 1
    for start in range(0, rfs.shape[0], max(1, neuron_block)):
        check_cancelled(cancel_event)
        stop = min(rfs.shape[0], start + max(1, neuron_block))
        rows = np.asarray(rfs[start:stop])
        rows[rows >= 0.99] -= 1.0
        np.nan_to_num(rows, copy=False)
        # Assign explicitly so a Zarr slice is durably updated; for NPY/RAM it
        # remains an inexpensive in-place-compatible write.
        rfs[start:stop] = rows
        for local_neuron, row in enumerate(rows):
            flat_row = row.reshape(-1)
            selection_values = np.abs(flat_row) if absolute else flat_row
            local_idx = int(np.argmax(selection_values))
            neuron_idx = start + local_neuron
            flat_max_idx[neuron_idx] = local_idx
            maxes[neuron_idx] = abs(flat_row[local_idx])

    if not is_structured_output:
        rfs = rfs.reshape(expected_shape)
    
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
    paired_frequencies = np.asarray(
        paired_frequencies if paired_frequencies is not None else [], dtype=float
    )
    if paired_frequencies.size:
        if paired_frequencies.shape != np.asarray(sigmas).shape:
            raise ValueError("Paired coarse frequencies must contain exactly one value per sigma.")
        fmax_corr = paired_frequencies[smax.astype(int)]
    else:
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

            plt.figure()
            plt.imshow(np.ones((68, 180)), cmap='Greys')
            plt.plot([0, 175], [32, 32], 'k--')
            plt.plot([175, 175], [2, 66], 'k--')
            plt.xticks([0, 135, 179], [135, 0, -45])
            plt.yticks([0, 34, 67], [34, 0, -34])
            plt.axis('image')
            plt.xlabel('Azimuth')
            plt.ylabel('Elevation')
            plt.title('Screen positions')
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
    if hasattr(rfs, "flush"):
        rfs.flush()
    return rfs, maxe, maxe_corr, list(maxes)


def realign_Stim_mc(stim, syncEcho_flip_times, stim_times):
    # Completely vectorized searchsorted execution replacing structural loop appending
    """Function for realign Stim mc.

    Args:
        stim: Input value for this operation.
        syncEcho_flip_times: Input value for this operation.
        stim_times: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    idx = np.searchsorted(stim_times, syncEcho_flip_times, side='right') - 1
    return stim[:, :, idx]


def _vectorized_pearson_r(x, y):
    """Computes column-wise Pearson correlation between matrices x and y of shape (T, N)"""
    x_norm = (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-9)
    y_norm = (y - np.mean(y, axis=0)) / (np.std(y, axis=0) + 1e-9)
    return np.mean(x_norm * y_norm, axis=0)


def repetability_trial(resps_all, neuron_pos):
    # Vectorized trial cross-comparisons bypassing cell loops
    """Function for repetability trial.

    Args:
        resps_all: Input value for this operation.
        neuron_pos: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    c01 = _vectorized_pearson_r(resps_all[0], resps_all[1])
    c02 = _vectorized_pearson_r(resps_all[0], resps_all[2])
    c12 = _vectorized_pearson_r(resps_all[1], resps_all[2])
    respcorrs3 = (c01 + c02 + c12) / 3.0

    plt.figure()
    plt.scatter(neuron_pos[:, 1], neuron_pos[:, 0], s=5, vmax=1, c=respcorrs3, cmap='Greys')
    plt.colorbar()

    return respcorrs3


def repetability_trial2(resps_all, neuron_pos):
    """Function for repetability trial2.

    Args:
        resps_all: Input value for this operation.
        neuron_pos: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for repetability trial3.

    Args:
        resps_all: Input value for this operation.
        neuron_pos: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for match cumulative cdf.

    Args:
        source: Input value for this operation.
        template: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)

    src_quantiles = np.cumsum(src_counts) / source.size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template.size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def plotcolorbar(vmin, vmax, cmap='coolwarm'):
    """Function for plotcolorbar.

    Args:
        vmin: Input value for this operation.
        vmax: Input value for this operation.
        cmap: Input value for this operation.
    """
    if vmax < 1:
        l = np.random.randint(vmin * 1000, vmax * 1000, 10000).reshape(100, 100) / 1000
    else:
        l = np.random.randint(vmin, vmax, 10000).reshape(100, 100)
    plt.figure()
    plt.imshow(l, vmin=vmin, vmax=vmax, cmap=cmap)
    plt.colorbar()


def plotVariability(resp):
    """Function for plotVariability.

    Args:
        resp: Input value for this operation.
    """
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
    """Function for cart2pol4d.

    Args:
        x: Input value for this operation.
        y: Input value for this operation.
        dp: Input value for this operation.
        dn: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    xx, yy, dd, nn = np.meshgrid(x, y, dp, dn, sparse=True)
    rho = np.hypot(xx, yy) + nn
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    
    # Broadcast back to dense right before returning to guarantee 100% backward compatibility
    dd_dense = np.broadcast_to(dd, rho.shape).copy()
    nn_dense = np.broadcast_to(nn, rho.shape).copy()
    return rho, phi, dd_dense, nn_dense


def cart2pol3d(x, y, dp):
    # Math runs sparse to save peak RAM
    """Function for cart2pol3d.

    Args:
        x: Input value for this operation.
        y: Input value for this operation.
        dp: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    xx, yy, dd = np.meshgrid(x, y, dp, sparse=True)
    rho = np.hypot(xx, yy)
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    
    # Broadcast back to dense right before returning to guarantee 100% backward compatibility
    dd_dense = np.broadcast_to(dd, rho.shape).copy()
    return rho, phi, dd_dense


def cart2pol(x, y):
    # Completely safe as-is; rho and phi naturally evaluate to dense arrays
    """Function for cart2pol.

    Args:
        x: Input value for this operation.
        y: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    xx, yy = np.meshgrid(x, y, sparse=True)
    rho = np.hypot(xx, yy)
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    return rho, phi


def cart2pol_noise(x, y, n):
    # Completely safe as-is; rho and phi naturally evaluate to dense arrays
    """Function for cart2pol noise.

    Args:
        x: Input value for this operation.
        y: Input value for this operation.
        n: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    xx, yy, nn = np.meshgrid(x, y, n, sparse=True)
    rho = np.hypot(xx, yy) + nn
    phi = np.mod(np.arctan2(yy, xx), 2 * np.pi)
    return rho, phi


def preferedDirection(videodata, spks, x, y, o, w_i, w_r, window_size=5):
    """Function for preferedDirection.

    Args:
        videodata: Input value for this operation.
        spks: Input value for this operation.
        x: Input value for this operation.
        y: Input value for this operation.
        o: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        window_size: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    rho, phi = cart2pol(w_r, w_i)
    plt.figure()
    plt.plot(phi * 180 / np.pi)
    angle = phi * 180 / np.pi
    pref_dir = np.mean((np.mean(spks[:, :, 9275], axis=0) > 0) * angle * (abs(angle) >= o - 10))
    return angle, pref_dir


def PCcorrelation(spks, neuron_pos):
    """Function for PCcorrelation.

    Args:
        spks: Input value for this operation.
        neuron_pos: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for NeuronCorrelation.

    Args:
        idx: Input value for this operation.
        spks: Input value for this operation.
        neuron_pos: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    device = resolve_compute_device(prefer_gpu=True)
    try:
        t_spks = torch.as_tensor(spks, device=device)
        cc_ = torch.corrcoef(t_spks.reshape(-1, t_spks.shape[2]).T)
    except:
        print('sparsenoise sp045 ?')
        t_spks = torch.as_tensor(spks, device=device)
        cc_ = torch.corrcoef(t_spks)
        
    cc_f_1 = cc_[idx].detach().cpu().numpy()
    neurocorr = np.asarray(cc_f_1 >= 0.08).nonzero()[0]
    
    # OPTIMIZATION: Explicitly clean up heavy local VRAM references
    res = cc_.detach().cpu().numpy()
    del t_spks, cc_
    
    return res, neurocorr


def predictSparseNoise(resp, stim, rfs, tp, save=False):
    """Function for predictSparseNoise.

    Args:
        resp: Input value for this operation.
        stim: Input value for this operation.
        rfs: Input value for this operation.
        tp: Input value for this operation.
        save: Input value for this operation.
    """
    img = resp[tp].reshape(1, -1) @ rfs.reshape(rfs.shape[0], -1)
    img = img.reshape(8, 20)

    fig, ax = plt.subplots(8, 1)
    ax[0].imshow(img)
    for i in range(1, 8):
        ax[i].imshow(stim[tp - i + 1])
    if save:
        tifffile.imwrite('tp3000reconstructed.tif', img.reshape(54, 135))


def predictPinkNoise(maxes, vis_n, spks, rfs, tp, L, videodata, dt=50, save=False,
                     nx=None, ny=None, n_orientations=None, n_frequencies=None,
                     nx_full=None, ny_full=None):
    """Function for predictPinkNoise.

    Args:
        maxes: Input value for this operation.
        vis_n: Input value for this operation.
        spks: Input value for this operation.
        rfs: Input value for this operation.
        tp: Input value for this operation.
        L: Input value for this operation.
        videodata: Input value for this operation.
        dt: Input value for this operation.
        save: Input value for this operation.
        nx: Input value for this operation.
        ny: Input value for this operation.
        n_orientations: Input value for this operation.
        n_frequencies: Input value for this operation.
        nx_full: Input value for this operation.
        ny_full: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if nx is None or ny is None:
        nx, ny = coarse_grid_dimensions(rfs[0].shape[1], rfs[0].shape[2])
    if n_orientations is None:
        n_orientations = rfs[0].shape[3]
    if n_frequencies is None:
        n_frequencies = rfs[0].shape[5] if rfs[0].ndim > 5 else 1
    if nx_full is None:
        nx_full = L.shape[0]
    if ny_full is None:
        ny_full = L.shape[1]

    grid_shape = (spks.shape[-1], nx, ny, n_orientations, n_frequencies)
    rfs_model = np.zeros(grid_shape)

    # Replaced performance-heavy loop with native advanced array indexing
    rfs_model[np.arange(spks.shape[-1]), maxes[0].astype(int), maxes[1].astype(int), maxes[2].astype(int), maxes[3].astype(int)] = 1

    try:
        rfs_model = rfs[0] * rfs_model.reshape(grid_shape)
    except Exception:
        rfs_model = np.swapaxes(rfs[0], 1, 2) * rfs_model.reshape(grid_shape)

    r = np.sum(np.mean(spks[:, :, :], axis=0)[tp - dt:tp], axis=0).reshape(1, -1)
    vis = r @ rfs_model.reshape(rfs_model.shape[0], -1)
    vis = vis.reshape(nx, ny, n_orientations, n_frequencies)

    vis_t = skimage.transform.resize(
        vis,
        (nx_full, ny_full, n_orientations, n_frequencies),
        order=5,
        anti_aliasing=True,
    )

    fig, ax = plt.subplots(n_orientations, n_frequencies)
    if n_orientations == 1 and n_frequencies == 1:
        ax = np.array([[ax]])
    elif n_orientations == 1 or n_frequencies == 1:
        ax = ax.reshape(n_orientations, n_frequencies)
    for i in range(n_orientations):
        for j in range(n_frequencies):
            ax[i, j].imshow(vis_t[:, :, i, j].T, cmap='coolwarm')

    pixel_count = nx_full * ny_full
    mid_sigma = min(2, L.shape[3] - 1)
    freq_idx = min(2, n_frequencies - 1)
    vs = vis_t[:, :, :, freq_idx].reshape(1, -1) @ L[:, :, :, mid_sigma, 0].reshape(-1, pixel_count)
    print(np.corrcoef(vs, np.mean(videodata[tp - 2 * dt:tp - dt], axis=0).flatten()))

    if save:
        tifffile.imwrite('tp3520videodata.tif',
                         np.mean(videodata[tp - dt:tp], axis=0))

    vc = vis_t.reshape(1, -1) @ L[:, :, :, :, 0].reshape(-1, pixel_count)
    if save:
        tifffile.imwrite('tp3520reconstructed_1wavelet.tif',
                         (vs + vc).reshape(ny_full, nx_full))

    return (vs + vc).reshape(ny_full, nx_full)


def compute_skewness_neurons(spks, plotting=False):
    """Function for compute skewness neurons.

    Args:
        spks: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    s = np.mean(spks, axis=0)
    # Skewness applied directly to the 1D array
    skewness = skew(s)

    if plotting:
        plt.figure()
        plt.hist(skewness, bins=100)
    return list(skewness)


def lowpassfilter(sig, N=1, Wn=1.5, fs=100):
    """Function for lowpassfilter.

    Args:
        sig: Input value for this operation.
        N: Input value for this operation.
        Wn: Input value for this operation.
        fs: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    sos = signal.butter(N, Wn, 'lowpass', fs=fs, output='sos')
    return signal.sosfilt(sos, sig)


def DirectionSelectivity(x, y, o, s, w_i_downsampled, w_r_downsampled):
    """Function for DirectionSelectivity.

    Args:
        x: Input value for this operation.
        y: Input value for this operation.
        o: Input value for this operation.
        s: Input value for this operation.
        w_i_downsampled: Input value for this operation.
        w_r_downsampled: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for DirectionSelectivityPlot.

    Args:
        idx: Input value for this operation.
        spks: Input value for this operation.
        x: Input value for this operation.
        y: Input value for this operation.
        o: Input value for this operation.
        s: Input value for this operation.
        w_i_downsampled: Input value for this operation.
        w_r_downsampled: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for SinCosPlot.

    Args:
        idx: Input value for this operation.
        spks: Input value for this operation.
        x: Input value for this operation.
        y: Input value for this operation.
        o: Input value for this operation.
        s: Input value for this operation.
        w_i_downsampled: Input value for this operation.
        w_r_downsampled: Input value for this operation.
        ncut: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for hanningconv.

    Args:
        interp_grid: Input value for this operation.
        n: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    kern = np.hanning(n).reshape(-1, 1)
    kern = kern * kern.T
    kern /= kern.sum()
    # fftconvolve is mathematically identical but exponentially faster for larger arrays
    hanning = signal.fftconvolve(interp_grid, kern, mode='same')
    return hanning

def hanningconv3d(interp_grid, n):
    """Function for hanningconv3d.

    Args:
        interp_grid: Input value for this operation.
        n: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    kern = np.hanning(n).reshape(-1, 1)
    kern = kern * kern.T
    kern = kern[:, :, np.newaxis] * kern.T
    kern /= kern.sum()
    # fftconvolve handles n-dimensional arrays naturally and much faster
    hanning = signal.fftconvolve(interp_grid, kern, mode='same')
    return hanning


def interpolateData3d(z, dx, dy, dp, ncut, smooth=True):
    """Function for interpolateData3d.

    Args:
        z: Input value for this operation.
        dx: Input value for this operation.
        dy: Input value for this operation.
        dp: Input value for this operation.
        ncut: Input value for this operation.
        smooth: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for interpolateData.

    Args:
        z: Input value for this operation.
        dx: Input value for this operation.
        dy: Input value for this operation.
        ncut: Input value for this operation.
        smooth: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for SinCosPlot2.

    Args:
        idx: Input value for this operation.
        spk: Input value for this operation.
        w_i: Input value for this operation.
        w_r: Input value for this operation.
        dphi: Input value for this operation.
        noise: Input value for this operation.
        ncut: Input value for this operation.
        smoothing_size: Input value for this operation.
        plotting: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for moving average.

    Args:
        a: Input value for this operation.
        n: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret / n


def UnexpectedFiring(pred, Y, videodata):
    """Function for UnexpectedFiring.

    Args:
        pred: Input value for this operation.
        Y: Input value for this operation.
        videodata: Input value for this operation.
    """
    unfire = np.logical_and(np.logical_not(pred >= 0.20), Y >= 0.5)
    plt.figure()
    plt.imshow(np.mean(videodata[np.asarray(unfire).nonzero()[0]], axis=0))
    plt.figure()
    plt.rcParams['axes.facecolor'] = 'none'
    plt.plot(Y, c='k')
    plt.plot()


def diffSinCosPlot(idx, spks, x, y, o, s, w_i_downsampled, w_r_downsampled):
    """Function for diffSinCosPlot.

    Args:
        idx: Input value for this operation.
        spks: Input value for this operation.
        x: Input value for this operation.
        y: Input value for this operation.
        o: Input value for this operation.
        s: Input value for this operation.
        w_i_downsampled: Input value for this operation.
        w_r_downsampled: Input value for this operation.
    """
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


