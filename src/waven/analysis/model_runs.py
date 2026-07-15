"""Top-level model execution routines.

These functions orchestrate per-neuron model fitting and full-resolution model
runs. Lower-level math helpers live in :mod:`waven.analysis.nonlinear_models`.
"""
from .common import *
from .receptive_fields import *
from .nonlinear_models import *
from .trial_stats import *
from .rf_correlation import streaming_cross_correlation
from ..runtime.performance import cpu_inner_thread_count
from joblib import parallel_config

SECONDS_PER_MINUTE = 60


class _WaveletWindow:
    """Lazy time/spatial window over a disk-backed full-model wavelet array."""

    def __init__(self, source, time_start, time_end, x_start, x_end, y_start, y_end):
        self.source = source
        self.offsets = (int(time_start), int(x_start), int(y_start))
        self.shape = (
            int(time_end - time_start), int(x_end - x_start), int(y_end - y_start),
            *tuple(int(dim) for dim in source.shape[3:]),
        )
        source_chunks = getattr(source, "chunks", None)
        if source_chunks:
            self.chunks = tuple(
                min(int(size), int(chunk)) for size, chunk in zip(self.shape, source_chunks)
            )

    def __getitem__(self, item):
        if not isinstance(item, tuple) or len(item) < 3:
            raise IndexError("Wavelet window requires time, x, and y indices.")

        mapped = []
        for dimension, index in enumerate(item[:3]):
            offset = self.offsets[dimension]
            if isinstance(index, slice):
                start = 0 if index.start is None else int(index.start)
                stop = self.shape[dimension] if index.stop is None else int(index.stop)
                mapped.append(slice(offset + start, offset + stop, index.step))
            else:
                mapped.append(offset + int(index))
        return self.source[tuple(mapped) + item[3:]]

def signaltonoiseScipy(a, axis=0, ddof=0):
    """Function for signaltonoiseScipy.

    Args:
        a: Input value for this operation.
        axis: Input value for this operation.
        ddof: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def _sem_over_trials(trials):
    """Function for sem over trials.

    Args:
        trials: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    trials = np.asarray(trials, dtype=float)
    if trials.ndim < 2 or trials.shape[0] <= 1:
        return None
    return np.nanstd(trials, axis=0, ddof=1) / np.sqrt(trials.shape[0])


def _plot_trace_with_optional_sem(ax, trials, x=None, color='k', label=None, show_sem_errorbars=False):
    """Function for plot trace with optional sem.

    Args:
        ax: Input value for this operation.
        trials: Input value for this operation.
        x: Input value for this operation.
        color: Input value for this operation.
        label: Input value for this operation.
        show_sem_errorbars: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    trials = np.asarray(trials, dtype=float)
    if trials.ndim == 1:
        mean = trials
        n_trials = 1
    else:
        mean = np.nanmean(trials, axis=0)
        n_trials = trials.shape[0]
    if x is None:
        x = np.arange(mean.shape[0])
    ax.plot(x, mean, c=color, label=label)
    sem = _sem_over_trials(trials)
    if show_sem_errorbars and sem is not None:
        every = max(1, mean.shape[0] // 100)
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            fmt='none',
            ecolor=color,
            elinewidth=0.6,
            capsize=1,
            alpha=0.55,
            errorevery=every,
        )
        return n_trials
    return None


def _set_sem_caption(fig, n_trials):
    """Function for set sem caption.

    Args:
        fig: Input value for this operation.
        n_trials: Input value for this operation.
    """
    if n_trials and n_trials > 1:
        fig._waven_caption = f"Error bars represent SEM over {n_trials} trials."


def _process_single_neuron(idx, maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1, n_min, double_wavelet_model, train_idx, test_idx, plotting, frames_per_minute, lastmin=False, show_sem_errorbars=False):
    """Function for process single neuron.

    Args:
        idx: Input value for this operation.
        maxes0: Input value for this operation.
        maxes1: Input value for this operation.
        spks: Input value for this operation.
        wavelets_i: Input value for this operation.
        wavelets_r: Input value for this operation.
        dt1: Input value for this operation.
        n_min: Input value for this operation.
        double_wavelet_model: Input value for this operation.
        train_idx: Input value for this operation.
        test_idx: Input value for this operation.
        plotting: Input value for this operation.
        frames_per_minute: Input value for this operation.
        lastmin: Input value for this operation.
        show_sem_errorbars: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    x = int(np.round(maxes0[0, idx]))
    y = int(np.round(maxes0[1, idx]))
    o = int(np.round(maxes0[2, idx]))
    s = int(np.round(maxes0[3, idx]))
    x1 = int(np.round(maxes1[0, idx]))
    y1 = int(np.round(maxes1[1, idx]))
    o1 = int(np.round(maxes1[2, idx]))
    s1 = int(np.round(maxes1[3, idx]))

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
    # The first sample has no preceding frame, so its instantaneous phase drift
    # is zero. Prepending literal zero instead spuriously injected ``phi[0]``.
    dphi = np.diff(phi, prepend=phi[0])
    dphi[abs(dphi) >= 3] = np.nan
    nans, x_val = nan_helper(dphi)
    dphi[nans] = np.interp(x_val(nans), x_val(~nans), dphi[~nans])

    rho_inhib = np.hypot(w_r_inhib.ravel(), w_i_inhib.ravel())
    phi_inhib = np.unwrap(np.arctan2(w_i_inhib.ravel(), w_r_inhib.ravel()))
    dphi_inhib = np.diff(phi_inhib, prepend=phi_inhib[0])
    dphi_inhib[abs(dphi_inhib) >= 3] = np.nan
    nans, x_val = nan_helper(dphi_inhib)
    dphi_inhib[nans] = np.interp(x_val(nans), x_val(~nans), dphi_inhib[~nans])

    if not double_wavelet_model:
        w_i_inhib=np.zeros(w_i.shape)
        w_r_inhib = np.zeros(w_i.shape)
        dphi_inhib=np.zeros(dphi.shape)

    if plotting:
        spk = spks[:, :, idx]

        fig, ax = plt.subplots(6, 1, sharex=True)
        x_trace = np.arange(8500, 9000)
        sem_trials = _plot_trace_with_optional_sem(
            ax[0],
            spk[:, 8500:9000],
            x=x_trace,
            color='k',
            label='Trial-averaged spike activity',
            show_sem_errorbars=show_sem_errorbars,
        )
        _set_sem_caption(fig, sem_trials)
        ax[1].plot(x_trace, w_r[8500:9000], c='r')
        ax[2].plot(x_trace, w_i[8500:9000], c='b')
        ax[3].plot(x_trace, rho[8500:9000], c='k')
        ax[4].plot(x_trace, phi[8500:9000], c='k')
        ax[5].plot(x_trace, dphi[8500:9000], c='k')
        labels = [
            ("Spike activity", "Activity (a.u.)"),
            ("Cosine wavelet", "Wavelet value (a.u.)"),
            ("Sine wavelet", "Wavelet value (a.u.)"),
            ("Amplitude", "rho (a.u.)"),
            ("Phase", "phi (rad)"),
            ("Drift", "dphi (rad/frame)"),
        ]
        for axis, (title, ylabel) in zip(ax, labels):
            axis.set_title(title)
            axis.set_ylabel(ylabel)
        ax[-1].set_xlabel("Frame index")

        colors = []
        fig3d = plt.figure()
        ax = fig3d.add_subplot(projection='3d')
        for i in np.arange(1000, 3000, 1):
            c = np.mean(spk, axis=0)[i]
            color = plt.cm.coolwarm(255 * c / 100)
            colors.append(color)
            ax.scatter(rho[i - 1:i + 1], phi[i - 1:i + 1], dphi[i - 1:i + 1], s=10,
                       color=color)  # dphi[1000:1100])#, color=colors)
        ax.set_title("Wavelet trajectory colored by spike activity")
        ax.set_xlabel("Amplitude rho (a.u.)")
        ax.set_ylabel("Phase phi (rad)")
        ax.set_zlabel("Drift dphi (rad/frame)")

    vis_resp, a, nonlinparams, rhophiparams, plots, unrectified, w, interp = GetNeuronVisresponse(idx, w_i, w_r, w_i_inhib,
                                                                                          w_r_inhib,
                                                                                          dphi.reshape(-1, 1),
                                                                                          dphi_inhib.reshape(-1, 1),
                                                                                          spks, dt1=dt1,
                                                                                          n_min=n_min,
                                                                                          train_idx=train_idx,
                                                                                          test_idx=test_idx,
                                                                                          double_wavelet_model=double_wavelet_model,
                                                                                          lastmin=lastmin, func=relu, sigma=15,
                                                                                          plotting=False,
                                                                                          frames_per_minute=frames_per_minute)
    return vis_resp, nonlinparams, rhophiparams, a, interp

def run_Model(maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1=9000,
              n_min=5, double_wavelet_model=True, train_idx=[0, 2],
              test_idx=[1, 3], lastmin=False, plotting=False, frames_per_minute=None,
              show_sem_errorbars=False):
    """Fit the coarse nonlinear Gabor-wavelet model for one or more neurons.

    Args:
        maxes0: Smoothed preferred RF indices with shape ``(at least 4,
            n_neurons)`` in x/y/orientation/size order.
        maxes1: Unsmoothened preferred RF indices with the same shape; they
            seed the inhibitory-feature selection.
        spks: Aligned neural responses with shape ``(trials, frames, neurons)``.
        wavelets_i: Imaginary coarse phase with shape ``(frames, x, y,
            orientations, sigmas)``. May be a disk-backed Zarr/NPY view.
        wavelets_r: Real coarse phase with the same shape as ``wavelets_i``.
        dt1: Maximum shared training-frame count.
        n_min: Model fitting duration in minutes.
        double_wavelet_model: Whether to include the inhibitory wavelet feature.
        train_idx: Zero-based trial indices used to fit the nonlinearity.
        test_idx: Zero-based held-out trial indices.
        lastmin: Whether to evaluate an additional final-minute holdout.
        plotting: Enable legacy diagnostic figures. GUI workers leave this false.
        frames_per_minute: Required movie FPS multiplied by 60.
        show_sem_errorbars: Compatibility argument for legacy plot callers.

    Returns:
        tuple: Predictions, nonlinear parameters, rho/phi parameters, scalar
        metrics, and interpolation objects, each ordered by neuron.

    Raises:
        ValueError: If cache axes, RF indices, trial splits, or shared frame length are incompatible.
    """
    if wavelets_i.ndim != 5 or wavelets_r.ndim != 5:
        raise ValueError(
            "run_Model expects 5D coarse wavelets with shape "
            "(time, coarse_nx, coarse_ny, n_thetas, n_sigmas). "
            f"Got {wavelets_i.shape} and {wavelets_r.shape}."
        )
    if wavelets_i.shape != wavelets_r.shape:
        raise ValueError(
            f"Real/imag coarse wavelets must share a shape, got "
            f"{wavelets_i.shape} and {wavelets_r.shape}"
        )
    if frames_per_minute is None:
        raise ValueError(
            "run_Model requires frames_per_minute from the configured stimulus "
            "frame rate. Pass int(Hz) * 60 from the GUI/config."
        )
    frames_per_minute = int(frames_per_minute)

    if spks.ndim != 3:
        raise ValueError(
            "run_Model expects spikes with shape (trials, time, neurons); "
            f"got {spks.shape}."
        )
    num_neurons = int(spks.shape[2])
    if num_neurons <= 0:
        raise ValueError("run_Model received no neurons.")
    train_idx = [int(index) for index in train_idx]
    test_idx = [int(index) for index in test_idx]
    if not train_idx or not test_idx:
        raise ValueError("run_Model requires at least one training and one holdout trial.")
    invalid_trials = [
        index for index in train_idx + test_idx if index < 0 or index >= int(spks.shape[0])
    ]
    if invalid_trials:
        raise ValueError(
            f"run_Model trial index/indices are outside 0-{spks.shape[0] - 1}: {invalid_trials}."
        )
    overlap = sorted(set(train_idx) & set(test_idx))
    if overlap:
        raise ValueError(f"run_Model train and holdout trial indices overlap: {overlap}")
    for label, params in (("smoothed", maxes0), ("raw", maxes1)):
        params = np.asarray(params)
        if params.ndim != 2 or params.shape[0] < 4 or params.shape[1] != num_neurons:
            raise ValueError(
                f"run_Model {label} RF parameters must have shape (at least 4, {num_neurons}); "
                f"got {params.shape}."
            )
        limits = np.asarray(wavelets_i.shape[1:5], dtype=int)
        invalid = np.logical_or(~np.isfinite(params[:4]), params[:4] < 0)
        invalid |= params[:4] > (limits[:, None] - 1)
        if np.any(invalid):
            raise ValueError(
                f"run_Model {label} RF parameters are outside wavelet bounds {tuple(limits)}. "
                "Re-run Coarse RF analysis for the current wavelet cache."
            )
    shared_frames = min(int(spks.shape[1]), int(wavelets_i.shape[0]), int(dt1))
    if shared_frames < 2:
        raise ValueError("run_Model needs at least two frames shared by spikes and wavelets.")
    if shared_frames != int(dt1):
        dt1 = shared_frames
    # A selected GUI neuron is a one-neuron job.  Keep it serial so disk-backed
    # Zarr inputs are not pickled into a worker process. Plotting is serial too.
    if plotting or num_neurons == 1:
        parallel_results = [
            _process_single_neuron(
                idx, maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1, n_min,
                double_wavelet_model, train_idx, test_idx, plotting, frames_per_minute,
                lastmin=lastmin, show_sem_errorbars=show_sem_errorbars
            )
            for idx in range(num_neurons)
        ]
    else:
        n_jobs = model_parallel_jobs()
        # Each process may enter NumPy/SciPy/BLAS.  Limit its inner native
        # workers so ``n_jobs`` processes do not each claim every CPU core.
        with parallel_config(backend="loky", inner_max_num_threads=cpu_inner_thread_count(n_jobs)):
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(_process_single_neuron)(
                    idx, maxes0, maxes1, spks, wavelets_i, wavelets_r, dt1, n_min,
                    double_wavelet_model, train_idx, test_idx, plotting, frames_per_minute,
                    lastmin=lastmin, show_sem_errorbars=show_sem_errorbars
                ) for idx in range(num_neurons)
            )

    Predictions, nonlinParams, RhoPhiParams, Metrics, interpolators = [], [], [], [], []

    for res in parallel_results:
        Predictions.append(res[0])
        nonlinParams.append(res[1])
        RhoPhiParams.append(res[2])
        Metrics.append([res[3][0], res[3][1], res[3][2][0][1]])
        interpolators.append(res[4])

    del parallel_results

    return np.array(Predictions), np.array(nonlinParams), np.array(RhoPhiParams), np.array(Metrics), interpolators


def run_Full_Model(maxes0, maxes1, spks, idxs, thetas, sigmas, frequencies, visual_coverage, neuron_pos,
                   wavelet_path='.',
                   savepath='outputs', n_min=5, tt=None,
                   memmapping=True, train_idx=None, test_idx=None, double_wavelet_model=False, lastmin=False,
                   plotting=False, frames_per_minute=None, coarse_shape=None,
                   hz=None, show_sem_errorbars=False):
    """Refine coarse RF seeds against full-model real/imaginary wavelets.

    Args:
        maxes0: Smoothed coarse preferred indices with shape ``(at least 4,
            n_neurons)``.
        maxes1: Raw coarse preferred indices with the same neuron axis.
        spks: Aligned responses with shape ``(trials, frames, neurons)``.
        idxs: Zero-based neuron indices to refine.
        thetas: Orientation-bin values or indices used by legacy outputs.
        sigmas: Full-model sigma values in analysis pixels.
        frequencies: Full-model spatial frequencies in cycles per analysis pixel.
        visual_coverage: Four visual-degree bounds in left/right/top/bottom order.
        neuron_pos: `(n_neurons, 2+)` positions used by neighbourhood smoothing.
        wavelet_path: Folder containing `dwt_videodata2_r/i.zarr` or compatible
            NPY phases.
        savepath: Folder for durable model-result arrays.
        n_min: Training-window duration in minutes.
        tt: Required ``[start, stop]`` frame range shared by spikes and phases.
        memmapping: Prefer disk-backed phase loading.
        train_idx: Zero-based fitting trial indices.
        test_idx: Zero-based held-out trial indices.
        double_wavelet_model: Include inhibitory wavelet features when true.
        lastmin: Evaluate an optional final-minute holdout.
        plotting: Enable legacy diagnostic figures; false is safe for GUI workers.
        frames_per_minute: Movie FPS multiplied by 60.
        coarse_shape: Expected coarse `(x, y)` grid for validating seed indices.
        hz: Movie frame rate, used when ``frames_per_minute`` is omitted.
        show_sem_errorbars: Compatibility argument for legacy figure callers.

    Returns:
        tuple: Full-model predictions, refined preferred parameters, nonlinear
        parameters, rho/phi parameters, metrics, selectivity outputs, and
        interpolators in the legacy public ordering.

    Raises:
        ValueError: If phase cache axes, trial splits, frame range, or coarse seed coordinates are incompatible.
    """
    if tt is None:
        tt = [0, 18000]
    if train_idx is None:
        train_idx = [0, 2]
    if test_idx is None:
        test_idx = [1, 3]
    if frames_per_minute is None:
        if hz is None:
            raise ValueError(
                "run_Full_Model requires frames_per_minute or hz from the "
                "configured stimulus frame rate."
            )
        frames_per_minute = int(round(float(hz) * SECONDS_PER_MINUTE))
    frames_per_minute = int(frames_per_minute)
    hz = float(hz) if hz is not None else frames_per_minute / SECONDS_PER_MINUTE
    train_idx = [int(i) for i in train_idx]
    test_idx = [int(i) for i in test_idx]
    if not train_idx or not test_idx:
        raise ValueError("run_Full_Model requires at least one train and one test trial.")
    if spks.ndim != 3:
        raise ValueError(
            "run_Full_Model expects spikes with shape (trials, time, neurons); "
            f"got {spks.shape}."
        )
    n_trials = int(spks.shape[0])
    invalid = [i for i in train_idx + test_idx if i < 0 or i >= n_trials]
    if invalid:
        raise ValueError(f"Trial index/indices out of range for {n_trials} trials: {invalid}")
    overlap = sorted(set(train_idx) & set(test_idx))
    if overlap:
        raise ValueError(f"Train and test trial indices overlap: {overlap}")
    if len(tt) != 2 or int(tt[0]) < 0 or int(tt[1]) <= int(tt[0]):
        raise ValueError(f"run_Full_Model requires a valid [start, stop] frame range; got {tt!r}.")
    if int(tt[1]) > int(spks.shape[1]):
        raise ValueError(
            f"Full-model frame range ends at {tt[1]}, but neural data has only {spks.shape[1]} frames."
        )
    if idxs is None:
        selected_indices = list(range(int(spks.shape[2])))
    else:
        selected_indices = [int(idx) for idx in idxs]
    invalid_indices = [idx for idx in selected_indices if idx < 0 or idx >= int(spks.shape[2])]
    if invalid_indices:
        raise ValueError(
            f"Full-model neuron index/indices are outside 0-{spks.shape[2] - 1}: {invalid_indices}."
        )
    for label, params in (("raw", maxes0), ("smoothed", maxes1)):
        params = np.asarray(params)
        if params.ndim != 2 or params.shape[0] < 4 or params.shape[1] < int(spks.shape[2]):
            raise ValueError(
                f"run_Full_Model {label} RF parameters must have at least four rows and one column per neuron; "
                f"got {params.shape}."
            )
        if not np.all(np.isfinite(params[:4, selected_indices])):
            raise ValueError(
                f"run_Full_Model {label} RF parameters contain non-finite values for selected neurons. "
                "Re-run Coarse RF analysis."
            )
    Predictions = []
    Metrics = []
    Params = []
    nonlinParams = []
    RhoPhiParams = []
    OS = []
    xM, xm, yM, ym = visual_coverage
    interpolators = []

    if memmapping:
        print(f"[INFO] Run Full Model: loading disk-backed full wavelets from {wavelet_path}")
        try:
            wavelets_i = load_array(
                os.path.join(wavelet_path, 'dwt_videodata2_i.zarr'),
                mmap_mode='r',
            )
            wavelets_r = load_array(
                os.path.join(wavelet_path, 'dwt_videodata2_r.zarr'),
                mmap_mode='r',
            )
        except Exception:
            print("[INFO] Run Full Model: Zarr phase cache unavailable; using NPY memory maps.")
            from ..stimulus import load_stimulus_simple_cell2

            wavelets_r, wavelets_i = load_stimulus_simple_cell2(
                wavelet_path,
                tt=tt,
                downsampling=False,
            )
            memmapping = False
    else:
        from ..stimulus import load_stimulus_simple_cell2

        wavelets_r, wavelets_i = load_stimulus_simple_cell2(
            wavelet_path,
            tt=tt,
            downsampling=False,
        )

    nx_full, ny_full, n_orientations, n_sigmas_w, n_frequencies = wavelet_feature_dims(
        wavelets_r,
        sigmas=np.asarray(sigmas),
        frequencies=np.asarray(frequencies),
    )
    if wavelets_r.ndim != 6 or wavelets_i.ndim != 6:
        raise ValueError(
            "run_Full_Model expects full-model wavelets with shape "
            "(time, NX, NY, n_thetas, n_sigmas_full, n_frequencies). "
            f"Got {wavelets_r.shape} and {wavelets_i.shape}."
        )
    if wavelets_r.shape != wavelets_i.shape:
        raise ValueError(
            f"Real/imag full-model wavelets must share a shape, got "
            f"{wavelets_r.shape} and {wavelets_i.shape}."
        )
    if int(tt[1]) > int(wavelets_r.shape[0]):
        raise ValueError(
            f"Full-model frame range ends at {tt[1]}, but full wavelets contain only "
            f"{wavelets_r.shape[0]} frames. Rebuild the full-model wavelet cache for this movie."
        )
    if neuron_pos.ndim != 2 or neuron_pos.shape[0] < int(spks.shape[2]):
        raise ValueError(
            "Full-model neuron positions do not match the neural cache: "
            f"positions {getattr(neuron_pos, 'shape', None)}, neurons {spks.shape[2]}."
        )
    if coarse_shape is None:
        # Preserve the historical scripted API while allowing the GUI to use
        # arbitrary metadata-derived downsampling percentages.
        scale_x, scale_y = coarse_to_full_scale(nx_full, ny_full)
    else:
        coarse_nx, coarse_ny = (int(coarse_shape[0]), int(coarse_shape[1]))
        if coarse_nx <= 0 or coarse_ny <= 0:
            raise ValueError(f"coarse_shape must contain positive dimensions, got {coarse_shape!r}")
        scale_x, scale_y = nx_full / coarse_nx, ny_full / coarse_ny
        for label, params in (("raw", maxes0), ("smoothed", maxes1)):
            coordinates = np.asarray(params[:2, selected_indices], dtype=float)
            upper = np.asarray((coarse_nx - 1, coarse_ny - 1), dtype=float)[:, None]
            if np.any(coordinates < 0) or np.any(coordinates > upper):
                raise ValueError(
                    f"run_Full_Model {label} coarse RF coordinates are outside "
                    f"the current coarse grid {(coarse_nx, coarse_ny)}. Re-run Coarse RF analysis."
                )
    corr_shape = (n_orientations, n_sigmas_w, n_frequencies)
    compute_device = "cuda" if torch.cuda.is_available() else "cpu"

    def _corr_features(features, response):
        """Function for corr features.

        Args:
            features: Input value for this operation.
            response: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        features = np.asarray(features, dtype=np.float32)
        response = np.asarray(response, dtype=np.float32).reshape(-1)
        if features.shape[0] != response.shape[0]:
            raise ValueError(
                f"Feature/response time mismatch: {features.shape[0]} != {response.shape[0]}"
            )
        flat = features.reshape(features.shape[0], -1).T
        with torch.no_grad():
            feature_tensor = torch.as_tensor(flat, device=compute_device, dtype=torch.float32)
            response_tensor = torch.as_tensor(response.reshape(1, -1), device=compute_device, dtype=torch.float32)
            feature_tensor = feature_tensor - feature_tensor.mean(dim=1, keepdim=True)
            response_tensor = response_tensor - response_tensor.mean(dim=1, keepdim=True)
            denom = torch.linalg.norm(feature_tensor, dim=1) * torch.linalg.norm(response_tensor)
            corr = torch.matmul(feature_tensor, response_tensor.T).squeeze(1) / torch.clamp(denom, min=1e-12)
            corr = torch.nan_to_num(corr).cpu().numpy()
        return corr

    def _best_phase_correlation(real_features, imag_features, response, output_shape):
        """Function for best phase correlation.

        Args:
            real_features: Input value for this operation.
            imag_features: Input value for this operation.
            response: Input value for this operation.
            output_shape: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if isinstance(real_features, _WaveletWindow):
            # Initial full-model refinement used to materialize a five-minute
            # x/y window.  The lazy view delegates to chunk-aligned RF
            # sufficient statistics instead.
            response_matrix = np.asarray(response, dtype=np.float32).reshape(-1, 1)
            corr_real = streaming_cross_correlation(real_features, response_matrix).reshape(output_shape)
            corr_imag = streaming_cross_correlation(imag_features, response_matrix).reshape(output_shape)
        else:
            corr_real = _corr_features(real_features, response).reshape(output_shape)
            corr_imag = _corr_features(imag_features, response).reshape(output_shape)
        real_score = np.nanmax(np.abs(corr_real))
        imag_score = np.nanmax(np.abs(corr_imag))
        if real_score >= imag_score:
            corr = corr_real
            phase = "real"
        else:
            corr = corr_imag
            phase = "imaginary"
        best = np.unravel_index(np.nanargmax(np.abs(corr)), corr.shape)
        return phase, corr, tuple(int(i) for i in best)

    def _window_bounds(x0, y0, radius):
        """Function for window bounds.

        Args:
            x0: Input value for this operation.
            y0: Input value for this operation.
            radius: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        x_start = max(0, x0 - radius)
        x_end = min(nx_full, x0 + radius + 1)
        y_start = max(0, y0 - radius)
        y_end = min(ny_full, y0 + radius + 1)
        return x_start, x_end, y_start, y_end

    def _findBestPos_phase_specific(x, y, o, s, nmin=5, plotting=False):
        """Function for findBestPos phase specific.

        Args:
            x: Input value for this operation.
            y: Input value for this operation.
            o: Input value for this operation.
            s: Input value for this operation.
            nmin: Input value for this operation.
            plotting: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        # Small synthetic or cropped movies can be narrower than the legacy
        # five-pixel margin. Clipping to real feature bounds keeps refinement
        # valid for every positive full-resolution grid.
        x0 = int(np.clip(np.round(x * scale_x), 0, nx_full - 1))
        y0 = int(np.clip(np.round(y * scale_y), 0, ny_full - 1))
        w = 10
        t_start = int(tt[0])
        t_end = min(t_start + nmin * frames_per_minute, wavelets_r.shape[0], spks.shape[1])
        if t_end - t_start <= 1:
            raise ValueError("Fine refinement needs at least two training frames.")
        spk_train = np.mean(spks[train_idx, t_start:t_end, idx], axis=0)
        x_start, x_end, y_start, y_end = _window_bounds(x0, y0, w)
        wavelets_r_ = _WaveletWindow(wavelets_r, t_start, t_end, x_start, x_end, y_start, y_end)
        wavelets_i_ = _WaveletWindow(wavelets_i, t_start, t_end, x_start, x_end, y_start, y_end)
        phase, cc_initial, best = _best_phase_correlation(
            wavelets_r_,
            wavelets_i_,
            spk_train,
            wavelets_r_.shape[1:],
        )
        x_local, y_local, o, s, f = best
        x_global = x_start + x_local
        y_global = y_start + y_local

        for iteration in range(10):
            phase, cc_f_1, osf_best = _best_phase_correlation(
                np.asarray(wavelets_r[t_start:t_end, x_global, y_global, :, :, :]),
                np.asarray(wavelets_i[t_start:t_end, x_global, y_global, :, :, :]),
                spk_train,
                corr_shape,
            )
            o, s, f = osf_best

            if plotting:
                fig, ax = plt.subplots(n_orientations)
                fig.suptitle(f"Local orientation/size/frequency correlation search ({phase} phase)")
                if n_orientations == 1:
                    ax = [ax]
                for i in range(n_orientations):
                    ax[i].imshow(cc_f_1[i].T, vmin=-np.max(abs(cc_f_1)), vmax=np.max(abs(cc_f_1)), cmap='coolwarm')
                    ax[i].set_title(f"Orientation {i}")
                    ax[i].set_xlabel("Size index")
                    ax[i].set_ylabel("Frequency index")

            spatial_features = (
                np.asarray(wavelets_r[t_start:t_end, x_start:x_end, y_start:y_end, o, s, f])
                if phase == "real"
                else np.asarray(wavelets_i[t_start:t_end, x_start:x_end, y_start:y_end, o, s, f])
            )
            cc_f_1_xy = _corr_features(spatial_features, spk_train).reshape(spatial_features.shape[1:])
            new_x_local, new_y_local = np.unravel_index(np.nanargmax(np.abs(cc_f_1_xy)), cc_f_1_xy.shape)
            new_x_global = x_start + int(new_x_local)
            new_y_global = y_start + int(new_y_local)

            if plotting:
                plt.figure()
                plt.imshow(cc_f_1_xy.T, vmax=np.max(abs(cc_f_1_xy)), cmap='coolwarm', aspect='equal')
                plt.xticks(np.linspace(0, cc_f_1_xy.shape[0] - 1, 3).astype(int))
                plt.yticks(np.linspace(0, cc_f_1_xy.shape[1] - 1, 3).astype(int))
                plt.title(f"Local x/y correlation search ({phase} phase)")
                plt.xlabel("Local X position (pixels)")
                plt.ylabel("Local Y position (pixels)")

            if new_x_global == x_global and new_y_global == y_global:
                break
            x_local, y_local = int(new_x_local), int(new_y_local)
            x_global, y_global = new_x_global, new_y_global

        return (x_global, y_global, o, s, f)
    
    
    def findBestPos_profiled(x, y, o, s, nmin=5, plotting=False):
        """Function for findBestPos profiled.

        Args:
            x: Input value for this operation.
            y: Input value for this operation.
            o: Input value for this operation.
            s: Input value for this operation.
            nmin: Input value for this operation.
            plotting: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        return _findBestPos_phase_specific(x, y, o, s, nmin=nmin, plotting=plotting)


    def findBestPos(x, y, o, s, nmin=5, plotting=False):
        """Function for findBestPos.

        Args:
            x: Input value for this operation.
            y: Input value for this operation.
            o: Input value for this operation.
            s: Input value for this operation.
            nmin: Input value for this operation.
            plotting: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        return _findBestPos_phase_specific(x, y, o, s, nmin=nmin, plotting=plotting)
    
    
    train_start = int(tt[0])
    train_stop = min(train_start + n_min * frames_per_minute, wavelets_r.shape[0], spks.shape[1])
    if train_stop - train_start <= 1:
        raise ValueError("Full model needs at least two training frames.")

    list_neurons = selected_indices
    for idx in list_neurons:  # np.asarray(neuron_pos[:, 1]>600).nonzero()[0]:[1024, 732, 1789, 3279, 614]:#
        x, y, o, s = maxes1[:4, idx]

        x1, y1, o1, s1 = maxes0[:4, idx]

        if memmapping:
            (x, y, o, s, f) = findBestPos_profiled(int(np.round(x)), int(np.round(y)), int(np.round(o)),
                                                   int(np.round(s)), nmin=n_min, plotting=plotting)

            (x1, y1, o1, s1, f1) = findBestPos_profiled(int(np.round(x1)), int(np.round(y1)), int(np.round(o1)),
                                                        int(np.round(s1)), nmin=n_min, plotting=plotting)
        else:
            (x, y, o, s, f) = findBestPos(int(np.round(x)), int(np.round(y)), int(np.round(o)),
                                          int(np.round(s)), nmin=n_min, plotting=plotting)

            (x1, y1, o1, s1, f1) = findBestPos(int(np.round(x1)), int(np.round(y1)), int(np.round(o1)),
                                               int(np.round(s1)), nmin=n_min, plotting=plotting)

        if memmapping:
            wavelets_i_ = wavelets_i[train_start:train_stop, x, y, :, :, :]
            wavelets_r_ = wavelets_r[train_start:train_stop, x, y, :, :, :]
            wc = np.sqrt(np.power(wavelets_r_, 2) + np.power(
                wavelets_i_, 2))
        else:
            wc = np.sqrt(np.power(wavelets_r[train_start:train_stop, x, y, :, :, :], 2) + np.power(
                wavelets_i[train_start:train_stop, x, y, :, :, :], 2))

        train_frames = wc.shape[0]
        wc_tensor = torch.as_tensor(wc.reshape(train_frames, -1).T, device=compute_device, dtype=torch.float32)
        spks_tensor = torch.as_tensor(np.mean(spks[train_idx, train_start:train_stop, idx], axis=0).reshape(1, -1), device=compute_device, dtype=torch.float32)
        
        cc_f_1_o = torch.corrcoef(torch.cat((wc_tensor, spks_tensor), dim=0)).cpu().numpy()[-1:, :-1]
        cc_f_1_o = cc_f_1_o.reshape(corr_shape)
        pp = cc_f_1_o[:, s, f]
        ori_selectivity = signaltonoiseScipy(pp)  # abs(np.max(pp)) / abs(np.mean(pp))

        OS.append(ori_selectivity)
        Params.append([[x, y, o, s, f], [x1, y1, o1, s1, f1]])

        if plotting:
            plt.rcParams.update({
                "font.size": 8,
                "svg.fonttype": "none"  # Texte éditable dans Inkscape
            })

            if memmapping:
                wavelets_i_ = wavelets_i[train_start:train_stop, x, y, :, :, :]
                wavelets_r_ = wavelets_r[train_start:train_stop, x, y, :, :, :]
                wc = np.sqrt(np.power(wavelets_r_, 2) + np.power(wavelets_i_, 2))
            else:
                wc = np.sqrt(np.power(wavelets_r[train_start:train_stop, x, y, :, :, :], 2) + np.power(
                    wavelets_i[train_start:train_stop, x, y, :, :, :], 2))

            plot_frames = wc.shape[0]
            wc_tensor_plot = torch.as_tensor(wc.reshape(plot_frames, -1).T, device=compute_device, dtype=torch.float32)
            
            cc_f_1_xy = torch.corrcoef(torch.cat((wc_tensor_plot, spks_tensor), dim=0)).cpu().numpy()[-1:, :-1]
            if cc_f_1_xy.size == nx_full * ny_full:
                cc_f_1_xy = cc_f_1_xy.reshape(nx_full, ny_full)
            else:
                side = int(np.sqrt(cc_f_1_xy.size))
                cc_f_1_xy = cc_f_1_xy.reshape(side, side)

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
            ax[0].set_xlabel("Azimuth (deg)")
            ax[0].set_ylabel("Correlation (r)")

            ax[2].plot(np.append(cc_f_1_o[:, s, f], cc_f_1_o[0, s, f]), 'o-', c='k', markersize=markersize)
            ax[2].set_xticks(
                [0, max(1, n_orientations // 2), max(2, n_orientations - 1)],
                [0, 90, 180],
            )
            ax[2].spines["top"].set_visible(False)
            ax[2].spines["right"].set_visible(False)
            ax[2].set_title('Orientation (deg)')
            ax[2].set_xlabel("Orientation (deg)")
            ax[2].set_ylabel("Correlation (r)")

            ax[1].plot(elev_val, i * cc_f_1_xy[x, :], c='k')
            ax[1].set_xticks([-30, 0, 30])
            ax[1].spines["top"].set_visible(False)
            ax[1].spines["right"].set_visible(False)
            ax[1].set_title('Elevation (deg)')
            ax[1].set_xlabel("Elevation (deg)")
            ax[1].set_ylabel("Correlation (r)")

            mm = max(cc_f_1_o.min(), cc_f_1_o.max(), key=abs)

            ax[3].plot(sigmas, cc_f_1_o[o, :, f], 'o-', c='k', markersize=markersize)
            ax[3].set_xticks(sigmas)
            ax[3].spines["top"].set_visible(False)
            ax[3].spines["right"].set_visible(False)
            ax[3].set_title('Size (deg)')
            ax[3].set_xlabel("Size (deg)")
            ax[3].set_ylabel("Correlation (r)")

            ax[4].plot(frequencies, cc_f_1_o[o, s, :], 'o-', c='k', markersize=markersize)
            ax[4].set_xticks(np.floor(frequencies * 100).astype(int) / 100)
            ax[4].tick_params(axis='x', labelrotation=45)
            ax[4].spines["top"].set_visible(False)
            ax[4].spines["right"].set_visible(False)
            ax[4].set_title('Frequency (cdp)')
            ax[4].set_xlabel("Spatial frequency (cycles/deg)")
            ax[4].set_ylabel("Correlation (r)")

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
        phi = np.unwrap(phi)
        dphi = np.diff(phi, prepend=phi[0]) * hz
        dphi = np.clip(dphi, -2 * np.pi, 2 * np.pi)
        nans, dx = nan_helper(dphi)
        dphi[nans] = np.interp(dx(nans), dx(~nans), dphi[~nans])

        # Vectorized polar coordinate calculation (100x faster, zero unnecessary RAM)
        rho_inhib = np.hypot(w_r_inhib.flatten(), w_i_inhib.flatten())
        phi_inhib = np.arctan2(w_i_inhib.flatten(), w_r_inhib.flatten())
        phi_inhib = np.unwrap(phi_inhib)
        dphi_inhib = np.diff(phi_inhib, prepend=phi_inhib[0]) * hz
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
            ax[5].set_xlabel("Amplitude rho (a.u.)")
            ax[5].set_ylabel("Response gain (a.u.)")

            b_x = np.linspace(0, 2 * np.pi, ncut + 1)
            ax[6].plot(b_x, plot[1], c='k')
            ax[6].set_ylim(bottom=0, top=np.max(plot[1]))
            ax[6].spines["top"].set_visible(False)
            ax[6].spines["right"].set_visible(False)
            ax[6].set_xticks([0, np.pi, 2 * np.pi])
            ax[6].set_xticklabels([0, r'$\pi$', r'$2\pi$'])
            ax[6].set_title('Phase (rad)')
            ax[6].set_xlabel("Phase phi (rad)")
            ax[6].set_ylabel("Response gain (a.u.)")

            c_x = np.linspace(-1, 1, ncut)
            ax[7].plot(c_x, plot[2], c='k')
            ax[7].set_ylim(bottom=0, top=np.max(plot[2]))
            ax[7].spines["top"].set_visible(False)
            ax[7].spines["right"].set_visible(False)
            ax[7].set_xticks([-1, 0, 1])
            ax[7].set_title('Drift (a.u.)')
            ax[7].set_xlabel("Drift dphi (a.u.)")
            ax[7].set_ylabel("Response gain (a.u.)")
            ax[8].axis("off")

            pref_phase = b_x[np.argmax(plot[1])]
            pref_ori = thetas[o] * 180 / np.pi
            pref_size = sigmas[s]

            plt.title('Receptive field ' + str(idx))

        if plotting:
            star = 4500
            stop = star + 500
            spk = spks[:, :, idx]
            fig, ax = plt.subplots(7, 1, sharex=True)
            plt.rcParams.update({
                "font.size": 8,
                "svg.fonttype": "none"  # Texte éditable dans Inkscape
            })
            x_trace = np.arange(star, stop)
            sem_trials = _plot_trace_with_optional_sem(
                ax[2],
                spk[:, star:stop],
                x=x_trace,
                color='k',
                label='Trial-averaged spike activity',
                show_sem_errorbars=show_sem_errorbars,
            )
            _set_sem_caption(fig, sem_trials)
            ax[2].set_title("Spike activity")
            ax[2].set_ylabel("Activity (a.u.)")
            ax[3].plot(x_trace, rho[star:stop], c='k')
            ax[3].set_title("Amplitude")
            ax[3].set_ylabel("rho (a.u.)")
            ax3 = ax[3].twinx()
            ax3.plot(x_trace, np.mean(spk, axis=0)[star:stop], c='k', linestyle='dotted')
            ax3.set_ylabel("Activity (a.u.)")
            ax[4].plot(x_trace, np.unwrap(phi[star:stop]), c='k')
            ax[4].set_title("Phase")
            ax[4].set_ylabel("phi (rad)")
            ax[4].yaxis.set_major_locator(ticker.MultipleLocator(base=2 * np.pi))
            ax[4].yaxis.set_major_formatter(FuncFormatter(pi_formatter))
            ax4 = ax[4].twinx()
            ax4.plot(x_trace, np.mean(spk, axis=0)[star:stop], c='k', linestyle='dotted')
            ax4.set_ylabel("Activity (a.u.)")
            dphi_smooth = rolling_avg(dphi, 10)[15:-15]
            dphi_segment = dphi_smooth[star:stop]
            ax[5].plot(x_trace[:len(dphi_segment)], dphi_segment)
            ax[5].set_title("Drift")
            ax[5].set_ylabel("dphi (rad/s)")
            ax[5].yaxis.set_major_locator(ticker.MultipleLocator(base=2 * np.pi))
            ax[5].yaxis.set_major_formatter(FuncFormatter(pi_formatter))
            ax[6].plot(x_trace, vis_resp[star:stop], c='r')
            ax[6].set_title("Model prediction")
            ax[6].set_xlabel("Frame index")
            ax[6].set_ylabel("Predicted activity (a.u.)")
            ax6 = ax[6].twinx()
            ax6.plot(x_trace, np.mean(spk, axis=0)[star:stop], c='k', linestyle='--')
            ax6.set_ylabel("Activity (a.u.)")
            ax[0].axis("off")
            ax[1].axis("off")
        Predictions.append(vis_resp)
        nonlinParams.append(nonlinparams)
        RhoPhiParams.append(rhophiparams)

        Metrics.append(a)
        interpolators.append(interp)

    M = [[m[0], m[1], m[2][0][1], m[3], m[4]] for m in Metrics]

    folder_name = 'model_results'
    full_path = os.path.join(savepath, folder_name)
    os.makedirs(full_path, exist_ok=True)

    model_tag = str(n_orientations)
    np.save(os.path.join(full_path , f'RPdp_predictions_{model_tag}_noneigh_c_smoothpos_{n_min}.npy'), Predictions)
    np.save(os.path.join(full_path , f'RPdp_nonlinparams_{model_tag}_noneigh_c_smoothpos_{n_min}.npy'), nonlinParams)
    np.save(os.path.join(full_path ,  f'RPdp_rhophiparams_{model_tag}_noneigh_c_smoothpos_{n_min}.npy'), RhoPhiParams)
    np.save(os.path.join(full_path ,  f'RPdp_metrics_{model_tag}_noneigh_c_smoothpos_{n_min}.npy'), M)
    np.save(os.path.join(full_path ,  f'RPdp_os_{model_tag}_noneigh_c_smoothpos_{n_min}.npy'), OS)
    np.save(os.path.join(full_path , f'RPdp_params_{model_tag}_noneigh_c_smoothpos_{n_min}.npy'), np.array(Params))
    with open(os.path.join(full_path ,  "interpolators_" + str(n_min) + '.pkl'), "wb") as f:
        pickle.dump(interpolators, f)
    return Predictions, Params, nonlinParams, RhoPhiParams, Metrics, OS, interpolators


def plotNeuralRaster(spks):
    """Plot neurons sorted by the first sparse-SVD component.

    Parameters
    ----------
    spks : np.ndarray
        Trial-by-time-by-neuron activity array.

    Returns
    -------
    None
        Displays a matplotlib raster plot.
    """
    test = spks.reshape(-1, spks.shape[2])
    k = 3
    res = svds(test, k, which='LM')
    u, s, v = res

    n_sort_ind = np.argsort(v[0])

    sorted_neurons = test[:, n_sort_ind[np.arange(0, spks.shape[2], 1)]]
    plt.figure()
    plt.imshow(sorted_neurons.T, vmin=0, vmax=0.8, cmap='Greys')
