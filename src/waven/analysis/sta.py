"""Spike-triggered receptive-field estimation for frame-aligned ephys data.

The existing coarse-RF workflow correlates firing rates with wavelet power.
STA is intentionally separate: it uses *integer spike counts* and the signed,
mean-centred stimulus itself.  The implementation streams movie chunks from
NPY memmaps or Zarr arrays, so a large stimulus is never expanded in RAM.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..runtime.performance import (
    OperationTelemetry,
    available_ram_bytes,
    compute_devices,
    enabled_feature,
    gpu_available_vram_bytes,
    model_parallel_jobs,
    sta_shuffle_batch_size,
)
from ..runtime.task_control import check_cancelled


GABOR_PARAMETER_NAMES = (
    "x0_px",
    "y0_px",
    "sigma_x_px",
    "sigma_y_px",
    "theta_rad",
    "frequency_cycles_per_px",
    "cosine_amplitude",
    "sine_amplitude",
    "baseline",
    "amplitude",
    "phase_rad",
)


def _fft_actual_sta(
    movie: Any,
    frame_counts: np.ndarray,
    lag_frames: np.ndarray,
    movie_mean: np.ndarray,
    scale: float,
    device: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return exact actual STAs for all lags with bounded CUDA FFT blocks.

    FFT is useful only for in-memory movies with many requested lags.  The
    shuffled null still uses bounded batched GEMMs because its truncated,
    circularly shifted windows would otherwise require an impractically large
    frequency-domain cube.  Returning ``None`` selects the universally safe
    streaming path for memmaps, Zarr arrays, CPUs, and small lag requests.
    """
    if not enabled_feature("STA_FFT", default=False) or not str(device).startswith("cuda"):
        return None
    if not isinstance(movie, np.ndarray) or isinstance(movie, np.memmap) or len(lag_frames) < 8:
        return None
    try:
        import torch

        n_frames = int(movie.shape[0])
        pixels = int(np.prod(movie.shape[1:]))
        n_neurons = int(frame_counts.shape[1])
        n_fft = 1 << (2 * n_frames - 1).bit_length()
        # Complex product plus inverse output; reserve most VRAM for the
        # existing GUI/caches and use a neuron block if needed.
        budget = max(1, int(gpu_available_vram_bytes() * 0.08))
        frequency_bins = n_fft // 2 + 1
        neuron_block = max(1, min(n_neurons, 32))
        pixel_block = max(1, min(pixels, budget // max(1, frequency_bins * neuron_block * 32)))
        if pixel_block < 1:
            return None

        counts_tensor = torch.as_tensor(frame_counts, dtype=torch.float32, device=device)
        counts_fft = torch.fft.rfft(counts_tensor, n=n_fft, dim=0)
        result = np.empty((len(lag_frames), pixels, n_neurons), dtype=np.float32)
        movie_flat = np.asarray(movie, dtype=np.float32).reshape(n_frames, pixels)
        centered_mean = movie_mean.reshape(1, pixels)
        lag_tensor = torch.as_tensor(lag_frames, dtype=torch.long, device=device)
        for pixel_start in range(0, pixels, pixel_block):
            pixel_end = min(pixels, pixel_start + pixel_block)
            stimulus = np.array(movie_flat[:, pixel_start:pixel_end], dtype=np.float32, copy=True)
            stimulus -= centered_mean[:, pixel_start:pixel_end]
            if scale != 1.0:
                stimulus *= np.float32(scale)
            stimulus_fft = torch.fft.rfft(
                torch.as_tensor(stimulus, dtype=torch.float32, device=device), n=n_fft, dim=0
            )
            for neuron_start in range(0, n_neurons, neuron_block):
                neuron_end = min(n_neurons, neuron_start + neuron_block)
                products = stimulus_fft.conj().unsqueeze(-1) * counts_fft[:, None, neuron_start:neuron_end]
                correlations = torch.fft.irfft(products, n=n_fft, dim=0)
                result[:, pixel_start:pixel_end, neuron_start:neuron_end] = (
                    correlations.index_select(0, lag_tensor).cpu().numpy()
                )
                del products, correlations
            del stimulus_fft
        totals = np.empty((len(lag_frames), n_neurons), dtype=np.float32)
        count_sums = np.cumsum(frame_counts[::-1], axis=0, dtype=np.float64)[::-1]
        for index, lag in enumerate(lag_frames):
            totals[index] = count_sums[int(lag)]
            result[index] = np.divide(
                result[index],
                totals[index][None, :],
                out=np.zeros_like(result[index]),
                where=totals[index][None, :] > 0,
            )
        return result, totals
    except RuntimeError as exc:
        print(f"STA FFT setup failed; using streamed GEMMs: {exc}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


@dataclass
class STAResult:
    """Disk-cacheable STA results for all requested lags and neurons.

    Arrays use ``(lags, neurons, y, x)`` image order.  This differs from the
    wavelet feature grid deliberately: STA is a direct image-domain result.
    """

    images: np.ndarray
    lag_frames: np.ndarray
    lag_ms: np.ndarray
    noise_std: np.ndarray
    peak_values: np.ndarray
    significant: np.ndarray
    total_spikes: np.ndarray
    gabor_params: np.ndarray
    gabor_rmse: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_paths: Dict[str, Path] = field(default_factory=dict)


def _movie_statistics(movie: Any, chunk_size: int, cancel_event=None) -> Tuple[np.ndarray, float, float]:
    """Return a streamed per-pixel movie mean plus the global extrema.

    STA must remove each pixel's temporal baseline, not one scalar luminance
    baseline for the whole screen.  A scalar leaves static spatial content
    (notably a display-sync/photodiode patch) in every lag image, where it can
    be mistaken for a receptive field.
    """
    n_frames = int(movie.shape[0])
    if n_frames < 1:
        raise ValueError("The STA stimulus movie contains no pixels.")
    pixel_sum = np.zeros(tuple(int(dim) for dim in movie.shape[1:]), dtype=np.float64)
    frames_seen = 0
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, n_frames, chunk_size):
        check_cancelled(cancel_event)
        block = np.asarray(movie[start : min(n_frames, start + chunk_size)])
        if not block.size:
            continue
        pixel_sum += np.sum(block, axis=0, dtype=np.float64)
        frames_seen += int(block.shape[0])
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
    if frames_seen == 0:
        raise ValueError("The STA stimulus movie contains no pixels.")
    return (pixel_sum / frames_seen).astype(np.float32), minimum, maximum


def _sta_chunk_size(n_frames: int, pixels: int, n_neurons: int, device: str) -> int:
    """Choose a bounded chunk from available host/device memory."""
    per_frame = max(1, (pixels + n_neurons) * np.dtype(np.float32).itemsize)
    if str(device).startswith("cuda:"):
        try:
            device_id = int(str(device).split(":", 1)[1])
        except (IndexError, ValueError):
            device_id = 0
        budget = max(1, int(gpu_available_vram_bytes(device_id) * 0.12))
        # Input, response, output workspace, and transfer headroom.
        return max(8, min(n_frames, 1024, budget // max(per_frame * 6, 1)))
    budget = max(1, int(available_ram_bytes() * 0.04))
    return max(8, min(n_frames, 1024, budget // max(per_frame * 4, 1)))


def _matmul_stimulus_counts(stimulus: np.ndarray, counts: np.ndarray, device: str) -> np.ndarray:
    """Return ``stimulus.T @ counts`` with bounded CUDA fallback support."""
    if not str(device).startswith("cuda:"):
        return stimulus.T @ counts
    try:
        import torch

        with torch.no_grad():
            stim_tensor = torch.as_tensor(stimulus, dtype=torch.float32, device=device)
            count_tensor = torch.as_tensor(counts, dtype=torch.float32, device=device)
            product = (stim_tensor.T @ count_tensor).cpu().numpy()
        del stim_tensor, count_tensor
        return product
    except Exception as exc:
        # A full STA should still complete on CPU if one chunk does not fit a
        # busy GPU.  The caller keeps chunking, so this fallback stays bounded.
        print(f"STA CUDA chunk failed on {device}; using CPU for this chunk: {exc}")
        return stimulus.T @ counts


def _phase_gabor_components(
    params: Sequence[float], height: int, width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return envelope, cosine carrier, and sine carrier for a phase Gabor."""
    x0, y0, sigma_x, sigma_y, theta, frequency = map(float, params[:6])
    yy, xx = np.indices((height, width), dtype=np.float64)
    x = xx - x0
    y = yy - y0
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    x_rot = x * cos_theta + y * sin_theta
    y_rot = -x * sin_theta + y * cos_theta
    envelope = np.exp(
        -0.5
        * ((x_rot / max(sigma_x, 1e-6)) ** 2 + (y_rot / max(sigma_y, 1e-6)) ** 2)
    )
    phase_argument = 2.0 * np.pi * frequency * x_rot
    return envelope, np.cos(phase_argument), np.sin(phase_argument)


def phase_gabor_image(params: Sequence[float], height: int, width: int) -> np.ndarray:
    """Evaluate the STA-specific sine/cosine phase Gabor model.

    The model is ``baseline + envelope * (a_cos*cos(carrier) +
    a_sin*sin(carrier))``.  Fitting the two carrier amplitudes independently
    is what allows it to recover the sign-dependent stripes in an STA image.
    """
    if len(params) < 9 or not np.all(np.isfinite(np.asarray(params[:9], dtype=float))):
        return np.full((height, width), np.nan, dtype=np.float32)
    envelope, cosine, sine = _phase_gabor_components(params, height, width)
    cosine_amplitude = float(params[6])
    sine_amplitude = float(params[7])
    baseline = float(params[8])
    return (baseline + envelope * (cosine_amplitude * cosine + sine_amplitude * sine)).astype(np.float32)


def fit_phase_gabor(sta_image: np.ndarray, max_nfev: int = 300) -> Tuple[np.ndarray, float]:
    """Fit a phase-sensitive spatial Gabor to one significant STA image.

    Returns the eleven values described by :data:`GABOR_PARAMETER_NAMES` and
    an RMSE.  Failed/degenerate fits are represented by all-NaN parameters.
    This never uses the squared real/imaginary (power) formulation used by the
    wavelet RF workflow.
    """
    image = np.asarray(sta_image, dtype=np.float64)
    if image.ndim != 2 or min(image.shape) < 3 or not np.any(np.isfinite(image)):
        return np.full(len(GABOR_PARAMETER_NAMES), np.nan, dtype=np.float32), np.nan
    from scipy.optimize import least_squares

    height, width = image.shape
    finite = np.isfinite(image)
    values = image[finite]
    value_range = float(np.ptp(values))
    if value_range <= np.finfo(float).eps:
        return np.full(len(GABOR_PARAMETER_NAMES), np.nan, dtype=np.float32), np.nan

    peak_y, peak_x = np.unravel_index(np.nanargmax(np.abs(image)), image.shape)
    baseline = float(np.nanmedian(values))
    amplitude_bound = max(value_range * 8.0, 1e-6)
    spatial_max = float(max(height, width))
    lower = np.array(
        [0.0, 0.0, 0.5, 0.5, 0.0, 1.0 / max(spatial_max * 2.0, 2.0), -amplitude_bound, -amplitude_bound, baseline - amplitude_bound],
        dtype=float,
    )
    upper = np.array(
        [width - 1.0, height - 1.0, spatial_max, spatial_max, np.pi, 0.5, amplitude_bound, amplitude_bound, baseline + amplitude_bound],
        dtype=float,
    )
    yy, xx = np.indices(image.shape, dtype=np.float64)

    def residuals(params: np.ndarray) -> np.ndarray:
        envelope, cosine, sine = _phase_gabor_components(params, height, width)
        prediction = params[8] + envelope * (params[6] * cosine + params[7] * sine)
        return (prediction[finite] - image[finite]).ravel()

    best = None
    for theta in np.linspace(0.0, 3.0 * np.pi / 4.0, 4):
        initial = np.array(
            [
                float(peak_x),
                float(peak_y),
                max(1.0, width / 5.0),
                max(1.0, height / 5.0),
                theta,
                min(0.25, max(0.04, 2.0 / max(width, height))),
                float(image[peak_y, peak_x] - baseline),
                0.0,
                baseline,
            ],
            dtype=float,
        )
        try:
            candidate = least_squares(
                residuals,
                initial,
                bounds=(lower, upper),
                method="trf",
                max_nfev=max_nfev,
            )
        except Exception:
            continue
        if not candidate.success or not np.all(np.isfinite(candidate.x)):
            continue
        score = float(np.mean(candidate.fun ** 2))
        if best is None or score < best[0]:
            best = (score, candidate.x)

    if best is None:
        return np.full(len(GABOR_PARAMETER_NAMES), np.nan, dtype=np.float32), np.nan
    score, fitted = best
    amplitude = float(np.hypot(fitted[6], fitted[7]))
    # a_cos*cos(qx) + a_sin*sin(qx) = amplitude*cos(qx + phase).
    phase = float(np.arctan2(-fitted[7], fitted[6]))
    params = np.concatenate((fitted, np.array([amplitude, phase], dtype=float))).astype(np.float32)
    return params, float(np.sqrt(score))


def _fit_significant_gabors(images: np.ndarray, significant: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit only STA images that passed the shuffle threshold."""
    n_lags, n_neurons = significant.shape
    params = np.full((n_lags, n_neurons, len(GABOR_PARAMETER_NAMES)), np.nan, dtype=np.float32)
    rmse = np.full((n_lags, n_neurons), np.nan, dtype=np.float32)
    work = [(lag, neuron) for lag, neuron in zip(*np.nonzero(significant))]
    if not work:
        return params, rmse

    def one_fit(lag: int, neuron: int):
        fitted, error = fit_phase_gabor(np.asarray(images[lag, neuron]))
        return lag, neuron, fitted, error

    workers = min(len(work), max(1, model_parallel_jobs()))
    if workers == 1:
        completed = map(lambda item: one_fit(*item), work)
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="waven-sta-fit")
        completed = executor.map(lambda item: one_fit(*item), work)
    try:
        for lag, neuron, fitted, error in completed:
            params[lag, neuron] = fitted
            rmse[lag, neuron] = error
    finally:
        if workers != 1:
            executor.shutdown(wait=True)
    return params, rmse


def _normalize_sta_output_format(output_format: str) -> str:
    """Validate the persistent STA array format."""
    fmt = str(output_format).strip().lower()
    if fmt not in {"npy", "zarr"}:
        raise ValueError(f"Unsupported STA output format {output_format!r}; choose 'npy' or 'zarr'.")
    return fmt


def _write_zarr_array(
    path: Path,
    array: Any = None,
    chunks: Optional[Tuple[int, ...]] = None,
    *,
    shape: Optional[Tuple[int, ...]] = None,
    dtype: Optional[np.dtype] = None,
):
    """Create/write a compressed Zarr array while supporting Zarr v2 and v3."""
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ImportError("STA Zarr output requires 'zarr' and 'numcodecs'; choose NPY or install project requirements.") from exc
    if array is not None:
        shape = tuple(int(dim) for dim in array.shape)
        dtype = np.dtype(array.dtype)
    elif shape is None or dtype is None:
        raise ValueError("Zarr output needs either an array or both shape and dtype.")
    else:
        shape = tuple(int(dim) for dim in shape)
        dtype = np.dtype(dtype)
    chunks = chunks or tuple(max(1, min(dim, 256)) for dim in shape)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    kwargs = {"mode": "w", "shape": shape, "chunks": chunks, "dtype": dtype}
    try:
        store = zarr.open(str(path), compressor=compressor, **kwargs)
    except TypeError:
        store = zarr.open(str(path), compressors=[compressor], **kwargs)
    if array is not None:
        store[:] = array
    return store


def _write_sta_cache(result: STAResult, output_dir: Path, output_format: str = "npy") -> Dict[str, Path]:
    """Persist STA arrays in NPY or compressed Zarr format."""
    fmt = _normalize_sta_output_format(output_format)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".npy" if fmt == "npy" else ".zarr"
    paths = {
        "images": output_dir / f"sta_images{suffix}",
        "lag_frames": output_dir / f"sta_lag_frames{suffix}",
        "lag_ms": output_dir / f"sta_lag_ms{suffix}",
        "noise_std": output_dir / f"sta_noise_std{suffix}",
        "peak_values": output_dir / f"sta_peak_values{suffix}",
        "significant": output_dir / f"sta_significant{suffix}",
        "total_spikes": output_dir / f"sta_total_spikes{suffix}",
        "gabor_params": output_dir / f"sta_phase_gabor_params{suffix}",
        "gabor_rmse": output_dir / f"sta_phase_gabor_rmse{suffix}",
        "metadata": output_dir / "sta_metadata.json",
    }
    arrays = {
        "images": result.images,
        "lag_frames": result.lag_frames,
        "lag_ms": result.lag_ms,
        "noise_std": result.noise_std,
        "peak_values": result.peak_values,
        "significant": result.significant,
        "total_spikes": result.total_spikes,
        "gabor_params": result.gabor_params,
        "gabor_rmse": result.gabor_rmse,
    }
    source_image_path = result.cache_paths.get("images")
    for name, array in arrays.items():
        if name == "images" and source_image_path is not None:
            try:
                if Path(source_image_path).resolve() == paths[name].resolve():
                    continue
            except OSError:
                pass
        if fmt == "npy":
            np.save(paths[name], np.asarray(array))
        else:
            _write_zarr_array(paths[name], array)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)
    return paths


def write_sta_result(result: STAResult, output_dir: Path, output_format: str = "npy") -> Dict[str, Path]:
    """Export an in-memory STA result in the requested persistent format."""
    return _write_sta_cache(result, Path(output_dir), output_format)


def compute_sta(
    movie: Any,
    spike_counts: np.ndarray,
    fps: float,
    *,
    max_lag_ms: float = 150.0,
    n_shuffles: int = 100,
    significance_sd: float = 3.0,
    shuffle_seconds: Tuple[float, float] = (3.0, 5.0),
    scale_stimulus: bool = True,
    random_seed: Optional[int] = 0,
    output_dir: Optional[Path] = None,
    output_format: str = "npy",
    cancel_event=None,
) -> STAResult:
    """Compute frame-lagged, shuffle-tested ephys STAs from raw spike counts.

    ``movie`` is an array-like ``(frames, height, width)`` object, including a
    NumPy memmap or Zarr array.  ``spike_counts`` must be ``(trials, frames,
    neurons)`` and contain counts rather than firing rates.  Trials are kept
    separate logically: their count vectors are summed at each matching movie
    frame, rather than concatenating across trial boundaries.
    """
    if float(fps) <= 0:
        raise ValueError("STA requires a positive stimulus FPS.")
    if len(getattr(movie, "shape", ())) != 3:
        raise ValueError(f"STA movie must have shape (frames, height, width), got {getattr(movie, 'shape', None)}.")
    counts = np.asarray(spike_counts)
    if counts.ndim != 3:
        raise ValueError(f"STA spike counts must have shape (trials, frames, neurons), got {counts.shape}.")
    n_movie_frames, height, width = map(int, movie.shape)
    n_trials, n_frames, n_neurons = map(int, counts.shape)
    if n_trials < 1 or n_neurons < 1:
        raise ValueError("STA requires at least one trial and one neuron.")
    if n_movie_frames != n_frames:
        raise ValueError(
            "STA movie and spike-count frame axes must match exactly: "
            f"movie={n_movie_frames}, spike_counts={n_frames}."
        )
    if np.any(counts < 0):
        raise ValueError("STA spike counts must be non-negative.")
    if int(n_shuffles) < 1:
        raise ValueError("STA requires at least one circular-shuffle draw.")
    if float(max_lag_ms) < 0:
        raise ValueError("STA maximum lag must be non-negative.")

    frame_interval_ms = 1000.0 / float(fps)
    lag_frames = np.arange(
        int(np.floor(float(max_lag_ms) / frame_interval_ms + 1e-9)) + 1,
        dtype=np.int32,
    )
    lag_frames = lag_frames[lag_frames < n_frames]
    if lag_frames.size == 0:
        raise ValueError("STA lag window contains no alignable movie frames.")
    lag_ms = lag_frames.astype(np.float32) * frame_interval_ms
    pixels = int(height * width)

    devices = compute_devices(allow_multi_gpu=True)
    chunk_size = _sta_chunk_size(n_frames, pixels, n_neurons, devices[0])
    movie_mean, movie_min, movie_max = _movie_statistics(movie, chunk_size, cancel_event)
    scale = 1.0
    if scale_stimulus:
        # The global extrema give a safe bound on deviation from every pixel's
        # temporal mean without reading the disk-backed movie a second time.
        scale = 1.0 / max(
            abs(movie_min - float(np.max(movie_mean))),
            abs(movie_max - float(np.min(movie_mean))),
            1e-12,
        )
    frame_counts = np.sum(counts, axis=0, dtype=np.float32)
    # Releasing the caller's trial-sized input early matters for ephys sessions
    # with many trials. ``frame_counts`` is enough because every trial shows
    # the same stimulus frame at a given frame index.
    del counts

    image_shape = (len(lag_frames), n_neurons, height, width)
    output_format = _normalize_sta_output_format(output_format)
    image_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / f"sta_images.{output_format}"
        if output_format == "npy":
            images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float32, shape=image_shape)
        else:
            images = _write_zarr_array(
                image_path,
                chunks=(1, max(1, min(n_neurons, 32)), max(1, min(height, 128)), max(1, min(width, 128))),
                shape=image_shape,
                dtype=np.float32,
            )
    else:
        images = np.empty(image_shape, dtype=np.float32)

    min_shift = int(np.ceil(float(shuffle_seconds[0]) * float(fps)))
    max_shift = int(np.floor(float(shuffle_seconds[1]) * float(fps)))
    min_shift = max(1, min_shift)
    max_shift = min(max_shift, n_frames - 1)
    if min_shift > max_shift:
        min_shift, max_shift = 1, max(1, n_frames - 1)
    rng = np.random.default_rng(random_seed)
    shifts_by_lag = {
        int(lag): rng.integers(min_shift, max_shift + 1, size=int(n_shuffles), endpoint=False)
        for lag in lag_frames
    }
    telemetry = OperationTelemetry("Spike-triggered averaging")

    def centered_block(start: int, stop: int) -> np.ndarray:
        # ``np.asarray`` can be a view for ordinary arrays and NPY memmaps.
        # Centre a bounded working copy so the cached stimulus is never
        # modified while STA iterates through lags and shuffles.
        block = np.asarray(movie[start:stop], dtype=np.float32).reshape(stop - start, pixels).copy()
        block -= movie_mean.reshape(1, pixels)
        if scale_stimulus:
            block *= np.float32(scale)
        return block

    def one_sta_batch(lag: int, shifts: Sequence[Optional[int]], device: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute one lag for several actual/circular-shuffle count vectors.

        The old implementation reread every stimulus chunk once per shuffle.
        Here a bounded batch concatenates several shifted response vectors into
        one matrix product.  It is mathematically identical while reducing
        movie I/O and GPU-launch overhead by the batch size.
        """
        valid_frames = n_frames - lag
        shifts = tuple(shifts)
        n_draws = len(shifts)
        weighted = np.zeros((pixels, n_draws, n_neurons), dtype=np.float64)
        denominator = np.zeros((n_draws, n_neurons), dtype=np.float64)
        for start in range(0, valid_frames, chunk_size):
            check_cancelled(cancel_event)
            stop = min(valid_frames, start + chunk_size)
            stimulus = centered_block(start, stop)
            response_indices = np.arange(lag + start, lag + stop, dtype=np.int64)
            indexed_responses = []
            for shift in shifts:
                indices = response_indices if shift is None else (response_indices - int(shift)) % n_frames
                indexed_responses.append(np.asarray(frame_counts[indices], dtype=np.float32))
            response = np.stack(indexed_responses, axis=1)
            product = _matmul_stimulus_counts(
                stimulus, response.reshape(stop - start, n_draws * n_neurons), device
            ).reshape(pixels, n_draws, n_neurons)
            weighted += product
            denominator += np.sum(response, axis=0, dtype=np.float64)
        nonzero = denominator > 0
        weighted = np.divide(
            weighted,
            denominator[None, :, :],
            out=np.zeros_like(weighted),
            where=nonzero[None, :, :],
        )
        return weighted.astype(np.float32), denominator.astype(np.float32)

    fft_actual = _fft_actual_sta(
        movie, frame_counts, lag_frames, movie_mean, scale, devices[0]
    )
    if fft_actual is not None:
        print("STA using bounded CUDA FFT for the actual lagged averages; shuffled null uses batched GEMMs.")

    def one_lag(lag_index: int, lag: int, device: str):
        check_cancelled(cancel_event)
        if fft_actual is None:
            actual_batch, totals_batch = one_sta_batch(lag, (None,), device)
            actual_flat = actual_batch[:, 0, :]
            totals = totals_batch[0]
        else:
            actual_flat = fft_actual[0][lag_index]
            totals = fft_actual[1][lag_index]
        null_sum = np.zeros(n_neurons, dtype=np.float64)
        null_sum_sq = np.zeros(n_neurons, dtype=np.float64)
        shifts = shifts_by_lag[lag]
        batch_size = sta_shuffle_batch_size(pixels, n_neurons, len(shifts), device)
        for shuffle_start in range(0, len(shifts), batch_size):
            check_cancelled(cancel_event)
            batch_shifts = shifts[shuffle_start:shuffle_start + batch_size]
            shuffled_flat, _ = one_sta_batch(lag, tuple(int(shift) for shift in batch_shifts), device)
            null_sum += np.sum(shuffled_flat, axis=(0, 1), dtype=np.float64)
            null_sum_sq += np.sum(shuffled_flat * shuffled_flat, axis=(0, 1), dtype=np.float64)
            if shuffle_start and shuffle_start % max(10, batch_size) == 0:
                telemetry.maybe_report()
        null_samples = float(int(n_shuffles) * pixels)
        variance = np.maximum(null_sum_sq / null_samples - (null_sum / null_samples) ** 2, 0.0)
        return lag_index, actual_flat, totals, np.sqrt(variance).astype(np.float32)

    lag_results = {}
    if len(devices) > 1 and len(lag_frames) > 1:
        print(f"STA using {len(devices)} GPUs: {', '.join(devices)}")
        with ThreadPoolExecutor(max_workers=min(len(devices), len(lag_frames)), thread_name_prefix="waven-sta") as executor:
            futures = {
                executor.submit(one_lag, index, int(lag), devices[index % len(devices)]): index
                for index, lag in enumerate(lag_frames)
            }
            for future in as_completed(futures):
                result = future.result()
                lag_results[result[0]] = result
    else:
        print(f"STA using {devices[0]} with {chunk_size} stimulus frames per chunk.")
        for index, lag in enumerate(lag_frames):
            result = one_lag(index, int(lag), devices[0])
            lag_results[index] = result
            print(f"STA lag {index + 1}/{len(lag_frames)} complete ({lag_ms[index]:.1f} ms).")

    total_spikes = np.empty((len(lag_frames), n_neurons), dtype=np.float32)
    noise_std = np.empty_like(total_spikes)
    for index in range(len(lag_frames)):
        _, flat_image, totals, null_std = lag_results[index]
        images[index] = flat_image.T.reshape(n_neurons, height, width)
        total_spikes[index] = totals
        noise_std[index] = null_std
    if isinstance(images, np.memmap):
        images.flush()

    peak_values = np.empty((len(lag_frames), n_neurons), dtype=np.float32)
    for index in range(len(lag_frames)):
        peak_values[index] = np.max(np.abs(np.asarray(images[index]).reshape(n_neurons, pixels)), axis=1)
    significant = (total_spikes > 0) & (noise_std > 0) & (peak_values >= float(significance_sd) * noise_std)
    gabor_params, gabor_rmse = _fit_significant_gabors(images, significant)
    telemetry.report()

    result = STAResult(
        images=images,
        lag_frames=lag_frames,
        lag_ms=lag_ms,
        noise_std=noise_std,
        peak_values=peak_values,
        significant=significant,
        total_spikes=total_spikes,
        gabor_params=gabor_params,
        gabor_rmse=gabor_rmse,
        metadata={
            "version": 2,
            "movie_shape": [n_movie_frames, height, width],
            "spike_count_shape": [n_trials, n_frames, n_neurons],
            "fps": float(fps),
            "movie_mean": float(np.mean(movie_mean, dtype=np.float64)),
            "movie_centering": "per_pixel_temporal_mean",
            "movie_scale": float(scale),
            "stimulus_scaled_to_minus_one_one": bool(scale_stimulus),
            "max_lag_ms": float(max_lag_ms),
            "n_shuffles": int(n_shuffles),
            "shuffle_seconds": [float(shuffle_seconds[0]), float(shuffle_seconds[1])],
            "significance_sd": float(significance_sd),
            "gabor_parameter_names": list(GABOR_PARAMETER_NAMES),
            "gabor_model": "baseline + envelope * (a_cos*cos(carrier) + a_sin*sin(carrier))",
            "output_format": output_format,
        },
        cache_paths={"images": image_path} if image_path is not None else {},
    )
    if output_dir is not None:
        result.cache_paths = _write_sta_cache(result, output_dir, output_format)
    return result


__all__ = [
    "GABOR_PARAMETER_NAMES",
    "STAResult",
    "compute_sta",
    "write_sta_result",
    "fit_phase_gabor",
    "phase_gabor_image",
]
