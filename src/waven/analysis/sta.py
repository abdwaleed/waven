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
    gpu_available_vram_bytes,
    model_parallel_jobs,
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


def _movie_statistics(movie: Any, chunk_size: int, cancel_event=None) -> Tuple[float, float, float]:
    """Return a streamed movie mean/min/max without materialising the movie."""
    n_frames = int(movie.shape[0])
    total = 0.0
    samples = 0
    minimum = np.inf
    maximum = -np.inf
    for start in range(0, n_frames, chunk_size):
        check_cancelled(cancel_event)
        block = np.asarray(movie[start : min(n_frames, start + chunk_size)])
        if not block.size:
            continue
        total += float(np.sum(block, dtype=np.float64))
        samples += int(block.size)
        minimum = min(minimum, float(np.min(block)))
        maximum = max(maximum, float(np.max(block)))
    if samples == 0:
        raise ValueError("The STA stimulus movie contains no pixels.")
    return total / samples, minimum, maximum


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


def _write_sta_cache(result: STAResult, output_dir: Path) -> Dict[str, Path]:
    """Persist small STA result arrays beside the streamed image memmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "images": output_dir / "sta_images.npy",
        "lag_frames": output_dir / "sta_lag_frames.npy",
        "lag_ms": output_dir / "sta_lag_ms.npy",
        "noise_std": output_dir / "sta_noise_std.npy",
        "peak_values": output_dir / "sta_peak_values.npy",
        "significant": output_dir / "sta_significant.npy",
        "total_spikes": output_dir / "sta_total_spikes.npy",
        "gabor_params": output_dir / "sta_phase_gabor_params.npy",
        "gabor_rmse": output_dir / "sta_phase_gabor_rmse.npy",
        "metadata": output_dir / "sta_metadata.json",
    }
    # ``images`` is normally already an on-disk memmap at this exact path.
    if not isinstance(result.images, np.memmap):
        np.save(paths["images"], np.asarray(result.images, dtype=np.float32))
    np.save(paths["lag_frames"], result.lag_frames)
    np.save(paths["lag_ms"], result.lag_ms)
    np.save(paths["noise_std"], result.noise_std)
    np.save(paths["peak_values"], result.peak_values)
    np.save(paths["significant"], result.significant)
    np.save(paths["total_spikes"], result.total_spikes)
    np.save(paths["gabor_params"], result.gabor_params)
    np.save(paths["gabor_rmse"], result.gabor_rmse)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(result.metadata, handle, indent=2)
    return paths


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
        scale = 1.0 / max(abs(movie_min - movie_mean), abs(movie_max - movie_mean), 1e-12)
    frame_counts = np.sum(counts, axis=0, dtype=np.float32)
    # Releasing the caller's trial-sized input early matters for ephys sessions
    # with many trials. ``frame_counts`` is enough because every trial shows
    # the same stimulus frame at a given frame index.
    del counts

    image_shape = (len(lag_frames), n_neurons, height, width)
    image_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "sta_images.npy"
        images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.float32, shape=image_shape)
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
        block -= np.float32(movie_mean)
        if scale_stimulus:
            block *= np.float32(scale)
        return block

    def one_sta(lag: int, shift: Optional[int], device: str) -> Tuple[np.ndarray, np.ndarray]:
        """Compute one lag's raw weighted average and count denominator."""
        valid_frames = n_frames - lag
        weighted = np.zeros((pixels, n_neurons), dtype=np.float64)
        denominator = np.zeros(n_neurons, dtype=np.float64)
        for start in range(0, valid_frames, chunk_size):
            check_cancelled(cancel_event)
            stop = min(valid_frames, start + chunk_size)
            stimulus = centered_block(start, stop)
            response_indices = np.arange(lag + start, lag + stop, dtype=np.int64)
            if shift is not None:
                # Exactly ``np.roll(frame_counts, shift)[lag:...]`` without
                # allocating a new ``frames x neurons`` array per shuffle.
                response_indices = (response_indices - int(shift)) % n_frames
            response = np.asarray(frame_counts[response_indices], dtype=np.float32)
            weighted += _matmul_stimulus_counts(stimulus, response, device)
            denominator += np.sum(response, axis=0, dtype=np.float64)
        nonzero = denominator > 0
        weighted[:, nonzero] /= denominator[nonzero]
        weighted[:, ~nonzero] = 0.0
        return weighted.astype(np.float32), denominator.astype(np.float32)

    def one_lag(lag_index: int, lag: int, device: str):
        check_cancelled(cancel_event)
        actual_flat, totals = one_sta(lag, None, device)
        null_sum = np.zeros(n_neurons, dtype=np.float64)
        null_sum_sq = np.zeros(n_neurons, dtype=np.float64)
        for shuffle_index, shift in enumerate(shifts_by_lag[lag], start=1):
            check_cancelled(cancel_event)
            shuffled_flat, _ = one_sta(lag, int(shift), device)
            null_sum += np.sum(shuffled_flat, axis=0, dtype=np.float64)
            null_sum_sq += np.sum(shuffled_flat * shuffled_flat, axis=0, dtype=np.float64)
            if shuffle_index % 10 == 0:
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

    peak_values = np.max(np.abs(images.reshape(len(lag_frames), n_neurons, pixels)), axis=2).astype(np.float32)
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
            "version": 1,
            "movie_shape": [n_movie_frames, height, width],
            "spike_count_shape": [n_trials, n_frames, n_neurons],
            "fps": float(fps),
            "movie_mean": float(movie_mean),
            "movie_scale": float(scale),
            "stimulus_scaled_to_minus_one_one": bool(scale_stimulus),
            "max_lag_ms": float(max_lag_ms),
            "n_shuffles": int(n_shuffles),
            "shuffle_seconds": [float(shuffle_seconds[0]), float(shuffle_seconds[1])],
            "significance_sd": float(significance_sd),
            "gabor_parameter_names": list(GABOR_PARAMETER_NAMES),
            "gabor_model": "baseline + envelope * (a_cos*cos(carrier) + a_sin*sin(carrier))",
        },
        cache_paths={"images": image_path} if image_path is not None else {},
    )
    if output_dir is not None:
        result.cache_paths = _write_sta_cache(result, output_dir)
    return result


__all__ = [
    "GABOR_PARAMETER_NAMES",
    "STAResult",
    "compute_sta",
    "fit_phase_gabor",
    "phase_gabor_image",
]
