"""Standard PSTH-weighted spike-triggered averages for Coarse RF inspection.

This implementation intentionally accepts the aligned, trial-averaged response
already used by the Coarse RF Individual Neuron view. It calculates one direct
stimulus average for each preceding-frame lag; it does not use shuffle tests,
Gabor fitting, a separate cache, or a separate analysis workflow.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_MAX_WINDOW_MS = 300.0


@dataclass(frozen=True)
class PSTHSTAResult:
    """STA maps and lag diagnostics for one trial-averaged neuron response."""

    maps: np.ndarray
    lag_frames: np.ndarray
    lag_ms: np.ndarray
    variances: np.ndarray
    peak_lag_index: int

    @property
    def peak_lag_frame(self) -> int:
        """Frame lag whose spatial STA map has the greatest pixel variance."""
        return int(self.lag_frames[self.peak_lag_index])

    @property
    def peak_lag_ms(self) -> float:
        """Time lag in milliseconds whose spatial STA map has greatest variance."""
        return float(self.lag_ms[self.peak_lag_index])


def lag_frames_within_window(fps: float, max_lag: int, window_ms: float = DEFAULT_MAX_WINDOW_MS) -> np.ndarray:
    """Return integer lags that do not exceed the requested time window.

    For example, at 30 Hz a 300 ms window contains lags 0 through 9. A lag of
    10 would be 333.3 ms and is deliberately excluded.
    """
    if not np.isfinite(fps) or float(fps) <= 0:
        raise ValueError("STA requires a positive stimulus frame rate.")
    if int(max_lag) < 0:
        raise ValueError("max_lag must be zero or greater.")
    if not np.isfinite(window_ms) or float(window_ms) < 0:
        raise ValueError("window_ms must be zero or greater.")
    maximum_from_time = int(np.floor(float(window_ms) * float(fps) / 1000.0 + 1e-12))
    return np.arange(min(int(max_lag), maximum_from_time) + 1, dtype=np.int32)


def compute_psth_sta(
    stimulus_movie: np.ndarray,
    averaged_psth: np.ndarray,
    fps: float,
    max_lag: int,
    *,
    window_ms: float = DEFAULT_MAX_WINDOW_MS,
) -> PSTHSTAResult:
    """Calculate standard rate-weighted STAs for all valid preceding-frame lags.

    ``averaged_psth[t]`` weights stimulus frame ``t - lag``. For each lag, the
    calculation is the standard STA weighted average:

    ``STA(lag) = sum_t PSTH[t] * stimulus[t - lag] / sum_t PSTH[t]``.

    The implementation uses one BLAS-backed vector/matrix product per lag, not
    a Python loop over stimulus frames. ``max_lag`` is additionally clipped to
    ``window_ms`` (300 ms by default), so no displayed map exceeds the chosen
    physiological time window.
    """
    movie = np.asarray(stimulus_movie)
    psth = np.asarray(averaged_psth, dtype=np.float64).reshape(-1)
    if movie.ndim != 3:
        raise ValueError(f"stimulus_movie must have shape (frames, y, x), got {movie.shape}.")
    if movie.shape[0] != psth.size:
        raise ValueError(
            "STA stimulus and averaged PSTH must have identical frame counts: "
            f"movie={movie.shape[0]}, psth={psth.size}."
        )
    if movie.shape[0] < 1:
        raise ValueError("STA requires at least one aligned stimulus frame.")
    if not np.all(np.isfinite(psth)):
        raise ValueError("averaged_psth must contain only finite values.")
    if np.any(psth < 0):
        raise ValueError("averaged_psth must be non-negative for a standard rate-weighted STA.")

    lags = lag_frames_within_window(fps, max_lag, window_ms)
    valid_lags = lags[lags < movie.shape[0]]
    if valid_lags.size == 0:
        raise ValueError("No STA lags remain after applying the frame/time limits.")

    height, width = movie.shape[1:]
    flat_movie = np.asarray(movie, dtype=np.float32).reshape(movie.shape[0], height * width)
    maps = np.zeros((valid_lags.size, height, width), dtype=np.float32)
    for output_index, lag in enumerate(valid_lags):
        weights = psth[int(lag):]
        weight_sum = float(np.sum(weights, dtype=np.float64))
        if weight_sum > 0:
            # One vector/matrix product computes every image pixel at once.
            maps[output_index] = (weights @ flat_movie[:movie.shape[0] - int(lag)] / weight_sum).reshape(height, width)

    variances = np.var(maps, axis=(1, 2), dtype=np.float64).astype(np.float32)
    peak_lag_index = int(np.argmax(variances))
    lag_ms = valid_lags.astype(np.float32) * (1000.0 / float(fps))
    return PSTHSTAResult(
        maps=maps,
        lag_frames=valid_lags,
        lag_ms=lag_ms,
        variances=variances,
        peak_lag_index=peak_lag_index,
    )


__all__ = ["DEFAULT_MAX_WINDOW_MS", "PSTHSTAResult", "compute_psth_sta", "lag_frames_within_window"]
