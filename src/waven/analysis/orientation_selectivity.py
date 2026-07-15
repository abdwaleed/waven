"""Orientation selectivity metrics for tuning curves."""
from __future__ import annotations

import numpy as np


def _clean_rates(rates):
    """Function for clean rates.

    Args:
        rates: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    rates = np.asarray(rates, dtype=float)
    rates = np.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
    if rates.size and np.nanmin(rates) < 0:
        rates = rates - np.nanmin(rates)
    return rates


def close_orientation_curve(angles_deg, values):
    """Return a consistently ordered orientation curve closed at 180 degrees.

    Orientation is axial: 0 and 180 degrees denote the same physical axis.
    Plotting an extra copy of the first value at the first angle plus 180 makes
    that periodicity visible without including the duplicated value in OSI or
    gOSI calculations.
    """
    angles = np.asarray(angles_deg, dtype=float).reshape(-1) % 180.0
    curve = np.asarray(values, dtype=float).reshape(-1)
    if angles.size != curve.size:
        raise ValueError("angles_deg and values must have the same length")
    if angles.size == 0:
        return angles, curve
    order = np.argsort(angles, kind="stable")
    angles = angles[order]
    curve = curve[order]
    # A caller may already provide a periodic endpoint. Retain one copy only.
    if angles.size > 1 and np.isclose(angles[0], angles[-1]):
        angles = angles[:-1]
        curve = curve[:-1]
    if angles.size == 0:
        return angles, curve
    return np.append(angles, angles[0] + 180.0), np.append(curve, curve[0])


def calculate_osi(angles_deg, rates):
    """Calculate the orientation selectivity index using pref/orth responses."""
    angles_deg = np.asarray(angles_deg, dtype=float) % 180
    rates = _clean_rates(rates)
    if angles_deg.shape[0] != rates.shape[0]:
        raise ValueError("angles_deg and rates must have the same length")
    if rates.size == 0:
        return np.nan

    pref_idx = int(np.nanargmax(rates))
    pref_angle = angles_deg[pref_idx]
    orth_angle = (pref_angle + 90) % 180
    angular_distance = np.abs(((angles_deg - orth_angle + 90) % 180) - 90)
    orth_idx = int(np.nanargmin(angular_distance))

    r_pref = float(rates[pref_idx])
    r_orth = float(rates[orth_idx])
    denom = r_pref + r_orth
    if denom == 0:
        return np.nan
    # Preserve low but real selectivity values. Rounding here previously turned
    # weak responses into literal zeros before population statistics were made.
    return float((r_pref - r_orth) / denom)


def calculate_gosi(angles_deg, rates):
    """Calculate global orientation selectivity from the vector sum."""
    angles_deg = np.asarray(angles_deg, dtype=float)
    rates = _clean_rates(rates)
    if angles_deg.shape[0] != rates.shape[0]:
        raise ValueError("angles_deg and rates must have the same length")

    denominator = float(np.sum(rates))
    if denominator == 0:
        return np.nan
    numerator = np.abs(np.sum(rates * np.exp(2 * np.deg2rad(angles_deg) * 1j)))
    return float(numerator / denominator)


def orientation_selectivity_from_tuning(orientation_tuning, angles_deg=None):
    """Return OSI and gOSI for a one-dimensional orientation tuning curve."""
    rates = np.asarray(orientation_tuning, dtype=float)
    if rates.ndim != 1:
        raise ValueError("orientation_tuning must be one-dimensional")
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, rates.size, endpoint=False)
    else:
        angles_deg = np.asarray(angles_deg, dtype=float)
        if angles_deg.size == rates.size and angles_deg.size > 1:
            if np.isclose(angles_deg[0] % 180, angles_deg[-1] % 180):
                angles_deg = angles_deg[:-1]
                rates = rates[:-1]
        if angles_deg.size == rates.size + 1 and np.isclose(angles_deg[0] % 180, angles_deg[-1] % 180):
            angles_deg = angles_deg[:-1]
        if angles_deg.size != rates.size:
            raise ValueError("angles_deg must match the orientation tuning length")
    return calculate_osi(angles_deg, rates), calculate_gosi(angles_deg, rates)


def selectivity_for_rfs(rfs, angles_deg=None):
    """Compute OSI/gOSI for each neuron from an RF tuple returned by RF analysis."""
    rf_values = np.asarray(rfs[0])
    maxes = np.asarray(rfs[1], dtype=int)
    if rf_values.ndim != 6:
        raise ValueError(f"Expected RF values with 6 dimensions, got {rf_values.shape}")
    n_neurons = rf_values.shape[0]
    n_orientations = rf_values.shape[3]
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, n_orientations, endpoint=False)

    osi = np.full(n_neurons, np.nan, dtype=float)
    gosi = np.full(n_neurons, np.nan, dtype=float)
    orientation_tuning = np.full((n_neurons, n_orientations), np.nan, dtype=float)
    for neuron_idx in range(n_neurons):
        x, y, _o, size_idx, freq_idx = maxes[:5, neuron_idx]
        tuning = rf_values[neuron_idx, x, y, :, size_idx, freq_idx]
        orientation_tuning[neuron_idx] = tuning
        osi[neuron_idx], gosi[neuron_idx] = orientation_selectivity_from_tuning(
            tuning,
            angles_deg,
        )
    return {
        "angles_deg": np.asarray(angles_deg, dtype=float),
        "orientation_tuning": orientation_tuning,
        "osi": osi,
        "gosi": gosi,
    }


def firing_rate_orientation_tuning(spikes, wavelets_complex, rfs, angles_deg=None):
    """Compute non-negative firing-rate tuning curves at preferred RF features.

    Args:
        spikes: Trial-aligned neural responses with shape
            ``(n_trials, n_frames, n_neurons)`` or an averaged response with
            shape ``(n_frames, n_neurons)``. Ephys alignment stores firing rates
            in Hz; two-photon values are the aligned non-negative activity
            produced by the suite2p path.
        wavelets_complex: Non-negative coarse wavelet energy with shape
            ``(n_frames, nx, ny, n_orientations, n_sigmas)`` or
            ``(n_frames, nx, ny, n_orientations, n_sigmas, n_frequencies)``.
        rfs: RF tuple returned by receptive-field analysis. The preferred
            feature indices in ``rfs[1]`` choose the x/y/sigma/frequency slice
            used to turn frame-wise firing into orientation tuning.
        angles_deg: Optional orientation bin centers in degrees.

    Returns:
        dict: ``angles_deg``, ``orientation_tuning``, ``osi``, and ``gosi``.
            The tuning values are weighted mean firing rates, not RF
            correlations.
    """
    responses = np.asarray(spikes, dtype=float)
    trial_responses = responses if responses.ndim == 3 else None
    if responses.ndim == 3:
        responses = np.nanmean(responses, axis=0)
    if responses.ndim != 2:
        raise ValueError(f"Expected spikes with 2 or 3 dimensions, got {responses.shape}")
    responses = np.nan_to_num(responses, nan=0.0, posinf=0.0, neginf=0.0)
    responses = np.clip(responses, 0.0, None)

    wavelets = wavelets_complex
    wavelet_shape = tuple(getattr(wavelets, "shape", ()))
    if len(wavelet_shape) not in {5, 6}:
        raise ValueError(f"Expected wavelets with 5 or 6 dimensions, got {wavelet_shape}")

    maxes = np.asarray(rfs[1], dtype=int)
    n_frames = min(responses.shape[0], wavelet_shape[0])
    n_neurons = min(responses.shape[1], maxes.shape[1])
    n_orientations = wavelet_shape[3]
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, n_orientations, endpoint=False)

    osi = np.full(n_neurons, np.nan, dtype=float)
    gosi = np.full(n_neurons, np.nan, dtype=float)
    orientation_tuning = np.full((n_neurons, n_orientations), np.nan, dtype=float)
    trial_orientation_tuning = None
    if trial_responses is not None:
        trial_orientation_tuning = np.full(
            (trial_responses.shape[0], n_neurons, n_orientations), np.nan, dtype=float
        )
    for neuron_idx in range(n_neurons):
        x, y, _o, size_idx, freq_idx = maxes[:5, neuron_idx]
        if not (
            0 <= x < wavelet_shape[1]
            and 0 <= y < wavelet_shape[2]
            and 0 <= size_idx < wavelet_shape[4]
            and (len(wavelet_shape) == 5 or 0 <= freq_idx < wavelet_shape[5])
        ):
            continue
        if len(wavelet_shape) == 5:
            weights = np.asarray(wavelets[:n_frames, x, y, :, size_idx], dtype=float)
        else:
            weights = np.asarray(wavelets[:n_frames, x, y, :, size_idx, freq_idx], dtype=float)
        weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, None)
        rates = responses[:n_frames, neuron_idx]
        denom = np.sum(weights, axis=0)
        tuning = np.divide(
            np.sum(weights * rates[:, None], axis=0),
            denom,
            out=np.zeros(n_orientations, dtype=float),
            where=denom > 0,
        )
        orientation_tuning[neuron_idx] = tuning
        if trial_orientation_tuning is not None:
            trial_rates = np.clip(
                np.nan_to_num(trial_responses[:, :n_frames, neuron_idx], nan=0.0),
                0.0,
                None,
            )
            trial_orientation_tuning[:, neuron_idx, :] = np.divide(
                np.sum(weights[None, :, :] * trial_rates[:, :, None], axis=1),
                denom[None, :],
                out=np.zeros((trial_rates.shape[0], n_orientations), dtype=float),
                where=denom[None, :] > 0,
            )
        osi[neuron_idx], gosi[neuron_idx] = orientation_selectivity_from_tuning(
            tuning,
            angles_deg,
        )

    result = {
        "angles_deg": np.asarray(angles_deg, dtype=float),
        "orientation_tuning": orientation_tuning,
        "osi": osi,
        "gosi": gosi,
        "source": "firing_rate",
    }
    if trial_orientation_tuning is not None:
        result["trial_orientation_tuning"] = trial_orientation_tuning
        if trial_orientation_tuning.shape[0] > 1:
            result["orientation_sem"] = np.nanstd(
                trial_orientation_tuning, axis=0, ddof=1
            ) / np.sqrt(trial_orientation_tuning.shape[0])
    return result
