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
    return round(float((r_pref - r_orth) / denom), 6)


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
    return round(float(numerator / denominator), 6)


def orientation_selectivity_from_tuning(orientation_tuning, angles_deg=None):
    """Return OSI and gOSI for a one-dimensional orientation tuning curve."""
    rates = np.asarray(orientation_tuning, dtype=float)
    if rates.ndim != 1:
        raise ValueError("orientation_tuning must be one-dimensional")
    if rates.size > 1 and np.isclose(rates[0], rates[-1]):
        rates = rates[:-1]
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, rates.size, endpoint=False)
    else:
        angles_deg = np.asarray(angles_deg, dtype=float)
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
