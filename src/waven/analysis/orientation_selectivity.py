"""Orientation selectivity metrics for tuning curves."""
from __future__ import annotations

import math
import time

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

    if np.asarray(angles_deg).size != n_orientations:
        raise ValueError("angles_deg must match the coarse wavelet orientation axis")
    angles_deg = np.asarray(angles_deg, dtype=float)
    osi = np.full(n_neurons, np.nan, dtype=float)
    gosi = np.full(n_neurons, np.nan, dtype=float)
    orientation_tuning = np.full((n_neurons, n_orientations), np.nan, dtype=float)
    trial_orientation_tuning = None
    if trial_responses is not None:
        trial_orientation_tuning = np.full(
            (trial_responses.shape[0], n_neurons, n_orientations), np.nan, dtype=float
        )

    # A point selection from Zarr still decompresses its complete storage chunk.
    # The earlier feature-group loop therefore reread a spatial/time chunk for
    # almost every neuron.  Group preferred features by their Zarr chunk, read
    # each time/spatial/sigma/frequency tile once, then gather all matching
    # neuron weights and accumulate weighted means in a batched einsum.
    chunks = tuple(int(value) for value in (getattr(wavelets, "chunks", None) or wavelet_shape))
    time_chunk = max(1, min(n_frames, chunks[0]))
    x_chunk = max(1, min(wavelet_shape[1], chunks[1]))
    y_chunk = max(1, min(wavelet_shape[2], chunks[2]))
    sigma_chunk = max(1, min(wavelet_shape[4], chunks[4]))
    frequency_chunk = (
        max(1, min(wavelet_shape[5], chunks[5])) if len(wavelet_shape) == 6 else 1
    )
    chunk_groups = {}
    for neuron_idx in range(n_neurons):
        x, y, _o, size_idx, freq_idx = maxes[:5, neuron_idx]
        if not (
            0 <= x < wavelet_shape[1]
            and 0 <= y < wavelet_shape[2]
            and 0 <= size_idx < wavelet_shape[4]
            and (len(wavelet_shape) == 5 or 0 <= freq_idx < wavelet_shape[5])
        ):
            continue
        key = (
            int(x) // x_chunk,
            int(y) // y_chunk,
            int(size_idx) // sigma_chunk,
            int(freq_idx) // frequency_chunk if len(wavelet_shape) == 6 else 0,
        )
        chunk_groups.setdefault(key, []).append(neuron_idx)

    valid_neuron_ids = np.array(
        [neuron_id for neuron_ids in chunk_groups.values() for neuron_id in neuron_ids], dtype=int
    )
    weighted_sum = np.zeros((n_neurons, n_orientations), dtype=np.float64)
    weight_sum = np.zeros((n_neurons, n_orientations), dtype=np.float64)
    trial_weighted_sum = (
        np.zeros((trial_responses.shape[0], n_neurons, n_orientations), dtype=np.float64)
        if trial_responses is not None
        else None
    )
    total_groups = len(chunk_groups)
    report_every = max(1, min(10, math.ceil(total_groups / 10)))
    read_seconds = 0.0
    compute_seconds = 0.0
    started = time.perf_counter()
    print(
        f"[FIRING RATE] Starting chunk-batched orientation tuning | "
        f"neurons={valid_neuron_ids.size}/{n_neurons} | feature chunks={total_groups} | "
        f"time chunk={time_chunk}"
    )
    for group_number, ((x_group, y_group, sigma_group, frequency_group), neuron_ids) in enumerate(
        chunk_groups.items(), start=1
    ):
        neuron_ids = np.asarray(neuron_ids, dtype=int)
        feature_indices = maxes[:5, neuron_ids]
        x0, y0 = x_group * x_chunk, y_group * y_chunk
        sigma0 = sigma_group * sigma_chunk
        frequency0 = frequency_group * frequency_chunk
        x1 = min(wavelet_shape[1], x0 + x_chunk)
        y1 = min(wavelet_shape[2], y0 + y_chunk)
        sigma1 = min(wavelet_shape[4], sigma0 + sigma_chunk)
        frequency1 = (
            min(wavelet_shape[5], frequency0 + frequency_chunk)
            if len(wavelet_shape) == 6
            else 1
        )
        local_x = feature_indices[0] - x0
        local_y = feature_indices[1] - y0
        local_sigma = feature_indices[3] - sigma0
        local_frequency = feature_indices[4] - frequency0
        for time_start in range(0, n_frames, time_chunk):
            time_end = min(n_frames, time_start + time_chunk)
            read_started = time.perf_counter()
            if len(wavelet_shape) == 5:
                block = np.asarray(wavelets[time_start:time_end, x0:x1, y0:y1, :, sigma0:sigma1])
                weights = np.asarray(
                    np.moveaxis(block[:, local_x, local_y, :, local_sigma], 0, 1),
                    dtype=np.float64,
                )
            else:
                block = np.asarray(
                    wavelets[
                        time_start:time_end, x0:x1, y0:y1, :, sigma0:sigma1, frequency0:frequency1
                    ]
                )
                weights = np.asarray(
                    np.moveaxis(
                        block[:, local_x, local_y, :, local_sigma, local_frequency], 0, 1
                    ),
                    dtype=np.float64,
                )
            read_seconds += time.perf_counter() - read_started
            compute_started = time.perf_counter()
            np.nan_to_num(weights, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            np.maximum(weights, 0.0, out=weights)
            rates = np.take(responses[time_start:time_end], neuron_ids, axis=1)
            weighted_sum[neuron_ids] += np.sum(
                weights * rates[:, :, None], axis=0, dtype=np.float64
            )
            weight_sum[neuron_ids] += weights.sum(axis=0, dtype=np.float64)
            if trial_weighted_sum is not None:
                trial_rates = np.clip(
                    np.nan_to_num(
                        np.take(trial_responses[:, time_start:time_end, :], neuron_ids, axis=2),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                    0.0,
                    None,
                )
                trial_weighted_sum[:, neuron_ids] += np.sum(
                    weights[None, :, :, :] * trial_rates[:, :, :, None],
                    axis=1,
                    dtype=np.float64,
                )
            compute_seconds += time.perf_counter() - compute_started
        if group_number % report_every == 0 or group_number == total_groups:
            print(
                f"[FIRING RATE] Feature chunks {group_number}/{total_groups} | "
                f"neurons in chunk={neuron_ids.size} | read={read_seconds:.1f}s | compute={compute_seconds:.1f}s"
            )

    metric_started = time.perf_counter()
    if valid_neuron_ids.size:
        orientation_tuning[valid_neuron_ids] = np.divide(
            weighted_sum[valid_neuron_ids],
            weight_sum[valid_neuron_ids],
            out=np.zeros((valid_neuron_ids.size, n_orientations), dtype=float),
            where=weight_sum[valid_neuron_ids] > 0,
        )
        if trial_orientation_tuning is not None:
            trial_orientation_tuning[:, valid_neuron_ids] = np.divide(
                trial_weighted_sum[:, valid_neuron_ids],
                weight_sum[None, valid_neuron_ids],
                out=np.zeros(
                    (trial_responses.shape[0], valid_neuron_ids.size, n_orientations), dtype=float
                ),
                where=weight_sum[None, valid_neuron_ids] > 0,
            )

        # Tuning is explicitly non-negative above, so this vectorized form is
        # numerically equivalent to calculate_osi/calculate_gosi while avoiding
        # a Python-level metric pass over every neuron.
        rates = np.nan_to_num(orientation_tuning[valid_neuron_ids], nan=0.0)
        preferred = np.argmax(rates, axis=1)
        orientation_angles = angles_deg % 180.0
        orthogonal = (orientation_angles[preferred] + 90.0) % 180.0
        distances = np.abs(((orientation_angles[None, :] - orthogonal[:, None] + 90.0) % 180.0) - 90.0)
        orthogonal_indices = np.argmin(distances, axis=1)
        preferred_rates = rates[np.arange(rates.shape[0]), preferred]
        orthogonal_rates = rates[np.arange(rates.shape[0]), orthogonal_indices]
        osi_denominator = preferred_rates + orthogonal_rates
        osi_values = np.divide(
            preferred_rates - orthogonal_rates,
            osi_denominator,
            out=np.full(valid_neuron_ids.size, np.nan, dtype=float),
            where=osi_denominator > 0,
        )
        phase = np.exp(2j * np.deg2rad(orientation_angles))
        gosi_denominator = rates.sum(axis=1)
        gosi_values = np.divide(
            np.abs(rates @ phase),
            gosi_denominator,
            out=np.full(valid_neuron_ids.size, np.nan, dtype=float),
            where=gosi_denominator > 0,
        )
        osi[valid_neuron_ids] = osi_values
        gosi[valid_neuron_ids] = gosi_values
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
    metric_seconds = time.perf_counter() - metric_started
    print(
        f"[FIRING RATE] Completed orientation tuning | elapsed={time.perf_counter() - started:.1f}s | "
        f"chunk reads={read_seconds:.1f}s | weighted sums={compute_seconds:.1f}s | "
        f"OSI/gOSI and SEM={metric_seconds:.1f}s"
    )
    return result


def correlation_orientation_tuning(spikes, wavelets_complex, rfs, angles_deg=None):
    """Precompute trial-wise orientation-correlation curves at RF features.

    The mean curve is read from the established coarse-RF correlation tensor so
    its values remain identical to the displayed correlation graph.  Trial
    correlations are accumulated in feature-chunk groups, which avoids one
    Zarr read per neuron and keeps the compact ``trials x neurons x angles``
    result ready for display and export.
    """
    responses = np.asarray(spikes, dtype=float)
    if responses.ndim == 2:
        responses = responses[None, :, :]
    if responses.ndim != 3:
        raise ValueError(f"Expected spikes with 2 or 3 dimensions, got {responses.shape}")
    responses = np.nan_to_num(responses, nan=0.0, posinf=0.0, neginf=0.0)

    wavelets = wavelets_complex
    wavelet_shape = tuple(getattr(wavelets, "shape", ()))
    if len(wavelet_shape) not in {5, 6}:
        raise ValueError(f"Expected wavelets with 5 or 6 dimensions, got {wavelet_shape}")
    rf_values = np.asarray(rfs[0])
    maxes = np.asarray(rfs[1], dtype=int)
    n_frames = min(responses.shape[1], wavelet_shape[0])
    n_neurons = min(responses.shape[2], maxes.shape[1], rf_values.shape[0])
    n_trials = responses.shape[0]
    n_orientations = wavelet_shape[3]
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, n_orientations, endpoint=False)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if angles_deg.size != n_orientations:
        raise ValueError("angles_deg must match the coarse wavelet orientation axis")

    mean_tuning = np.full((n_neurons, n_orientations), np.nan, dtype=float)
    trial_tuning = np.full((n_trials, n_neurons, n_orientations), np.nan, dtype=float)
    chunks = tuple(int(value) for value in (getattr(wavelets, "chunks", None) or wavelet_shape))
    time_chunk = max(1, min(n_frames, chunks[0]))
    x_chunk = max(1, min(wavelet_shape[1], chunks[1]))
    y_chunk = max(1, min(wavelet_shape[2], chunks[2]))
    sigma_chunk = max(1, min(wavelet_shape[4], chunks[4]))
    frequency_chunk = max(1, min(wavelet_shape[5], chunks[5])) if len(wavelet_shape) == 6 else 1

    chunk_groups = {}
    for neuron_idx in range(n_neurons):
        x, y, _orientation, sigma_idx, frequency_idx = maxes[:5, neuron_idx]
        if not (
            0 <= x < wavelet_shape[1]
            and 0 <= y < wavelet_shape[2]
            and 0 <= sigma_idx < wavelet_shape[4]
            and (len(wavelet_shape) == 5 or 0 <= frequency_idx < wavelet_shape[5])
        ):
            continue
        mean_tuning[neuron_idx] = rf_values[neuron_idx, x, y, :, sigma_idx, frequency_idx]
        key = (
            int(x) // x_chunk,
            int(y) // y_chunk,
            int(sigma_idx) // sigma_chunk,
            int(frequency_idx) // frequency_chunk if len(wavelet_shape) == 6 else 0,
        )
        chunk_groups.setdefault(key, []).append(neuron_idx)

    sum_x = np.zeros((n_neurons, n_orientations), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    sum_y = np.zeros((n_trials, n_neurons), dtype=np.float64)
    sum_y2 = np.zeros_like(sum_y)
    sum_xy = np.zeros((n_trials, n_neurons, n_orientations), dtype=np.float64)
    started = time.perf_counter()
    for (x_group, y_group, sigma_group, frequency_group), neuron_ids in chunk_groups.items():
        neuron_ids = np.asarray(neuron_ids, dtype=int)
        feature_indices = maxes[:5, neuron_ids]
        x0, y0 = x_group * x_chunk, y_group * y_chunk
        sigma0 = sigma_group * sigma_chunk
        frequency0 = frequency_group * frequency_chunk
        x1 = min(wavelet_shape[1], x0 + x_chunk)
        y1 = min(wavelet_shape[2], y0 + y_chunk)
        sigma1 = min(wavelet_shape[4], sigma0 + sigma_chunk)
        frequency1 = min(wavelet_shape[5], frequency0 + frequency_chunk) if len(wavelet_shape) == 6 else 1
        local_x = feature_indices[0] - x0
        local_y = feature_indices[1] - y0
        local_sigma = feature_indices[3] - sigma0
        local_frequency = feature_indices[4] - frequency0
        for time_start in range(0, n_frames, time_chunk):
            time_end = min(n_frames, time_start + time_chunk)
            if len(wavelet_shape) == 5:
                block = np.asarray(wavelets[time_start:time_end, x0:x1, y0:y1, :, sigma0:sigma1])
                features = np.moveaxis(block[:, local_x, local_y, :, local_sigma], 0, 1)
            else:
                block = np.asarray(
                    wavelets[time_start:time_end, x0:x1, y0:y1, :, sigma0:sigma1, frequency0:frequency1]
                )
                features = np.moveaxis(
                    block[:, local_x, local_y, :, local_sigma, local_frequency], 0, 1
                )
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            trial_responses = np.take(responses[:, time_start:time_end, :], neuron_ids, axis=2)
            sum_x[neuron_ids] += np.sum(features, axis=0, dtype=np.float64)
            sum_x2[neuron_ids] += np.sum(features * features, axis=0, dtype=np.float64)
            sum_y[:, neuron_ids] += np.sum(trial_responses, axis=1, dtype=np.float64)
            sum_y2[:, neuron_ids] += np.sum(trial_responses * trial_responses, axis=1, dtype=np.float64)
            sum_xy[:, neuron_ids] += np.einsum(
                "tgo,ntg->ngo", features, trial_responses, optimize=True, dtype=np.float64
            )

    count = float(n_frames)
    numerator = count * sum_xy - sum_y[:, :, None] * sum_x[None, :, :]
    x_variance = np.maximum(count * sum_x2 - sum_x * sum_x, 0.0)
    y_variance = np.maximum(count * sum_y2 - sum_y * sum_y, 0.0)
    denominator = np.sqrt(y_variance[:, :, None] * x_variance[None, :, :])
    trial_tuning = np.divide(
        numerator,
        denominator,
        out=trial_tuning,
        where=denominator > 0,
    )
    result = {
        "angles_deg": angles_deg,
        "orientation_tuning": mean_tuning,
        "trial_orientation_tuning": trial_tuning,
        "source": "correlation",
    }
    if n_trials > 1:
        result["orientation_sem"] = np.nanstd(trial_tuning, axis=0, ddof=1) / np.sqrt(n_trials)
    print(
        f"[CORRELATION] Precomputed trial-wise orientation tuning | "
        f"neurons={len(chunk_groups) and sum(len(group) for group in chunk_groups.values()) or 0}/{n_neurons} | "
        f"elapsed={time.perf_counter() - started:.1f}s"
    )
    return result


def orientation_tuning_bundle(spikes, wavelets_complex, rfs, angles_deg=None):
    """Precompute firing-rate and correlation tuning in one wavelet pass.

    Coarse-RF analysis needs two curves for every unit.  Both curves read the
    same preferred ``(x, y, sigma, frequency)`` feature from the same
    disk-backed wavelet cache; doing those reads in separate functions doubles
    the dominant Zarr I/O.  This helper keeps the established public functions
    above intact, but combines their bounded feature-chunk traversal for the
    GUI's normal analysis path.

    The returned mappings have the same schemas and numerical definitions as
    :func:`firing_rate_orientation_tuning` and
    :func:`correlation_orientation_tuning`, respectively.  In particular, the
    correlation mean is still the established RF-correlation value rather than
    a newly calculated value.
    """
    raw_responses = np.asarray(spikes, dtype=float)
    rate_trial_responses = raw_responses if raw_responses.ndim == 3 else None
    if raw_responses.ndim == 3:
        rate_mean_responses = np.nanmean(raw_responses, axis=0)
        correlation_responses = raw_responses
    elif raw_responses.ndim == 2:
        rate_mean_responses = raw_responses
        correlation_responses = raw_responses[None, :, :]
    else:
        raise ValueError(f"Expected spikes with 2 or 3 dimensions, got {raw_responses.shape}")
    rate_mean_responses = np.clip(
        np.nan_to_num(rate_mean_responses, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        None,
    )
    correlation_responses = np.nan_to_num(
        correlation_responses, nan=0.0, posinf=0.0, neginf=0.0
    )

    wavelets = wavelets_complex
    wavelet_shape = tuple(getattr(wavelets, "shape", ()))
    if len(wavelet_shape) not in {5, 6}:
        raise ValueError(f"Expected wavelets with 5 or 6 dimensions, got {wavelet_shape}")
    rf_values = np.asarray(rfs[0])
    maxes = np.asarray(rfs[1], dtype=int)
    n_frames = min(rate_mean_responses.shape[0], correlation_responses.shape[1], wavelet_shape[0])
    n_neurons = min(
        rate_mean_responses.shape[1], correlation_responses.shape[2], maxes.shape[1], rf_values.shape[0]
    )
    n_trials = correlation_responses.shape[0]
    n_orientations = wavelet_shape[3]
    if angles_deg is None:
        angles_deg = np.linspace(0, 180, n_orientations, endpoint=False)
    angles_deg = np.asarray(angles_deg, dtype=float)
    if angles_deg.size != n_orientations:
        raise ValueError("angles_deg must match the coarse wavelet orientation axis")

    chunks = tuple(int(value) for value in (getattr(wavelets, "chunks", None) or wavelet_shape))
    time_chunk = max(1, min(n_frames, chunks[0]))
    x_chunk = max(1, min(wavelet_shape[1], chunks[1]))
    y_chunk = max(1, min(wavelet_shape[2], chunks[2]))
    sigma_chunk = max(1, min(wavelet_shape[4], chunks[4]))
    frequency_chunk = max(1, min(wavelet_shape[5], chunks[5])) if len(wavelet_shape) == 6 else 1

    chunk_groups = {}
    mean_tuning = np.full((n_neurons, n_orientations), np.nan, dtype=float)
    for neuron_idx in range(n_neurons):
        x, y, _orientation, sigma_idx, frequency_idx = maxes[:5, neuron_idx]
        if not (
            0 <= x < wavelet_shape[1]
            and 0 <= y < wavelet_shape[2]
            and 0 <= sigma_idx < wavelet_shape[4]
            and (len(wavelet_shape) == 5 or 0 <= frequency_idx < wavelet_shape[5])
        ):
            continue
        mean_tuning[neuron_idx] = rf_values[neuron_idx, x, y, :, sigma_idx, frequency_idx]
        key = (
            int(x) // x_chunk,
            int(y) // y_chunk,
            int(sigma_idx) // sigma_chunk,
            int(frequency_idx) // frequency_chunk if len(wavelet_shape) == 6 else 0,
        )
        chunk_groups.setdefault(key, []).append(neuron_idx)

    valid_neuron_ids = np.array(
        [neuron_id for neuron_ids in chunk_groups.values() for neuron_id in neuron_ids], dtype=int
    )
    # Firing-rate accumulators (the same definitions as firing_rate_orientation_tuning).
    weighted_sum = np.zeros((n_neurons, n_orientations), dtype=np.float64)
    weight_sum = np.zeros((n_neurons, n_orientations), dtype=np.float64)
    trial_weighted_sum = (
        np.zeros((rate_trial_responses.shape[0], n_neurons, n_orientations), dtype=np.float64)
        if rate_trial_responses is not None
        else None
    )
    # Trial-wise Pearson accumulators (the same definitions as correlation_orientation_tuning).
    sum_x = np.zeros((n_neurons, n_orientations), dtype=np.float64)
    sum_x2 = np.zeros_like(sum_x)
    sum_y = np.zeros((n_trials, n_neurons), dtype=np.float64)
    sum_y2 = np.zeros_like(sum_y)
    sum_xy = np.zeros((n_trials, n_neurons, n_orientations), dtype=np.float64)

    total_groups = len(chunk_groups)
    report_every = max(1, min(10, math.ceil(total_groups / 10)))
    started = time.perf_counter()
    read_seconds = 0.0
    compute_seconds = 0.0
    print(
        f"[ORIENTATION] Starting shared chunk-batched tuning | "
        f"neurons={valid_neuron_ids.size}/{n_neurons} | feature chunks={total_groups} | "
        f"time chunk={time_chunk}"
    )
    for group_number, ((x_group, y_group, sigma_group, frequency_group), neuron_ids) in enumerate(
        chunk_groups.items(), start=1
    ):
        neuron_ids = np.asarray(neuron_ids, dtype=int)
        feature_indices = maxes[:5, neuron_ids]
        x0, y0 = x_group * x_chunk, y_group * y_chunk
        sigma0 = sigma_group * sigma_chunk
        frequency0 = frequency_group * frequency_chunk
        x1 = min(wavelet_shape[1], x0 + x_chunk)
        y1 = min(wavelet_shape[2], y0 + y_chunk)
        sigma1 = min(wavelet_shape[4], sigma0 + sigma_chunk)
        frequency1 = (
            min(wavelet_shape[5], frequency0 + frequency_chunk)
            if len(wavelet_shape) == 6
            else 1
        )
        local_x = feature_indices[0] - x0
        local_y = feature_indices[1] - y0
        local_sigma = feature_indices[3] - sigma0
        local_frequency = feature_indices[4] - frequency0
        for time_start in range(0, n_frames, time_chunk):
            time_end = min(n_frames, time_start + time_chunk)
            read_started = time.perf_counter()
            if len(wavelet_shape) == 5:
                block = np.asarray(wavelets[time_start:time_end, x0:x1, y0:y1, :, sigma0:sigma1])
                raw_features = np.moveaxis(block[:, local_x, local_y, :, local_sigma], 0, 1)
            else:
                block = np.asarray(
                    wavelets[
                        time_start:time_end, x0:x1, y0:y1, :, sigma0:sigma1, frequency0:frequency1
                    ]
                )
                raw_features = np.moveaxis(
                    block[:, local_x, local_y, :, local_sigma, local_frequency], 0, 1
                )
            read_seconds += time.perf_counter() - read_started
            compute_started = time.perf_counter()
            # Preserve the correlation calculation's source dtype, then make a
            # float64/non-negative copy for the weighted firing-rate sums.
            correlation_features = np.nan_to_num(
                raw_features, nan=0.0, posinf=0.0, neginf=0.0
            )
            rate_weights = np.asarray(correlation_features, dtype=np.float64)
            np.maximum(rate_weights, 0.0, out=rate_weights)

            mean_rates = np.take(rate_mean_responses[time_start:time_end], neuron_ids, axis=1)
            weighted_sum[neuron_ids] += np.sum(
                rate_weights * mean_rates[:, :, None], axis=0, dtype=np.float64
            )
            weight_sum[neuron_ids] += rate_weights.sum(axis=0, dtype=np.float64)
            if trial_weighted_sum is not None:
                rate_trials = np.clip(
                    np.nan_to_num(
                        np.take(
                            rate_trial_responses[:, time_start:time_end, :], neuron_ids, axis=2
                        ),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                    0.0,
                    None,
                )
                trial_weighted_sum[:, neuron_ids] += np.sum(
                    rate_weights[None, :, :, :] * rate_trials[:, :, :, None],
                    axis=1,
                    dtype=np.float64,
                )

            correlation_trials = np.take(
                correlation_responses[:, time_start:time_end, :], neuron_ids, axis=2
            )
            sum_x[neuron_ids] += np.sum(correlation_features, axis=0, dtype=np.float64)
            sum_x2[neuron_ids] += np.sum(
                correlation_features * correlation_features, axis=0, dtype=np.float64
            )
            sum_y[:, neuron_ids] += np.sum(correlation_trials, axis=1, dtype=np.float64)
            sum_y2[:, neuron_ids] += np.sum(
                correlation_trials * correlation_trials, axis=1, dtype=np.float64
            )
            sum_xy[:, neuron_ids] += np.einsum(
                "tgo,ntg->ngo", correlation_features, correlation_trials, optimize=True, dtype=np.float64
            )
            compute_seconds += time.perf_counter() - compute_started
        if group_number % report_every == 0 or group_number == total_groups:
            print(
                f"[ORIENTATION] Feature chunks {group_number}/{total_groups} | "
                f"neurons in chunk={neuron_ids.size} | read={read_seconds:.1f}s | "
                f"compute={compute_seconds:.1f}s"
            )

    firing_tuning = np.full((n_neurons, n_orientations), np.nan, dtype=float)
    firing_osi = np.full(n_neurons, np.nan, dtype=float)
    firing_gosi = np.full(n_neurons, np.nan, dtype=float)
    firing_trial_tuning = None
    if rate_trial_responses is not None:
        firing_trial_tuning = np.full(
            (rate_trial_responses.shape[0], n_neurons, n_orientations), np.nan, dtype=float
        )
    if valid_neuron_ids.size:
        firing_tuning[valid_neuron_ids] = np.divide(
            weighted_sum[valid_neuron_ids],
            weight_sum[valid_neuron_ids],
            out=np.zeros((valid_neuron_ids.size, n_orientations), dtype=float),
            where=weight_sum[valid_neuron_ids] > 0,
        )
        if firing_trial_tuning is not None:
            firing_trial_tuning[:, valid_neuron_ids] = np.divide(
                trial_weighted_sum[:, valid_neuron_ids],
                weight_sum[None, valid_neuron_ids],
                out=np.zeros(
                    (rate_trial_responses.shape[0], valid_neuron_ids.size, n_orientations), dtype=float
                ),
                where=weight_sum[None, valid_neuron_ids] > 0,
            )
        rates = np.nan_to_num(firing_tuning[valid_neuron_ids], nan=0.0)
        preferred = np.argmax(rates, axis=1)
        orientation_angles = angles_deg % 180.0
        orthogonal = (orientation_angles[preferred] + 90.0) % 180.0
        distances = np.abs(
            ((orientation_angles[None, :] - orthogonal[:, None] + 90.0) % 180.0) - 90.0
        )
        orthogonal_indices = np.argmin(distances, axis=1)
        preferred_rates = rates[np.arange(rates.shape[0]), preferred]
        orthogonal_rates = rates[np.arange(rates.shape[0]), orthogonal_indices]
        osi_denominator = preferred_rates + orthogonal_rates
        firing_osi[valid_neuron_ids] = np.divide(
            preferred_rates - orthogonal_rates,
            osi_denominator,
            out=np.full(valid_neuron_ids.size, np.nan, dtype=float),
            where=osi_denominator > 0,
        )
        phase = np.exp(2j * np.deg2rad(orientation_angles))
        gosi_denominator = rates.sum(axis=1)
        firing_gosi[valid_neuron_ids] = np.divide(
            np.abs(rates @ phase),
            gosi_denominator,
            out=np.full(valid_neuron_ids.size, np.nan, dtype=float),
            where=gosi_denominator > 0,
        )

    correlation_trial_tuning = np.full((n_trials, n_neurons, n_orientations), np.nan, dtype=float)
    count = float(n_frames)
    numerator = count * sum_xy - sum_y[:, :, None] * sum_x[None, :, :]
    x_variance = np.maximum(count * sum_x2 - sum_x * sum_x, 0.0)
    y_variance = np.maximum(count * sum_y2 - sum_y * sum_y, 0.0)
    denominator = np.sqrt(y_variance[:, :, None] * x_variance[None, :, :])
    correlation_trial_tuning = np.divide(
        numerator,
        denominator,
        out=correlation_trial_tuning,
        where=denominator > 0,
    )

    firing_result = {
        "angles_deg": angles_deg,
        "orientation_tuning": firing_tuning,
        "osi": firing_osi,
        "gosi": firing_gosi,
        "source": "firing_rate",
    }
    if firing_trial_tuning is not None:
        firing_result["trial_orientation_tuning"] = firing_trial_tuning
        if firing_trial_tuning.shape[0] > 1:
            firing_result["orientation_sem"] = np.nanstd(
                firing_trial_tuning, axis=0, ddof=1
            ) / np.sqrt(firing_trial_tuning.shape[0])
    correlation_result = {
        "angles_deg": angles_deg,
        "orientation_tuning": mean_tuning,
        "trial_orientation_tuning": correlation_trial_tuning,
        "source": "correlation",
    }
    if n_trials > 1:
        correlation_result["orientation_sem"] = np.nanstd(
            correlation_trial_tuning, axis=0, ddof=1
        ) / np.sqrt(n_trials)
    print(
        f"[ORIENTATION] Completed shared firing-rate/correlation tuning | "
        f"elapsed={time.perf_counter() - started:.1f}s | chunk reads={read_seconds:.1f}s | "
        f"metrics={compute_seconds:.1f}s"
    )
    return {"firing_rate": firing_result, "correlation": correlation_result}
