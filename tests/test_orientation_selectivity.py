"""Tests for orientation-axis display and selectivity invariants."""

import numpy as np

from waven.analysis.orientation_selectivity import (
    close_orientation_curve,
    firing_rate_orientation_tuning,
    orientation_selectivity_from_tuning,
)


def test_closed_orientation_curve_preserves_periodicity_without_metric_change():
    angles = np.array([90.0, 0.0, 135.0, 45.0])
    rates = np.array([2.0, 5.0, 1.0, 3.0])

    plot_angles, plot_rates = close_orientation_curve(angles, rates)

    assert np.array_equal(plot_angles, np.array([0.0, 45.0, 90.0, 135.0, 180.0]))
    assert plot_rates[0] == plot_rates[-1] == 5.0
    raw_metrics = orientation_selectivity_from_tuning(rates, angles)
    closed_metrics = orientation_selectivity_from_tuning(plot_rates, plot_angles)
    assert np.allclose(raw_metrics, closed_metrics, equal_nan=True)


def test_closed_orientation_curve_rejects_misaligned_values():
    try:
        close_orientation_curve([0.0, 90.0], [1.0])
    except ValueError as exc:
        assert "same length" in str(exc)
    else:  # pragma: no cover - makes the intended failure explicit
        raise AssertionError("Mismatched orientation inputs must be rejected")


def _reference_firing_rate_tuning(spikes, wavelets, rfs):
    """Pre-vectorization implementation retained as a numerical oracle."""
    responses = np.clip(np.nan_to_num(np.mean(spikes, axis=0), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    trial_tuning = np.full((spikes.shape[0], spikes.shape[2], wavelets.shape[3]), np.nan)
    tuning = np.full((spikes.shape[2], wavelets.shape[3]), np.nan)
    for neuron_index, (x, y, _orientation, sigma, _frequency) in enumerate(rfs[1][:5].T):
        weights = np.clip(np.nan_to_num(wavelets[:, x, y, :, sigma], nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        denominator = weights.sum(axis=0)
        tuning[neuron_index] = np.divide(
            np.sum(weights * responses[:, neuron_index, None], axis=0),
            denominator,
            out=np.zeros(wavelets.shape[3]),
            where=denominator > 0,
        )
        trial_rates = np.clip(np.nan_to_num(spikes[:, :, neuron_index], nan=0.0), 0.0, None)
        trial_tuning[:, neuron_index] = np.divide(
            np.sum(weights[None, :, :] * trial_rates[:, :, None], axis=1),
            denominator[None, :],
            out=np.zeros((spikes.shape[0], wavelets.shape[3])),
            where=denominator[None, :] > 0,
        )
    return tuning, trial_tuning


def test_grouped_firing_rate_tuning_matches_the_per_neuron_calculation():
    # Deliberately assign several neurons to the same feature group.  BLAS may
    # sum floating-point products in a different order, so compare at a much
    # tighter tolerance than the displayed metric precision.
    spikes = np.arange(3 * 7 * 4, dtype=float).reshape(3, 7, 4) % 11
    wavelets = (np.arange(7 * 2 * 2 * 3 * 2, dtype=float).reshape(7, 2, 2, 3, 2) % 5) + 1
    indices = np.array(
        [
            [0, 0, 1, 1],  # x
            [1, 1, 0, 0],  # y
            [0, 1, 2, 0],  # preferred orientation (not used for weights)
            [0, 0, 1, 1],  # sigma
            [0, 0, 0, 0],  # frequency
        ]
    )
    expected_tuning, expected_trial_tuning = _reference_firing_rate_tuning(
        spikes, wavelets, (None, indices)
    )

    result = firing_rate_orientation_tuning(spikes, wavelets, (None, indices))

    np.testing.assert_allclose(
        result["orientation_tuning"], expected_tuning, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        result["trial_orientation_tuning"], expected_trial_tuning, rtol=1e-12, atol=1e-12
    )
