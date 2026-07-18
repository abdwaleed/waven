"""Tests for orientation-axis display and selectivity invariants."""

import numpy as np
import pytest

from waven.analysis.orientation_selectivity import (
    close_orientation_curve,
    correlation_orientation_tuning,
    firing_rate_orientation_tuning,
    orientation_selectivity_from_tuning,
    orientation_tuning_bundle,
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


def test_firing_rate_tuning_reads_each_storage_chunk_once_per_time_tile():
    """Preferred features in one Zarr chunk must be read together."""

    class CountingChunkedArray:
        def __init__(self, values, chunks):
            self.values = values
            self.shape = values.shape
            self.chunks = chunks
            self.read_count = 0

        def __getitem__(self, item):
            self.read_count += 1
            return self.values[item]

    spikes = np.arange(2 * 6 * 4, dtype=float).reshape(2, 6, 4) % 7
    wavelets = (
        np.arange(6 * 8 * 8 * 3 * 2, dtype=float).reshape(6, 8, 8, 3, 2) % 5
    ) + 1
    # Neurons 0/1 share one (x, y, sigma) storage chunk; neurons 2/3 share a
    # second.  With two time tiles, the optimized implementation makes four
    # reads, instead of independently reading a tile for every neuron.
    chunked_wavelets = CountingChunkedArray(wavelets, chunks=(3, 4, 4, 3, 2))
    indices = np.array(
        [
            [0, 1, 5, 6],
            [0, 2, 5, 7],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )

    expected_tuning, expected_trial_tuning = _reference_firing_rate_tuning(
        spikes, wavelets, (None, indices)
    )
    result = firing_rate_orientation_tuning(spikes, chunked_wavelets, (None, indices))

    assert chunked_wavelets.read_count == 4
    np.testing.assert_allclose(result["orientation_tuning"], expected_tuning, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        result["trial_orientation_tuning"], expected_trial_tuning, rtol=1e-12, atol=1e-12
    )


def test_correlation_tuning_precomputes_trial_curves_without_changing_rf_means():
    wavelets = np.zeros((4, 1, 1, 2, 1), dtype=float)
    wavelets[:, 0, 0, 0, 0] = [0, 1, 2, 3]
    wavelets[:, 0, 0, 1, 0] = [3, 2, 1, 0]
    spikes = np.array(
        [
            [[0.0], [1.0], [2.0], [3.0]],
            [[3.0], [2.0], [1.0], [0.0]],
        ]
    )
    # The mean orientation curve is the pre-existing RF value, while the new
    # trial values are Pearson r estimates ready for export.
    rf_values = np.array([[[[[[0.25]], [[-0.25]]]]]])
    preferred_indices = np.array([[0], [0], [0], [0], [0]])

    result = correlation_orientation_tuning(
        spikes, wavelets, (rf_values, preferred_indices), angles_deg=[0, 90]
    )

    np.testing.assert_allclose(result["orientation_tuning"], [[0.25, -0.25]])
    np.testing.assert_allclose(
        result["trial_orientation_tuning"][:, 0, :], [[1.0, -1.0], [-1.0, 1.0]]
    )


def test_shared_orientation_bundle_matches_separate_curves_with_one_set_of_reads():
    """The GUI path must not reread identical preferred-feature Zarr tiles."""

    class CountingChunkedArray:
        def __init__(self, values, chunks):
            self.values = values
            self.shape = values.shape
            self.chunks = chunks
            self.read_count = 0

        def __getitem__(self, item):
            self.read_count += 1
            return self.values[item]

    generator = np.random.default_rng(42)
    spikes = generator.normal(size=(3, 6, 4))
    wavelets = generator.random(size=(6, 8, 8, 3, 2))
    indices = np.array(
        [
            [0, 1, 5, 6],
            [0, 2, 5, 7],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ]
    )
    rf_values = generator.normal(size=(4, 8, 8, 3, 2, 1))
    expected_firing = firing_rate_orientation_tuning(spikes, wavelets, (rf_values, indices))
    expected_correlation = correlation_orientation_tuning(spikes, wavelets, (rf_values, indices))

    chunked_wavelets = CountingChunkedArray(wavelets, chunks=(3, 4, 4, 3, 2))
    result = orientation_tuning_bundle(spikes, chunked_wavelets, (rf_values, indices))

    # Two spatial feature groups x two time chunks.  Calling the established
    # functions separately would make eight reads; the shared path makes four.
    assert chunked_wavelets.read_count == 4
    for key in ("orientation_tuning", "trial_orientation_tuning", "osi", "gosi"):
        np.testing.assert_allclose(
            result["firing_rate"][key], expected_firing[key], rtol=1e-12, atol=1e-12, equal_nan=True
        )
    for key in ("orientation_tuning", "trial_orientation_tuning"):
        np.testing.assert_allclose(
            result["correlation"][key], expected_correlation[key], rtol=1e-12, atol=1e-12, equal_nan=True
        )


def test_shared_orientation_bundle_reuses_compact_selected_feature_cache(tmp_path):
    """A later run must avoid the large spatial wavelet source without changing curves."""
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    generator = np.random.default_rng(81)
    spikes = generator.normal(size=(2, 8, 3))
    wavelets = generator.random(size=(8, 4, 4, 5, 2)).astype(np.float32)
    indices = np.array(
        [
            [0, 1, 3],
            [1, 2, 0],
            [0, 0, 0],
            [0, 1, 1],
            [0, 0, 0],
        ]
    )
    rf_values = generator.normal(size=(3, 4, 4, 5, 2, 1))
    cache_path = tmp_path / "selected_orientation_features.zarr"

    first = orientation_tuning_bundle(
        spikes,
        wavelets,
        (rf_values, indices),
        tuning_cache_path=cache_path,
    )
    second = orientation_tuning_bundle(
        spikes,
        wavelets,
        (rf_values, indices),
        tuning_cache_path=cache_path,
    )

    assert cache_path.is_dir()
    for curve_kind in ("firing_rate", "correlation"):
        for key in ("orientation_tuning", "trial_orientation_tuning"):
            np.testing.assert_allclose(
                second[curve_kind][key],
                first[curve_kind][key],
                rtol=0.0,
                atol=0.0,
                equal_nan=True,
            )
