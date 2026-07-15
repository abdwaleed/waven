"""Tests for orientation-axis display and selectivity invariants."""

import numpy as np

from waven.analysis.orientation_selectivity import (
    close_orientation_curve,
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
