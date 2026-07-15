"""Tests for the standard Coarse-RF PSTH-weighted STA backend."""

import numpy as np

from waven.analysis.psth_sta import compute_psth_sta, lag_frames_within_window


def test_lag_window_never_exceeds_300_ms():
    np.testing.assert_array_equal(lag_frames_within_window(30.0, 99), np.arange(10))
    np.testing.assert_array_equal(lag_frames_within_window(60.0, 4), np.arange(5))


def test_psth_sta_uses_preceding_stimulus_frames_with_vectorized_weighting():
    movie = np.arange(6 * 2 * 2, dtype=np.float32).reshape(6, 2, 2)
    psth = np.array([0, 0, 1, 2, 0, 0], dtype=np.float32)

    result = compute_psth_sta(movie, psth, fps=10.0, max_lag=3)

    expected_lag_zero = (movie[2] + 2 * movie[3]) / 3
    expected_lag_two = (movie[0] + 2 * movie[1]) / 3
    np.testing.assert_allclose(result.maps[0], expected_lag_zero)
    np.testing.assert_allclose(result.maps[2], expected_lag_two)
    np.testing.assert_allclose(result.lag_ms, [0, 100, 200, 300])


def test_peak_lag_tracks_largest_spatial_variance():
    movie = np.zeros((5, 2, 2), dtype=np.float32)
    movie[0] = [[-2, 2], [-2, 2]]
    movie[1:] = 1
    psth = np.array([0, 0, 1, 0, 0], dtype=np.float32)

    result = compute_psth_sta(movie, psth, fps=10.0, max_lag=4)

    assert result.peak_lag_frame == 2
    assert result.peak_lag_ms == 200.0
