"""Numerical-regression tests for the disk-backed Coarse RF correlation path."""

from __future__ import annotations

import numpy as np
import pytest

from waven.analysis.rf_correlation import (
    _gpu_cross_feature_batch_size,
    _gpu_full_cross_fits,
    _gpu_full_cross_peak_bytes,
    streaming_cross_correlation,
)


def test_gpu_rf_subtile_planner_downshifts_an_oversized_tile():
    """An oversized RF tile should retain GPU work as bounded feature subtiles."""
    batch = _gpu_cross_feature_batch_size(
        tile_features=1_000_000,
        n_neurons=230,
        time_chunk=1_024,
        free_vram_bytes=4 * 1024**3,
    )

    assert 0 < batch < 1_000_000
    # The planner's accounting must include the time-by-feature transfer as
    # well as the float64 cross-product/workspace reserve.
    assert batch * 8 * (230 * 3 + 1_024) <= int(4 * 1024**3 * 0.28)


def test_gpu_rf_full_tile_estimate_uses_live_accumulator_and_current_input_only():
    """A useful full tile should not be rejected by a triple-accumulator guess."""
    peak = _gpu_full_cross_peak_bytes(
        tile_features=691_200,
        n_neurons=230,
        time_chunk=170,
    )
    assert peak == 691_200 * 8 * (230 * 2 + 170)
    assert _gpu_full_cross_fits(691_200, 230, 170, 7 * 1024**3)
    assert not _gpu_full_cross_fits(691_200, 230, 170, 4 * 1024**3)


def test_structured_correlation_matches_reference_with_large_power_baseline(monkeypatch):
    """Preserve Pearson accuracy when wavelet power has a large DC offset.

    A raw-sum covariance reconstruction is vulnerable to cancellation in this
    case.  The test uses the structured (5-D) path because that is the path
    used by Coarse RF Zarr caches.
    """
    generator = np.random.default_rng(19)
    n_frames = 2048
    stimulus = (
        1_000_000.0
        + generator.normal(0.0, 1.0, size=(n_frames, 2, 2, 3, 2))
    ).astype(np.float32)
    response = (
        100.0 + generator.normal(0.0, 1.0, size=(n_frames, 4))
    ).astype(np.float32)

    actual = streaming_cross_correlation(stimulus, response)
    flat_stimulus = stimulus.reshape(n_frames, -1).astype(np.float64)
    reference = np.corrcoef(
        np.concatenate((flat_stimulus, response.astype(np.float64)), axis=1),
        rowvar=False,
    )[-response.shape[1] :, : flat_stimulus.shape[1]]

    np.testing.assert_allclose(actual, reference, rtol=0.0, atol=2e-6)
    assert np.isfinite(actual).all()
    assert np.abs(actual).max() <= 1.0

    # Prefetch changes only I/O scheduling; the stable covariance result must
    # remain bit-for-bit identical to the synchronous read path.
    monkeypatch.setenv("WAVEN_RF_PREFETCH", "0")
    synchronous = streaming_cross_correlation(stimulus, response)
    np.testing.assert_allclose(actual, synchronous, rtol=0.0, atol=0.0)


def test_structured_correlation_can_write_a_disk_backed_result(tmp_path):
    """Large GUI RF tensors remain sliceable without resident-RAM allocation."""
    generator = np.random.default_rng(41)
    stimulus = generator.normal(size=(64, 2, 3, 2, 1)).astype(np.float32)
    response = generator.normal(size=(64, 3)).astype(np.float32)
    output_path = tmp_path / "coarse_rf_correlations.npy"

    expected = streaming_cross_correlation(stimulus, response)
    actual = streaming_cross_correlation(stimulus, response, output_path=output_path)

    assert output_path.exists()
    assert isinstance(actual, np.memmap)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_structured_correlation_can_write_selected_zarr_layout(tmp_path):
    """Zarr RF output remains disk-backed in its x/y/feature layout."""
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    generator = np.random.default_rng(47)
    stimulus = generator.normal(size=(64, 2, 3, 2, 1)).astype(np.float32)
    response = generator.normal(size=(64, 3)).astype(np.float32)
    output_path = tmp_path / "coarse_rf_correlations.zarr"

    expected = streaming_cross_correlation(stimulus, response)
    actual = streaming_cross_correlation(
        stimulus,
        response,
        output_path=output_path,
        structured_output=True,
        output_feature_shape=(2, 3, 2, 1, 1),
    )

    assert output_path.is_dir()
    assert tuple(actual.shape) == (3, 2, 3, 2, 1, 1)
    np.testing.assert_allclose(
        np.asarray(actual).reshape(expected.shape),
        expected,
        rtol=0.0,
        atol=0.0,
    )
