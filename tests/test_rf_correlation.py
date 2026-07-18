"""Numerical-regression tests for the disk-backed Coarse RF correlation path."""

from __future__ import annotations

import numpy as np

from waven.analysis.rf_correlation import streaming_cross_correlation


def test_structured_correlation_matches_reference_with_large_power_baseline(monkeypatch):
    """Preserve Pearson accuracy when wavelet power has a large DC offset.

    A raw-sum covariance reconstruction is vulnerable to cancellation in this
    case.  The test uses the structured (5-D) path because that is the path
    used by Coarse RF Zarr caches.
    """
    monkeypatch.setenv("WAVEN_RF_GPU", "0")
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


def test_structured_correlation_can_write_a_disk_backed_result(tmp_path, monkeypatch):
    """Large GUI RF tensors remain sliceable without resident-RAM allocation."""
    monkeypatch.setenv("WAVEN_RF_GPU", "0")
    generator = np.random.default_rng(41)
    stimulus = generator.normal(size=(64, 2, 3, 2, 1)).astype(np.float32)
    response = generator.normal(size=(64, 3)).astype(np.float32)
    output_path = tmp_path / "coarse_rf_correlations.npy"

    expected = streaming_cross_correlation(stimulus, response)
    actual = streaming_cross_correlation(stimulus, response, output_path=output_path)

    assert output_path.exists()
    assert isinstance(actual, np.memmap)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
