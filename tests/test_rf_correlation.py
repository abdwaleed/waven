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
