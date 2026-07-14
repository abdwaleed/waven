"""CPU-only regression coverage for formerly CUDA-only legacy helpers."""

from __future__ import annotations

import numpy as np
import torch

from waven.analysis import nonlinear_models
from waven.wavelets import decomposition


def test_legacy_spike_triggered_helper_runs_without_cuda(monkeypatch):
    monkeypatch.setattr(nonlinear_models, "resolve_compute_device", lambda prefer_gpu=True: "cpu")
    spikes = np.array([0.0, 1.0, 0.0, 2.0, 1.0], dtype=np.float32)
    features = np.arange(5, dtype=np.float32).reshape(-1, 1)

    result = nonlinear_models.compute_sta(spikes, features, ran=2, nx=1, ny=1, n_orientations=1)

    assert result.shape == (2, 1, 1, 1)
    assert np.isfinite(result).all()


def test_legacy_wavelet_transform_runs_without_cuda(monkeypatch):
    monkeypatch.setattr(decomposition, "resolve_compute_device", lambda prefer_gpu=True: "cpu")
    frame = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    library = torch.ones((2, 1, 1, 2, 3), dtype=torch.float32)

    result = decomposition.waveletTransform(frame, 0, library)

    np.testing.assert_allclose(result, 6.0)
