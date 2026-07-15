"""CPU-only regression coverage for formerly CUDA-only legacy helpers."""

from __future__ import annotations

import numpy as np
import torch

from waven.wavelets import decomposition


def test_legacy_wavelet_transform_runs_without_cuda(monkeypatch):
    monkeypatch.setattr(decomposition, "resolve_compute_device", lambda prefer_gpu=True: "cpu")
    frame = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    library = torch.ones((2, 1, 1, 2, 3), dtype=torch.float32)

    result = decomposition.waveletTransform(frame, 0, library)

    np.testing.assert_allclose(result, 6.0)
