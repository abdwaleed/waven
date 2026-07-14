"""Regression tests for optimized convolution scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from waven.storage.array_store import load_array
from waven.wavelets.decomposition import waveletPowerDecompositionConv


def test_time_major_coarse_power_matches_group_major(tmp_path, monkeypatch):
    """Reading a frame chunk once must preserve the coarse-power tensor."""
    pytest.importorskip("zarr")
    generator = np.random.default_rng(11)
    movie = generator.normal(size=(5, 7, 6)).astype(np.float32)
    common = dict(
        videodata=movie,
        sigmas=np.array([1.0, 1.5]),
        folder_path=str(tmp_path),
        n_orientations=2,
        phase_offsets=(0.0, np.pi / 2),
        frame_chunk_size=2,
        filter_group_size=1,
    )
    monkeypatch.setenv("WAVEN_AMP", "0")
    monkeypatch.setenv("WAVEN_TORCH_COMPILE", "0")
    monkeypatch.setenv("WAVEN_MULTI_GPU", "0")
    monkeypatch.setenv("WAVEN_TIME_MAJOR_CONV", "0")
    group_path = waveletPowerDecompositionConv(output_stem="group_major", **common)
    monkeypatch.setenv("WAVEN_TIME_MAJOR_CONV", "1")
    time_path = waveletPowerDecompositionConv(output_stem="time_major", **common)

    group_major = np.asarray(load_array(group_path, mmap_mode="r"))
    time_major = np.asarray(load_array(time_path, mmap_mode="r"))
    np.testing.assert_allclose(time_major, group_major, rtol=2e-6, atol=2e-6)
