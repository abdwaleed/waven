"""Regression tests for optimized convolution scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from waven.storage.array_store import load_array
from waven.runtime.task_control import OperationCancelled
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

    # The default layout must match each frame/filter write tile.  Otherwise
    # time-major scheduling repeatedly recompresses a partially updated Zarr
    # chunk and can be slower than simply rereading the movie per group.
    import zarr

    time_major_zarr = zarr.open(time_path, mode="r")
    assert time_major_zarr.chunks[0] == common["frame_chunk_size"]
    assert time_major_zarr.chunks[-1] == common["filter_group_size"]
    assert not (tmp_path / "time_major.zarr.waven-progress.json").exists()


def test_direct_coarse_power_resumes_completed_filter_groups(tmp_path, monkeypatch, capsys):
    """Cancellation keeps completed direct-RF work without exposing it as final."""
    pytest.importorskip("zarr")

    class CancelBeforeSecondGroup:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 4

    common = dict(
        videodata=np.random.default_rng(12).normal(size=(5, 7, 6)).astype(np.float32),
        sigmas=np.array([1.0, 1.5]),
        folder_path=str(tmp_path),
        n_orientations=2,
        phase_offsets=(0.0, np.pi / 2),
        frame_chunk_size=2,
        filter_group_size=1,
        output_stem="resumable",
    )
    monkeypatch.setenv("WAVEN_AMP", "0")
    monkeypatch.setenv("WAVEN_TORCH_COMPILE", "0")
    monkeypatch.setenv("WAVEN_MULTI_GPU", "0")
    monkeypatch.setenv("WAVEN_TIME_MAJOR_CONV", "0")
    monkeypatch.setenv("WAVEN_ASYNC_WRITER", "0")
    monkeypatch.setenv("WAVEN_AUTOTUNE", "0")
    monkeypatch.setenv("WAVEN_PREFETCH", "0")

    with pytest.raises(OperationCancelled):
        waveletPowerDecompositionConv(cancel_event=CancelBeforeSecondGroup(), **common)

    progress_path = tmp_path / "resumable.zarr.waven-progress.json"
    assert progress_path.exists()
    output_path = waveletPowerDecompositionConv(**common)
    assert "skipping completed sigma group 1/2" in capsys.readouterr().out
    assert not progress_path.exists()
    assert np.asarray(load_array(output_path, mmap_mode="r")).shape == (5, 6, 7, 2, 2)
