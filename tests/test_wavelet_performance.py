"""Regression tests for optimized convolution scheduling."""

from __future__ import annotations

import numpy as np
import pytest

from waven.storage.array_store import load_array
from waven.runtime.task_control import OperationCancelled
from waven.wavelets import decomposition
from waven.wavelets.decomposition import (
    _coarse_power_zarr_layout,
    _full_convolution_zarr_layout,
    waveletPowerDecompositionConv,
)


def test_coarse_power_layout_is_write_and_rf_read_aligned():
    """The same time/sigma tiles must serve writing and RF tile reads."""
    shape = (18_000, 137, 77, 18, 4)
    chunks = _coarse_power_zarr_layout(shape, frame_chunk_size=1024, filter_group_size=2)

    assert chunks[0] == 1024
    assert chunks[3] == shape[3]
    assert chunks[4] == 2
    assert 1 <= chunks[1] <= shape[1]
    assert 1 <= chunks[2] <= shape[2]
    # The old fixed 16 x 16 grid needed 45 RF spatial reads.  This layout
    # deliberately uses fewer, still bounded tiles for the real workload.
    old_tiles = int(np.ceil(shape[1] / 16) * np.ceil(shape[2] / 16))
    new_tiles = int(np.ceil(shape[1] / chunks[1]) * np.ceil(shape[2] / chunks[2]))
    assert new_tiles < old_tiles


def test_convolution_resource_plan_scales_with_all_live_filter_buffers(monkeypatch):
    """A high-dimensional bank must reduce batches/groups before CUDA OOM."""
    monkeypatch.setenv("WAVEN_AMP", "0")
    monkeypatch.setattr(decomposition, "gpu_available_vram_bytes", lambda: 8 * 1024**3)

    small_bank = decomposition._conv_frame_chunk_size(
        18_000, 137, 77,
        filter_count=16,
        output_channels=8,
        device="cuda:0",
        kernel_shape=(31, 31),
        output_buffer_count=2,
    )
    large_bank = decomposition._conv_frame_chunk_size(
        18_000, 137, 77,
        filter_count=320,
        output_channels=160,
        device="cuda:0",
        kernel_shape=(91, 91),
        output_buffer_count=2,
    )
    small_group = decomposition._conv_filter_group_size(
        128, 137, 77, 8, "cuda:0",
        filter_channels=16,
        output_channels=8,
        output_buffer_count=2,
    )
    large_group = decomposition._conv_filter_group_size(
        128, 137, 77, 160, "cuda:0",
        filter_channels=320,
        output_channels=160,
        output_buffer_count=2,
    )

    assert large_bank < small_bank
    assert large_group < small_group


def test_full_convolution_layout_avoids_one_pixel_zarr_writes():
    """Frequency/sigma banks write spatial tiles rather than thousands of pixels."""
    shape = (18_000, 137, 77, 80, 4, 6)
    chunks = _full_convolution_zarr_layout(shape, frame_chunk_size=128, filter_group_size=3)

    assert chunks[0] == 128
    assert chunks[3:] == (80, 1, 1)
    assert chunks[1] * chunks[2] > 1


def test_time_major_coarse_power_matches_group_major(tmp_path, monkeypatch, capsys):
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
    # A configured multi-GPU mode may fall back to one usable device.  It must
    # not disable time-major scheduling, which avoids rereading the movie.
    monkeypatch.setenv("WAVEN_MULTI_GPU", "1")
    monkeypatch.setenv("WAVEN_TIME_MAJOR_CONV", "1")
    time_path = waveletPowerDecompositionConv(output_stem="time_major", **common)
    assert "using time-major filter scheduling" in capsys.readouterr().out

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


def test_direct_coarse_power_resume_keeps_the_persisted_time_chunk(tmp_path, monkeypatch, capsys):
    """A changed VRAM estimate must not cause partial Zarr chunk rewrites."""
    pytest.importorskip("zarr")

    class CancelBeforeSecondGroup:
        def __init__(self):
            self.calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 4

    common = dict(
        videodata=np.random.default_rng(31).normal(size=(5, 7, 6)).astype(np.float32),
        sigmas=np.array([1.0, 1.5]),
        folder_path=str(tmp_path),
        n_orientations=2,
        phase_offsets=(0.0, np.pi / 2),
        filter_group_size=1,
        output_stem="layout_resume",
    )
    monkeypatch.setenv("WAVEN_AMP", "0")
    monkeypatch.setenv("WAVEN_TORCH_COMPILE", "0")
    monkeypatch.setenv("WAVEN_MULTI_GPU", "0")
    monkeypatch.setenv("WAVEN_TIME_MAJOR_CONV", "0")
    monkeypatch.setenv("WAVEN_ASYNC_WRITER", "0")
    monkeypatch.setenv("WAVEN_AUTOTUNE", "0")
    monkeypatch.setenv("WAVEN_PREFETCH", "0")

    with pytest.raises(OperationCancelled):
        waveletPowerDecompositionConv(
            frame_chunk_size=2, cancel_event=CancelBeforeSecondGroup(), **common
        )

    output_path = waveletPowerDecompositionConv(frame_chunk_size=3, **common)
    output = load_array(output_path, mmap_mode="r")
    assert output.chunks[0] == 2
    assert "retaining interrupted cache time chunk 2 instead of newly tuned 3" in capsys.readouterr().out
