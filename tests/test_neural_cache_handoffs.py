"""Regression coverage for neural-cache metadata and Zarr trial selection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from waven import time_alignment as ta
from waven.data import neural as neural_io
from waven.storage.array_store import read_first_axis_indices
from waven.storage.neural_cache import (
    load_unit_ids,
    load_unit_info,
    save_aligned_neural_cache,
)


def test_read_first_axis_indices_matches_numpy_advanced_selection():
    source = np.arange(4 * 5 * 3).reshape(4, 5, 3)

    selected = read_first_axis_indices(source, [3, 1], slice(1, 4), slice(1, 3))

    np.testing.assert_array_equal(selected, source[[3, 1], 1:4, 1:3])


def test_read_first_axis_indices_has_basic_indexing_fallback():
    source_values = np.arange(4 * 5 * 3).reshape(4, 5, 3)

    class BasicIndexOnlyArray:
        def __getitem__(self, selection):
            if isinstance(selection[0], np.ndarray):
                raise IndexError("advanced first-axis indexing is unsupported")
            return source_values[selection]

    selected = read_first_axis_indices(
        BasicIndexOnlyArray(),
        [3, 1],
        slice(1, 4),
        slice(1, 3),
    )

    np.testing.assert_array_equal(selected, source_values[[3, 1], 1:4, 1:3])


def test_read_first_axis_indices_supports_disk_backed_zarr(tmp_path):
    zarr = pytest.importorskip("zarr")
    source_values = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    source = zarr.open(
        str(tmp_path / "spikes.zarr"),
        mode="w",
        shape=source_values.shape,
        chunks=(1, 3, 3),
        dtype=source_values.dtype,
    )
    source[:] = source_values

    selected = read_first_axis_indices(source, [3, 1], slice(1, 4), slice(1, 3))

    np.testing.assert_array_equal(selected, source_values[[3, 1], 1:4, 1:3])


def test_stale_optional_unit_sidecars_do_not_block_analysis(tmp_path):
    (tmp_path / "unit_ids.json").write_text(json.dumps(["old-a", "old-b"]), encoding="utf-8")
    (tmp_path / "unit_info.json").write_text(
        json.dumps({"version": 1, "units": [{"unit": "old"}]}),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Ignoring stale unit_ids"):
        assert load_unit_ids(tmp_path, n_neurons=3) is None
    with pytest.warns(RuntimeWarning, match="Ignoring stale unit_info"):
        assert load_unit_info(tmp_path, n_neurons=3) is None


def test_saving_without_optional_metadata_clears_old_sidecars(tmp_path):
    positions = np.zeros((2, 3), dtype=np.float32)
    spikes = np.zeros((2, 4, 2), dtype=np.float32)
    save_aligned_neural_cache(
        positions,
        spikes,
        tmp_path,
        unit_ids=["unit-a", "unit-b"],
        unit_info=[{"unit": "unit-a"}, {"unit": "unit-b"}],
    )

    save_aligned_neural_cache(positions, spikes, tmp_path)

    assert not (tmp_path / "unit_ids.json").exists()
    assert not (tmp_path / "unit_info.json").exists()


def test_two_photon_dispatch_accepts_shared_stimulus_duration(monkeypatch, tmp_path):
    """The GUI supplies duration for both workflows through one dispatcher."""
    received = {}

    def no_cache(*args, **kwargs):
        raise FileNotFoundError

    def load_two_photon(*args, **kwargs):
        received.update(kwargs)
        return ta.AlignedNeuralData(
            spikes=np.empty((0, 0)),
            neuron_pos=np.empty((0, 2)),
        )

    monkeypatch.setattr(ta, "load_neural_cache_pair", no_cache)
    monkeypatch.setattr(ta, "load_two_photon_spikes", load_two_photon)

    ta.load_aligned_spikes(
        ta.WORKFLOW_2P,
        experiment_info=("subject", "2024-07-23", 1),
        data_dir=tmp_path,
        data_dir_strings=[str(tmp_path)],
        suite2p_dir=tmp_path,
        block_end=1,
        n_planes=1,
        nb_frames=10,
        resolution=1.0,
        sampling_rate=None,
        save_dir=tmp_path,
        stimulus_duration=1.5,
    )

    assert received["stimulus_duration"] == 1.5


def test_two_photon_loader_accepts_shared_stimulus_duration(monkeypatch, tmp_path):
    """Two-photon loading must accept the GUI's shared duration argument."""
    expected_spikes = np.empty((2, 3))
    expected_positions = np.empty((2, 2))
    monkeypatch.setattr(
        ta,
        "load_neural_cache_pair",
        lambda *args, **kwargs: (expected_spikes, expected_positions, None, None),
    )

    actual = ta.load_two_photon_spikes(
        ("subject", "2024-07-23", 1),
        [str(tmp_path)],
        str(tmp_path),
        block_end=0,
        n_planes=1,
        nb_frames=10,
        resolution=1.0,
        spks_path=tmp_path / "spikes.npy",
        stimulus_duration=1.5,
    )

    assert actual.spikes is expected_spikes
    assert actual.neuron_pos is expected_positions


def _write_suite2p_plane(root, plane=0):
    plane_dir = root / f"plane{plane}"
    plane_dir.mkdir(parents=True)
    np.save(plane_dir / "spks.npy", np.zeros((2, 5), dtype=np.float32))
    np.save(plane_dir / "iscell.npy", np.array([[1.0, 0.9], [0.0, 0.1]]))
    np.save(
        plane_dir / "stat.npy",
        np.array([{"med": np.array([1.0, 2.0])}, {"med": np.array([3.0, 4.0])}], dtype=object),
    )


def _write_timeline_marker(root):
    timeline = root / "subject" / "2024-07-23" / "1" / "2024-07-23_1_subject_Timeline.mat"
    timeline.parent.mkdir(parents=True)
    timeline.touch()
    return timeline


def test_suite2p_preflight_requires_completed_output(tmp_path, monkeypatch):
    monkeypatch.delenv("WAVEN_SUITE2P_OUTPUT_DIR", raising=False)

    with pytest.raises(FileNotFoundError, match="completed Suite2p output"):
        neural_io.validate_suite2p_output(tmp_path / "raw_tiffs", n_planes=1)


def test_suite2p_preflight_accepts_cortex_lab_date_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("WAVEN_SUITE2P_OUTPUT_DIR", raising=False)
    legacy_suite2p = tmp_path / "subject" / "2024-07-23" / "suite2p"
    _write_suite2p_plane(legacy_suite2p)
    gui_default = tmp_path / "subject" / "2024-07-23" / "1" / "suite2p"

    assert neural_io.validate_suite2p_output(gui_default, n_planes=1) == legacy_suite2p


def test_timeline_preflight_uses_configured_external_root(tmp_path, monkeypatch):
    timeline_root = tmp_path / "timeline_root"
    timeline = _write_timeline_marker(timeline_root)
    monkeypatch.setenv("WAVEN_SUBJECT_DIRS", str(timeline_root))

    assert neural_io.validate_two_photon_timeline(
        ("subject", "2024-07-23", 1),
        [],
    ) == str(timeline)


def test_two_photon_loader_validates_and_uses_suite2p_output(tmp_path, monkeypatch):
    monkeypatch.delenv("WAVEN_SUITE2P_OUTPUT_DIR", raising=False)
    suite2p_root = tmp_path / "suite2p"
    _write_suite2p_plane(suite2p_root)
    _write_timeline_marker(tmp_path)
    received = {}

    def load_mesoscope(*args, **kwargs):
        received["suite2p_dir"] = args[2]
        return np.zeros((1, 3, 1)), None, np.zeros((1, 2))

    monkeypatch.setattr(neural_io, "loadSPKMesoscope", load_mesoscope)
    actual = ta.load_two_photon_spikes(
        ("subject", "2024-07-23", 1),
        [str(tmp_path)],
        str(suite2p_root),
        block_end=0,
        n_planes=1,
        nb_frames=3,
        resolution=1.0,
        save_dir=tmp_path / "cache",
    )

    assert received["suite2p_dir"] == str(suite2p_root)
    assert actual.spikes.shape == (1, 3, 1)
