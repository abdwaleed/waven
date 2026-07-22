"""Regression coverage for configurable ephys digital-input selection."""

from __future__ import annotations

import numpy as np
import pytest

from waven import time_alignment as ta
from waven.config import PipelineConfig, WORKFLOW_EPHYS
from waven.suite_ephys import DIO


def test_dio_discovery_uses_din_files_not_directory_names(tmp_path):
    """Arbitrarily named and split exports are ordered by their part number."""
    part_two = tmp_path / "recording.part2" / "digital_input"
    part_one = tmp_path / "a_folder_without_a_dio_suffix"
    part_two.mkdir(parents=True)
    part_one.mkdir()
    (part_one / "board_Din7.dat").touch()
    (part_one / "board_Din3.dat").touch()
    (part_two / "board_Din3.dat").touch()
    (tmp_path / "unrelated_dat_files").mkdir()
    (tmp_path / "unrelated_dat_files" / "notes.dat").touch()

    assert DIO.available_dio_ports(tmp_path) == (3, 7)
    assert DIO.get_dio_folders(tmp_path, channel_id=3) == [part_one, part_two]

    with pytest.raises(FileNotFoundError, match=r"Din8\.dat.*Available ports: 3, 7"):
        DIO.get_dio_folders(tmp_path, channel_id=8)


def test_ephys_port_round_trips_through_typed_configuration():
    config = PipelineConfig.from_mappings(
        workflow=WORKFLOW_EPHYS,
        analysis_params={"Photodiode Port": "7"},
    )

    assert config.analysis.photodiode_port == 7
    assert config.analysis.to_gui_mapping()["Photodiode Port"] == "7"


def test_ephys_dispatch_forwards_the_selected_photodiode_port(monkeypatch, tmp_path):
    received = {}

    def no_cache(*args, **kwargs):
        raise FileNotFoundError

    def align(*args, **kwargs):
        received.update(kwargs)
        return ta.AlignedNeuralData(
            spikes=np.empty((0, 0)),
            neuron_pos=np.empty((0, 2)),
        )

    monkeypatch.setattr(ta, "load_neural_cache_pair", no_cache)
    monkeypatch.setattr(ta, "align_ephys_data", align)

    ta.load_aligned_spikes(
        WORKFLOW_EPHYS,
        experiment_info=("subject", "2024-07-23", 1),
        data_dir=tmp_path,
        data_dir_strings=[str(tmp_path)],
        suite2p_dir=tmp_path,
        block_end=0,
        n_planes=None,
        nb_frames=10,
        resolution=None,
        sampling_rate=30_000,
        photodiode_port=7,
        save_dir=tmp_path,
        stimulus_duration=1.5,
    )

    assert received["photodiode_port"] == 7


def test_ephys_dispatch_requires_a_selected_photodiode_port(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ta,
        "load_neural_cache_pair",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError),
    )

    with pytest.raises(ValueError, match="selected Photodiode Port"):
        ta.load_aligned_spikes(
            WORKFLOW_EPHYS,
            experiment_info=("subject", "2024-07-23", 1),
            data_dir=tmp_path,
            data_dir_strings=[str(tmp_path)],
            suite2p_dir=tmp_path,
            block_end=0,
            n_planes=None,
            nb_frames=10,
            resolution=None,
            sampling_rate=30_000,
            save_dir=tmp_path,
            stimulus_duration=1.5,
        )
