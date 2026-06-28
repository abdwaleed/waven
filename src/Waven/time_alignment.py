"""Time-alignment entry points for two-photon and electrophysiology workflows.

Two-photon alignment delegates to the existing Cortex Lab / suite2p pipeline.
Electrophysiology alignment is provided as a template for user implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

WORKFLOW_2P = "2p"
WORKFLOW_EPHYS = "ephys"


@dataclass
class AlignedNeuralData:
    """Aligned spike responses and neuron positions shared by both workflows."""

    spikes: np.ndarray
    neuron_pos: np.ndarray
    aligned_spikes: Optional[np.ndarray] = None


def load_two_photon_spikes(
    experiment_info: Tuple[str, str, int],
    data_dirs: Sequence[str],
    suite2p_dir: str,
    block_end: int,
    n_planes: int,
    nb_frames: int,
    resolution: float,
    *,
    spks_path: Optional[Path] = None,
    threshold: float = 1.25,
    method: str = "frame2ttl",
    correct_positions: bool = True,
) -> AlignedNeuralData:
    """Load and time-align two-photon (suite2p) spike data.

    When ``spks_path`` is set, reads pre-aligned ``spikes.npy`` and sibling
    ``pos.npy`` and skips suite2p alignment.
    """
    from . import LoadPinkNoise as lpn

    if spks_path is None:
        spikes, aligned_spikes, neuron_pos = lpn.loadSPKMesoscope(
            experiment_info,
            list(data_dirs),
            suite2p_dir,
            block_end,
            n_planes,
            nb_frames,
            threshold=threshold,
            last=True,
            method=method,
        )
        if correct_positions:
            neuron_pos = lpn.correctNeuronPos(neuron_pos, resolution)
    else:
        spikes = np.load(spks_path, mmap_mode="r")
        pos_path = spks_path.parent / "pos.npy"
        neuron_pos = np.load(pos_path)
        aligned_spikes = None

    np.save("pos.npy", neuron_pos)
    np.save("spikes.npy", spikes)

    return AlignedNeuralData(
        spikes=spikes,
        neuron_pos=neuron_pos,
        aligned_spikes=aligned_spikes,
    )


def align_ephys_data(
    data_dir: Path,
    nb_frames: int,
    sampling_rate: float,
    **kwargs: Any,
) -> AlignedNeuralData:
    
    import suite_ephys.DIO as DIO
    import numpy as np
    import os
    import re

    #======================================
    # PKL-SPECIFIC FUNCTIONS
    #======================================
    def get_pkl_path(directory):
        file_pattern = re.compile(r"\.pkl$")
        for filename in os.listdir(directory):
            if file_pattern.search(filename):
                return os.path.join(directory, filename)
        raise FileNotFoundError("No .pkl file found in data directory.")

    def extract_pos_and_spikes(units, start_times, end_times, pd_time, pd_state, nb_frames):
        n_trials = len(start_times)
        n_neurons = len(units)

        neuron_pos = np.zeros((n_neurons, 2))
        spikes = np.zeros((n_trials, nb_frames, n_neurons))

        for neuron_idx, neuron_data in enumerate(units.values()):
            
            neuron_pos[neuron_idx, :] = neuron_data['position'][:2] 
            spike_train = np.array(neuron_data['spike_train'])

            for trial_idx, (start_time, end_time) in enumerate(zip(start_times, end_times)):
                trial_pd_mask = (pd_time >= start_time) & (pd_time <= end_time)
                trial_pd_time = pd_time[trial_pd_mask]
                trial_pd_state = pd_state[trial_pd_mask]

                state_changes = np.diff(trial_pd_state) != 0 
                state_changes = np.insert(state_changes, 0, False) 
                
                flip_times = trial_pd_time[state_changes] 
                frame_edges = np.concatenate(([trial_pd_time[0]], flip_times, [trial_pd_time[-1]]))

                # ENFORCE EXACT BIN COUNT to prevent ValueError
                if len(frame_edges) > nb_frames + 1:
                    frame_edges = frame_edges[:nb_frames + 1]
                elif len(frame_edges) < nb_frames + 1:
                    frame_edges = np.linspace(trial_pd_time[0], trial_pd_time[-1], nb_frames + 1)

                trial_spikes = spike_train[(spike_train >= start_time) & (spike_train <= end_time)]

                # 1. Get raw counts per bin
                binned_counts, _ = np.histogram(trial_spikes, bins=frame_edges)
                
                # 2. Calculate the exact duration of each bin in seconds
                # np.diff gets the distance between edges in samples; divide by sampling_rate for seconds
                bin_durations_sec = np.diff(frame_edges) / sampling_rate
                
                # 3. Safety Check: Prevent division by zero 
                # (In case a hardware glitch caused two PD pulses to register at the exact same sample)
                bin_durations_sec[bin_durations_sec == 0] = 1e-9
                
                # 4. Calculate Firing Rate (Hz)
                firing_rate_hz = binned_counts / bin_durations_sec
                
                spikes[trial_idx, :, neuron_idx] = firing_rate_hz

        # Return pure counts. Do not subtract timestamps here.
        return neuron_pos, spikes
    
    #======================================
    # DIN-SPECIFIC FUNCTIONS
    #======================================
    def get_dio_files(dio_dir):
        dio_folders = DIO.get_dio_folders(dio_dir)
        return sorted(dio_folders, key=lambda x:x.name)

    def choose_correct_din_file(dio_files, port):
        temp_time, pd_state = DIO.concatenate_din_data(dio_files, port)
        return (temp_time - temp_time[0]), pd_state # aligns timestamps relative to start time, making it index 0
    
    def get_frequency(pd_time, fs):
        time_diff = np.diff(pd_time) / fs 
        freq = 1. / time_diff / 1000 
        return np.insert(freq, 0, 0) 
    
    def get_possible_trial_edges(freq, time_array):
        bin_freq = (freq >= 0.01).astype(int)
        chng_freq = np.diff(bin_freq)
        chng_freq = np.insert(chng_freq, 0, 0)
        
        start_times_idx = chng_freq == +1
        end_times_idx = chng_freq == -1

        return time_array[start_times_idx], time_array[end_times_idx]
    
    def validate_edges(starts, ends, stim_dur, fs, tolerance=0.01): 
        if starts.shape != ends.shape:
            raise Exception("Start timestamps array and End timestamps array are not of same size.")

        valid_times = ends - starts
        ideal = stim_dur * 60 * fs
        valid_mask = abs(valid_times - ideal) <= tolerance * ideal

        return starts[valid_mask], ends[valid_mask] 
    
    #======================================
    # MAIN EXECUTION
    #======================================
    pkl_path = get_pkl_path(data_dir)
    
    with open(pkl_path, "rb") as data:
        import pickle
        pkl_data = pickle.load(data) # Make sure to actually load the pickle!
        SAMPLING_RATE = pkl_data['metadata']['sampling_frequencies'][0]
        units = pkl_data['units']

    STIMULUS_DURATION = 10 

    dio_files = get_dio_files(data_dir)
    pd_time, pd_state = choose_correct_din_file(dio_files, 3)
    freq = get_frequency(pd_time, SAMPLING_RATE)

    start_times, end_times = get_possible_trial_edges(freq, pd_time) 
    start_times, end_times = validate_edges(start_times, end_times, STIMULUS_DURATION, SAMPLING_RATE, 0.01)

    neuron_pos, spikes = extract_pos_and_spikes(units, start_times, end_times, pd_time, pd_state, nb_frames)

    return AlignedNeuralData(
        spikes=spikes,
        neuron_pos=neuron_pos,
        aligned_spikes=None,
    )

def load_aligned_spikes(
    workflow: str,
    *,
    experiment_info: Tuple[str, str, int],
    data_dir: Path,
    data_dir_strings: Sequence[str],
    suite2p_dir: Path,
    block_end: int,
    n_planes: Optional[int],
    nb_frames: int,
    resolution: Optional[float],
    sampling_rate: Optional[float],
    spks_path: Optional[Path] = None,
    threshold: float = 1.25,
    method: str = "frame2ttl",
    correct_positions: bool = True,
) -> AlignedNeuralData:
    """Dispatch spike loading to the workflow-specific alignment routine."""
    if spks_path is not None:
        spikes = np.load(spks_path, mmap_mode="r")
        pos_path = spks_path.parent / "pos.npy"
        neuron_pos = np.load(pos_path)
        return AlignedNeuralData(spikes=spikes, neuron_pos=neuron_pos)

    if workflow == WORKFLOW_2P:
        if resolution is None or n_planes is None:
            raise ValueError("Two-photon workflow requires Resolution and Number of Planes")
        return load_two_photon_spikes(
            experiment_info,
            data_dir_strings,
            str(suite2p_dir),
            block_end,
            n_planes,
            nb_frames,
            resolution,
            threshold=threshold,
            method=method,
            correct_positions=correct_positions,
        )

    if workflow == WORKFLOW_EPHYS:
        if sampling_rate is None:
            raise ValueError("Ephys workflow requires Sampling Rate (samples / sec)")
        return align_ephys_data(
            data_dir,
            nb_frames,
            sampling_rate,
            experiment_info=experiment_info,
            threshold=threshold,
            method=method,
        )

    raise ValueError(f"Unknown workflow: {workflow!r}")


# Re-export core 2p alignment helpers for visibility in the codebase.
from .LoadPinkNoise import align_datas, correctNeuronPos, loadSPKMesoscope  # noqa: E402

__all__ = [
    "AlignedNeuralData",
    "WORKFLOW_2P",
    "WORKFLOW_EPHYS",
    "align_datas",
    "align_ephys_data",
    "correctNeuronPos",
    "load_aligned_spikes",
    "load_two_photon_spikes",
    "loadSPKMesoscope",
]
