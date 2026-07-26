"""Time-alignment entry points for two-photon and electrophysiology workflows.

Two-photon alignment delegates to the existing Cortex Lab / suite2p pipeline.
Electrophysiology alignment is provided as a template for user implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .storage.neural_cache import (
    load_neural_cache_pair,
    save_aligned_neural_cache,
)
from .runtime.task_control import check_cancelled

WORKFLOW_2P = "2p"
WORKFLOW_EPHYS = "ephys"


@dataclass
class AlignedNeuralData:
    """Aligned spike responses and neuron positions shared by both workflows."""

    spikes: np.ndarray
    neuron_pos: np.ndarray
    aligned_spikes: Optional[np.ndarray] = None
    unit_ids: Optional[np.ndarray] = None
    unit_info: Optional[Sequence[Dict[str, Any]]] = None


def normalize_excluded_trial_numbers(values: Optional[Sequence[int]]) -> Tuple[int, ...]:
    """Validate and de-duplicate one-indexed trial numbers requested for removal."""
    normalized = []
    for value in values or ():
        if isinstance(value, bool):
            raise ValueError("Excluded Trial Numbers must contain positive whole numbers")
        number = int(value)
        if number != value or number < 1:
            raise ValueError("Excluded Trial Numbers must contain positive whole numbers")
        if number not in normalized:
            normalized.append(number)
    return tuple(normalized)


def exclude_identified_trials(
    spikes: np.ndarray,
    excluded_trial_numbers: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Drop requested one-indexed trials, reporting any request beyond the recording."""
    if np.ndim(spikes) < 1:
        raise ValueError("Aligned neural responses must have a trial axis")

    requested = normalize_excluded_trial_numbers(excluded_trial_numbers)
    if not requested:
        return spikes

    n_trials = int(spikes.shape[0])
    included = tuple(number for number in requested if number <= n_trials)
    ignored = tuple(number for number in requested if number > n_trials)
    if ignored:
        print(
            f"Identified {n_trials} trial(s), so the maximum identified trial is {n_trials}. "
            f"Ignored out-of-range exclusion(s) {list(ignored)}; "
            f"excluded matching trial(s) {list(included)}."
        )
    elif included:
        print(f"Excluded identified trial(s) {list(included)} from {n_trials} trial(s).")

    excluded = set(included)
    keep_indices = [index for index in range(n_trials) if index + 1 not in excluded]
    return np.asarray(spikes)[keep_indices]


def _save_aligned_outputs(
    neuron_pos: np.ndarray,
    spikes: np.ndarray,
    save_dir: Optional[Path],
    output_format: str = "npy",
    unit_ids: Optional[Sequence[object]] = None,
    unit_info: Optional[Sequence[Dict[str, Any]]] = None,
    cancel_event=None,
) -> dict:
    """Persist aligned arrays beside the experiment data for later GUI reuse."""
    check_cancelled(cancel_event)
    return save_aligned_neural_cache(
        neuron_pos,
        spikes,
        save_dir,
        output_format,
        unit_ids=unit_ids,
        unit_info=unit_info,
    )


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
    save_dir: Optional[Path] = None,
    output_format: str = "npy",
    stimulus_duration: Optional[float] = None,
    excluded_trial_numbers: Optional[Sequence[int]] = None,
    cancel_event=None,
) -> AlignedNeuralData:
    """Load and time-align two-photon (suite2p) spike data.

    When ``spks_path`` is set, reads pre-aligned ``spikes`` and sibling ``pos``
    caches in either NPY or Zarr format and skips suite2p alignment.

    ``stimulus_duration`` is accepted to keep the shared alignment dispatcher
    interface consistent with electrophysiology.  Two-photon alignment derives
    its timing from the recorded TTL/frame data, so it is not used here.
    """
    check_cancelled(cancel_event)
    from .data import neural as neural_io

    if spks_path is None:
        check_cancelled(cancel_event)
        suite2p_dir = neural_io.validate_suite2p_output(suite2p_dir, n_planes)
        neural_io.validate_two_photon_timeline(experiment_info, data_dirs)
        check_cancelled(cancel_event)
        spikes, aligned_spikes, neuron_pos = neural_io.loadSPKMesoscope(
            experiment_info,
            list(data_dirs),
            str(suite2p_dir),
            block_end,
            n_planes,
            nb_frames,
            threshold=threshold,
            last=True,
            method=method,
            cancel_event=cancel_event,
        )
        check_cancelled(cancel_event)
        if correct_positions:
            neuron_pos = neural_io.correctNeuronPos(neuron_pos, resolution, n_planes)
    else:
        spikes, neuron_pos, _, _ = load_neural_cache_pair(spks_path.parent, spks_path)
        aligned_spikes = None

    check_cancelled(cancel_event)
    spikes = exclude_identified_trials(spikes, excluded_trial_numbers)

    if spks_path is None:
        _save_aligned_outputs(neuron_pos, spikes, save_dir, output_format, cancel_event=cancel_event)

    return AlignedNeuralData(
        spikes=spikes,
        neuron_pos=neuron_pos,
        aligned_spikes=aligned_spikes,
    )


def align_ephys_data(
    data_dir: Path,
    nb_frames: int,
    sampling_rate: float,
    save_dir: Optional[Path] = None,
    output_format: str = "npy",
    stimulus_duration: Optional[float] = None,
    photodiode_port: Optional[int] = None,
    excluded_trial_numbers: Optional[Sequence[int]] = None,
    cancel_event=None,
    **kwargs: Any,
) -> AlignedNeuralData:
    
    """Function for align ephys data.

    Args:
        data_dir: Input value for this operation.
        nb_frames: Input value for this operation.
        sampling_rate: Input value for this operation.
        save_dir: Input value for this operation.
        kwargs: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    from .suite_ephys import DIO as DIO
    import numpy as np
    import os
    import re

    #======================================
    # PKL-SPECIFIC FUNCTIONS
    #======================================
    def get_pkl_path(directory):
        """Function for get pkl path.

        Args:
            directory: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        file_pattern = re.compile(r"\.pkl$")
        for filename in os.listdir(directory):
            if file_pattern.search(filename):
                return os.path.join(directory, filename)
        raise FileNotFoundError("No .pkl file found in data directory.")

    def extract_pos_and_spikes(units, start_times, end_times, pd_time, pd_state, nb_frames):
        """Function for extract pos and spikes.

        Args:
            units: Input value for this operation.
            start_times: Input value for this operation.
            end_times: Input value for this operation.
            pd_time: Input value for this operation.
            pd_state: Input value for this operation.
            nb_frames: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        n_trials = len(start_times)
        n_neurons = len(units)

        neuron_pos = np.zeros((n_neurons, 3))
        spikes = np.zeros((n_trials, nb_frames, n_neurons))

        unit_ids = np.asarray(list(units.keys()), dtype=object)
        unit_info = []
        for neuron_idx, (unit_id, neuron_data) in enumerate(units.items()):
            check_cancelled(cancel_event)
            
            position = np.asarray(neuron_data.get('position', []), dtype=float).ravel()
            neuron_pos[neuron_idx, :min(3, position.size)] = position[:3]
            spike_train = np.array(neuron_data['spike_train'])
            shank_group = neuron_data.get("shank_group")
            unit_info.append(
                {
                    "shank": f"shank{shank_group}" if shank_group is not None else "",
                    "unit": f"unit{unit_id}",
                    "shank_group": shank_group,
                    "unit_id": unit_id,
                    "position": position[:3].tolist(),
                    "quality": neuron_data.get("label", ""),
                    "n_spikes": int(neuron_data.get("n_spikes", spike_train.size)),
                }
            )

            for trial_idx, (start_time, end_time) in enumerate(zip(start_times, end_times)):
                check_cancelled(cancel_event)
                trial_pd_mask = (pd_time >= start_time) & (pd_time <= end_time)
                trial_pd_time = pd_time[trial_pd_mask]
                trial_pd_state = pd_state[trial_pd_mask]

                state_changes = np.diff(trial_pd_state) != 0 
                state_changes = np.insert(state_changes, 0, False) 
                
                flip_times = trial_pd_time[state_changes] 
                frame_edges = np.concatenate(([trial_pd_time[0]], flip_times, [trial_pd_time[-1]]))

                # CONVERT TO LIST: better for handle_dropped_frames algorithm (insertions into list) than memory-fixed numpy arrays
                frame_edges = [trial_pd_time[0]] + list(flip_times) + [trial_pd_time[-1]]

                # Apply the in-place fix
                handle_dropped_frames(frame_edges, nb_frames)

                # Convert back to a numpy array right before histogram binning
                frame_edges = np.array(frame_edges)

                trial_spikes = spike_train[(spike_train >= start_time) & (spike_train <= end_time)]

                # Count events in each photodiode-defined frame bin.
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

        # Return frame-aligned firing rates in Hz. Do not subtract timestamps here.
        return neuron_pos, spikes, unit_ids, unit_info
    
    def handle_dropped_frames(frame_edges: list, nb_frames: int) -> None:
        """
        Modifies the frame_edges list in-place to correct for dropped photodiode pulses
        and match the required length of (nb_frames + 1).
        """
        if len(frame_edges) < 2:
            return  # Not enough data to calculate gaps

        # 1. Calculate expected frame duration using the median
        frame_durations = np.diff(frame_edges)
        expected_duration = np.median(frame_durations)

        # 2. Patch missing pulses in the middle of the trial
        i = 0
        while i < len(frame_edges) - 1 and len(frame_edges) < nb_frames + 1:
            current_gap = frame_edges[i+1] - frame_edges[i]
            
            # If the gap is > 1.5x the expected duration, a pulse was missed
            if current_gap > 1.5 * expected_duration:
                # Insert the synthetic timestamp directly into the list
                frame_edges.insert(i + 1, frame_edges[i] + expected_duration)
            i += 1

        # 3. Handle tail-end mismatches
        if len(frame_edges) > nb_frames + 1:
            # Slice off any hardware bounce/excess pulses at the very end
            del frame_edges[nb_frames + 1:]
            
        elif len(frame_edges) < nb_frames + 1:
            # Append missing trailing edges if the recording cut off early
            missing_count = (nb_frames + 1) - len(frame_edges)
            for _ in range(missing_count):
                frame_edges.append(frame_edges[-1] + expected_duration)


    #======================================
    # DIN-SPECIFIC FUNCTIONS
    #======================================
    def get_dio_files(dio_dir, port):
        """Function for get dio files.

        Args:
            dio_dir: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        return DIO.get_dio_folders(dio_dir, channel_id=port)

    def choose_correct_din_file(dio_files, port):
        """Function for choose correct din file.

        Args:
            dio_files: Input value for this operation.
            port: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        temp_time, pd_state = DIO.concatenate_din_data(dio_files, port)
        return (temp_time - temp_time[0]), pd_state # aligns timestamps relative to start time, making it index 0
    
    def get_frequency(pd_time, fs):
        """Function for get frequency.

        Args:
            pd_time: Input value for this operation.
            fs: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        time_diff = np.diff(pd_time) / fs 
        freq = 1. / time_diff # in Hz
        return np.insert(freq, 0, 0) 
    
    # BINARIZATION code: Trial ON or OFF
    def get_possible_trial_edges(freq, time_array):
        """Function for get possible trial edges.

        Args:
            freq: Input value for this operation.
            time_array: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        bin_freq = (freq >= 10).astype(int) # in Hz, HARDCODED 10 VALUE
        chng_freq = np.diff(bin_freq)
        chng_freq = np.insert(chng_freq, 0, 0)
        
        start_times_idx = chng_freq == +1
        end_times_idx = chng_freq == -1

        return time_array[start_times_idx], time_array[end_times_idx]
    
    def validate_edges(starts, ends, stim_dur, fs, tolerance=0.01): 
        """Function for validate edges.

        Args:
            starts: Input value for this operation.
            ends: Input value for this operation.
            stim_dur: Input value for this operation.
            fs: Input value for this operation.
            tolerance: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if starts.shape != ends.shape:
            raise Exception("Start timestamps array and End timestamps array are not of same size.")

        valid_times = ends - starts
        ideal = stim_dur * fs
        valid_mask = abs(valid_times - ideal) <= tolerance * ideal

        return starts[valid_mask], ends[valid_mask] 
    
    #======================================
    # MAIN EXECUTION
    #======================================
    check_cancelled(cancel_event)
    pkl_path = get_pkl_path(data_dir)
    
    with open(pkl_path, "rb") as data:
        import pickle
        pkl_data = pickle.load(data) # Make sure to actually load the pickle!
        try:
            SAMPLING_RATE = pkl_data['metadata']['sampling_frequencies'][0]
        except KeyError:
            SAMPLING_RATE = pkl_data['metadata']['sampling_frequency']
        units = pkl_data['units']

    check_cancelled(cancel_event)
    if stimulus_duration is None or stimulus_duration <= 0:
        raise ValueError("Ephys alignment requires the stimulus movie duration from metadata.")

    if photodiode_port is None:
        raise ValueError("Ephys alignment requires a selected Photodiode Port")
    photodiode_port = int(photodiode_port)
    if photodiode_port < 0:
        raise ValueError("Photodiode Port must be a non-negative digital-input port number")
    dio_files = get_dio_files(data_dir, photodiode_port)
    print(
        f"Using photodiode digital-input port {photodiode_port} from "
        f"{len(dio_files)} discovered .dat folder(s)."
    )
    check_cancelled(cancel_event)
    pd_time, pd_state = choose_correct_din_file(dio_files, photodiode_port)
    freq = get_frequency(pd_time, SAMPLING_RATE)

    start_times, end_times = get_possible_trial_edges(freq, pd_time) 
    start_times, end_times = validate_edges(start_times, end_times, stimulus_duration, SAMPLING_RATE, 0.01)

    check_cancelled(cancel_event)
    neuron_pos, spikes, unit_ids, unit_info = extract_pos_and_spikes(
        units, start_times, end_times, pd_time, pd_state, nb_frames
    )
    check_cancelled(cancel_event)
    spikes = exclude_identified_trials(spikes, excluded_trial_numbers)
    
    print(neuron_pos.shape)
    print(spikes.shape)
    cache_dir = save_dir or data_dir
    _save_aligned_outputs(
        neuron_pos,
        spikes,
        cache_dir,
        output_format,
        unit_ids=unit_ids,
        unit_info=unit_info,
        cancel_event=cancel_event,
    )

    return AlignedNeuralData(
        spikes=spikes,
        neuron_pos=neuron_pos,
        aligned_spikes=None,
        unit_ids=unit_ids,
        unit_info=unit_info,
    )

# TEST EPHYS CODE
#import time
#start_time = time.time()
#                 30 * 60 * 10,
#                 30000)
#end_time = time.time()
#print(f"TIME: {end_time - start_time:.6f} seconds")
#
#raise SystemExit

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
    photodiode_port: Optional[int] = None,
    spks_path: Optional[Path] = None,
    threshold: float = 1.25,
    method: str = "frame2ttl",
    correct_positions: bool = True,
    save_dir: Optional[Path] = None,
    output_format: str = "npy",
    stimulus_duration: Optional[float] = None,
    excluded_trial_numbers: Optional[Sequence[int]] = None,
    reuse_cache: bool = True,
    cancel_event=None,
) -> AlignedNeuralData:
    """Dispatch spike loading to the workflow-specific alignment routine."""
    check_cancelled(cancel_event)
    cache_dir = Path(".") if save_dir is None else Path(save_dir)
    if save_dir is None:
        if workflow == WORKFLOW_2P and data_dir_strings:
            cache_dir = Path(data_dir_strings[0]) / experiment_info[0] / experiment_info[1] / str(experiment_info[2])
        else:
            cache_dir = Path(data_dir)

    if spks_path is not None:
        spikes, neuron_pos, _, _ = load_neural_cache_pair(spks_path.parent, spks_path)
        check_cancelled(cancel_event)
        return AlignedNeuralData(
            spikes=exclude_identified_trials(spikes, excluded_trial_numbers),
            neuron_pos=neuron_pos,
        )

    if reuse_cache:
        try:
            spikes, neuron_pos, _, _ = load_neural_cache_pair(cache_dir)
            check_cancelled(cancel_event)
            return AlignedNeuralData(
                spikes=exclude_identified_trials(spikes, excluded_trial_numbers),
                neuron_pos=neuron_pos,
            )
        except FileNotFoundError:
            pass

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
            save_dir=cache_dir,
            output_format=output_format,
            stimulus_duration=stimulus_duration,
            excluded_trial_numbers=excluded_trial_numbers,
            cancel_event=cancel_event,
        )

    if workflow == WORKFLOW_EPHYS:
        if sampling_rate is None:
            raise ValueError("Ephys workflow requires Sampling Rate (samples / sec)")
        if photodiode_port is None:
            raise ValueError("Ephys workflow requires a selected Photodiode Port")
        return align_ephys_data(
            data_dir,
            nb_frames,
            sampling_rate,
            save_dir=cache_dir,
            output_format=output_format,
            stimulus_duration=stimulus_duration,
            photodiode_port=photodiode_port,
            experiment_info=experiment_info,
            threshold=threshold,
            method=method,
            excluded_trial_numbers=excluded_trial_numbers,
            cancel_event=cancel_event,
        )

    raise ValueError(f"Unknown workflow: {workflow!r}")


# Re-export core 2p alignment helpers for visibility in the codebase.
from .data.neural import align_datas, correctNeuronPos, loadSPKMesoscope  # noqa: E402

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
