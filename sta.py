def sta(
    data_dir: Path,
    nb_frames: int,
    sampling_rate: float,
    video_path: Path,
    save_dir: Optional[Path] = None,
    output_format: str = "npy",
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
        for neuron_idx, neuron_data in enumerate(units.values()):
            
            position = np.asarray(neuron_data.get('position', []), dtype=float).ravel()
            neuron_pos[neuron_idx, :min(3, position.size)] = position[:3]
            spike_train = np.array(neuron_data['spike_train'])

            for trial_idx, (start_time, end_time) in enumerate(zip(start_times, end_times)):
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

        # Return frame-aligned firing rates in Hz. Do not subtract timestamps here.
        return neuron_pos, spikes, unit_ids
    
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
    def get_dio_files(dio_dir):
        """Function for get dio files.

        Args:
            dio_dir: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        dio_folders = DIO.get_dio_folders(dio_dir)
        return sorted(dio_folders, key=lambda x:x.name)

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
    pkl_path = get_pkl_path(data_dir)
    
    with open(pkl_path, "rb") as data:
        import pickle
        pkl_data = pickle.load(data) # Make sure to actually load the pickle!
        SAMPLING_RATE = pkl_data['metadata']['sampling_frequencies'][0]
        units = pkl_data['units']

    STIMULUS_DURATION = 10 * 60 # IN SECONDS

    dio_files = get_dio_files(data_dir)
    pd_time, pd_state = choose_correct_din_file(dio_files, 3)
    freq = get_frequency(pd_time, SAMPLING_RATE)

    start_times, end_times = get_possible_trial_edges(freq, pd_time) 
    start_times, end_times = validate_edges(start_times, end_times, STIMULUS_DURATION, SAMPLING_RATE, 0.01)

    neuron_pos, spikes, unit_ids = extract_pos_and_spikes(units, start_times, end_times, pd_time, pd_state, nb_frames)
    


    # ======================================
    # NEW: POPULATION STA COMPUTATION
    # ======================================
    sta_3d = None
    stimulus_video = kwargs.get("stimulus_video", None)
    max_lag = kwargs.get("max_lag", 5)

    if stimulus_video is not None:
        trials_r, frames_r, n_neurons = spikes.shape
        trials_s, frames_s, height, width = stimulus_video.shape

        if trials_r == trials_s and frames_r == frames_s:
            total_time = trials_r * frames_r
            total_pixels = height * width
            
            # Flatten to 2D matrices
            S_flat = stimulus_video.reshape(total_time, total_pixels)
            R_flat = spikes.reshape(total_time, n_neurons)
            
            sta_raw = np.zeros((max_lag + 1, total_pixels, n_neurons))
            
            for lag in range(max_lag + 1):
                if lag == 0:
                    S_shifted = S_flat
                    R_shifted = R_flat
                else:
                    S_shifted = S_flat[:-lag, :]
                    R_shifted = R_flat[lag:, :]
                
                spike_counts = np.sum(R_shifted, axis=0)
                spike_counts[spike_counts == 0] = 1e-9 
                
                # THE ONE-LINER: Population STA Matrix Algebra
                sta_raw[lag, :, :] = np.dot(S_shifted.T, R_shifted) / spike_counts
                
            # Reshape back to visual space
            sta_3d = sta_raw.reshape(max_lag + 1, height, width, n_neurons)
        else:
            print("Warning: Stimulus dimensions do not match spike dimensions. Skipping STA.")

    # Update your return object to include the STA
    return AlignedNeuralData(
        spikes=spikes,
        neuron_pos=neuron_pos,
        aligned_spikes=None,
        unit_ids=unit_ids,
        sta=sta_3d, # Add this attribute to your AlignedNeuralData dataclass/model
    )






    # _save_aligned_outputs(neuron_pos, spikes, save_dir or data_dir, output_format, unit_ids=unit_ids)
# 
    # return AlignedNeuralData(
    #     spikes=spikes,
    #     neuron_pos=neuron_pos,
    #     aligned_spikes=None,
    #     unit_ids=unit_ids,
    # )








# Example output: (120, 2073600)  <- 120 frames, and 1920*1080 pixels