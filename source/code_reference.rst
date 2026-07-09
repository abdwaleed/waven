waven Code Reference
====================

This page summarizes what each code module is for and what its public functions
or classes do.  It is intentionally written as a developer map rather than a
mathematical methods paper.

Application and entry points
----------------------------

``waven``
    Public package API.  Re-exports configuration classes and high-level
    pipeline functions for convenient imports.

``waven.__getattr__(name)``
    Lazily imports public subpackages such as ``app``, ``gui``, ``runtime``, and
    ``storage`` when accessed as attributes.

``waven.gui``
    Stable public GUI launcher module.

``waven.app.gui``
    Interactive GUI implementation.  It owns workflow selection, input forms,
    button callbacks, terminal redirection, progress display, cancellation,
    cleanup, figure embedding, plot cache export, and state save/load.

``waven.app.gui.select_workflow()``
    Returns the default workflow used by backwards-compatible callers.

``waven.app.gui.run(param_defaults, gabor_param, workflow=None)``
    Builds and runs the GUI using default analysis and Gabor parameters.

Configuration
-------------

``waven.config``
    Converts GUI/JSON configuration into typed Python objects.

``parse_literal(value, label)``
    Safely parses a Python literal string such as ``"[1, 2, 3]"`` without using
    ``eval``.

``parse_path(value)``
    Normalizes a required filesystem path, expanding ``~`` and environment
    variables.

``parse_optional_path(value)``
    Returns ``None`` for empty path-like values; otherwise normalizes the path.

``sibling_path_with_suffix(path, suffix)``
    Builds a sibling path by inserting a suffix before the extension.

``coarse_grid_dimensions(nx, ny)``
    Returns the coarse RF grid dimensions, currently 20 percent of full
    resolution rounded to integers.

``coarse_to_full_scale(nx, ny)``
    Returns scale factors for mapping coarse RF indices back to full-resolution
    coordinates.

``resolve_sigma_indices(library_sigmas, requested_sigmas)``
    Finds the sigma planes in a Gabor library corresponding to requested sigma
    values.

``GaborConfig``
    Dataclass-like configuration for Gabor library dimensions, sigmas, phases,
    frequencies, and output paths.

``AnalysisConfig``
    Dataclass-like configuration for stimulus movies, neural data, analysis
    coverage, model settings, and output directories.

``PipelineConfig``
    Top-level configuration object combining ``GaborConfig`` and
    ``AnalysisConfig``.

``default_pipeline_config(workflow)``
    Returns a default config template for two-photon or ephys workflows.

Pipeline orchestration
----------------------

``waven.pipeline``
    Scriptable orchestration layer for running analysis stages without the GUI.

``SpikeData``
    Container for frame-aligned spikes and corrected neuron positions.

``WaveletData``
    Container for real, imaginary, and magnitude coarse wavelets.

``RFAnalysisResult``
    Container for repeatability, skewness, receptive fields, and RF metadata.

``SimpleModelResult``
    Container for outputs from the fast nonlinear Gabor-wavelet model.

``PipelineOutputs``
    Aggregate return value from ``run_pipeline``.

``require_file(path, label)``
    Raises a clear error when a required file is missing.

``require_directory(path, label)``
    Raises a clear error when a required directory is missing.

``legacy_directory_string(path)``
    Formats a directory path for legacy code that concatenates strings.

``validate_spike_data(spikes, neuron_pos)``
    Checks core spike/position shape contracts before analysis.

``create_gabor_library(config)``
    Builds and saves a Gabor filter library described by a config object.

``create_coarse_gabor_library(config)``
    Builds the lower-resolution coupled library used by coarse RF search.

``create_fine_gabor_library(config)``
    Builds the full-resolution independent size/frequency library used by the
    full model.

``model_trial_indices(analysis, n_trials)``
    Resolves train/test trial indices, including automatic test-index selection.

``prepare_stimulus_wavelets(analysis, library_path=None, chunk_size=None)``
    Down-samples the stimulus movie and writes coarse real/imaginary wavelet
    phase files.

``load_spikes_and_positions(analysis, threshold, method, correct_positions)``
    Loads workflow-specific neural data and returns validated spike/position
    arrays.

``prepare_full_model_wavelets(analysis, gabor, library_path=None, output_dir=None)``
    Builds full-resolution wavelet arrays for ``run_Full_Model``.

``load_coarse_wavelets(analysis, gabor)``
    Loads or builds the coarse wavelet cache consumed by RF and fast model
    analysis.

``run_rf_analysis(analysis, gabor, spike_data, plotting=True)``
    Computes repeatability, skewness, RF correlations, and optional RF plots.

``smooth_best_positions(best_positions, neuron_pos, radius_um, min_neighbors)``
    Smooths preferred RF positions with nearby-neuron medians.

``run_simple_model(analysis, gabor, spike_data, wavelets, rf_result)``
    Runs the fast nonlinear model across neurons.

``run_full_model(analysis, spike_data, rf_result)``
    Runs the high-granularity model using configured full-model wavelet paths.

``run_pipeline(config, ...)``
    Runs selected pipeline stages and returns ``PipelineOutputs``.

Wavelets
--------

``waven.wavelets.filters``
    Builds Gabor kernels and filter libraries.

``has_enough_ram(required_bytes, safety_margin=1.20)``
    Logs whether a workload should fit in available RAM.

``makeGaborFilter(i, j, angle, sigma, phase, f, lx, ly, plot, freq)``
    Builds one spatial Gabor filter centered at a stimulus-grid coordinate.

``makeGaborFilter3D(i, j, angle, sigma, tp_w, f, lx, ly, alpha1, alpha2)``
    Builds a small stack of Gabor filters across phase offsets.

``makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq, cancel_event=None)``
    Builds a coupled Gabor library for coarse RF search.

``makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies, cancel_event=None)``
    Builds a full-resolution Gabor library with independent frequency planes.

``makeFilterLibrary3D(xs, ys, thetas, sigmas, offsets, f, tp_w, alpha1, alpha2, filename)``
    Builds and saves a legacy 3D Gabor library.

``waven.wavelets.decomposition``
    Applies Gabor libraries to down-sampled stimulus movies.

``downsample_video_binary(path, visual_coverage, analysis_coverage, shape, chunk_size, ratios, save_path, cancel_event=None)``
    Streams a binary stimulus movie into a down-sampled boolean NPY array.

``downsample_video_uint(path, shape)``
    Down-samples a movie to uint8 frames using threaded chunk processing.

``getWTfromNPY(videodata, waveletLibrary, phase, WT_flat, s_idx, filter_chunk_size, frequency_index, cancel_event=None)``
    Projects video frames onto one Gabor library slice and writes coefficients
    into a caller-provided output buffer.

``waveletTransform(frame, phase, L)``
    Legacy helper that applies one wavelet-library phase to one frame.

``waveletTransform3D(frame, L)``
    Legacy helper that applies a 3D wavelet library to one frame.

``waveletDecomposition(videodata, phase, sigmas, folder_path, library_path, cancel_event=None)``
    Writes coarse time-first wavelet coefficients for one phase.

``waveletDecompositionFull(videodata, phase, sigmas, frequencies, folder_path, library_path, ...)``
    Writes full-resolution wavelets as NPY or direct Zarr for one phase.

``getTrueRF(idx, rfs, L)``
    Legacy plotting helper for reconstructing one RF from a library.

Stimulus and storage
--------------------

``waven.stimulus.wavelet_cache``
    Loads coarse/full wavelet files and builds the down-sampled RF cache.

``load_stimulus_simple_cell(path, nx, ny, no, ns, nf, downsampling, nx0, ny0)``
    Loads coarse real/imaginary wavelet phase arrays.

``load_stimulus_simple_cell2_i(...)`` and ``load_stimulus_simple_cell2_r(...)``
    Load imaginary or real full-model wavelets, optionally spatially downsampled.

``load_stimulus_simple_cell2(...)``
    Loads both full-model real and imaginary wavelet arrays.

``coarseWavelet(path, downsampling, nx0, ny0, no, ns, nf, nx, ny, chunk_size, cancel_event=None)``
    Builds or loads ``dwt_downsampled_videodata.npy`` for RF/model analysis.

``waven.stimulus.full_model``
    Loads full-resolution stimulus wavelets for model analysis.

``load_wavelets(path, waveform, wavelet_baseline, nf)``
    Loads real/imaginary phases and returns normalized magnitude features.

``load_stimulus(path, tt, waveform, nx, ny, no, ns, nf, ...)``
    Loads a full-model stimulus slice and optionally downsamples it.

``waven.storage.array_store.load_array(path, ...)``
    Opens a Zarr store when available, otherwise loads NumPy arrays with optional
    memory mapping.

``waven.storage.wavelet_zarr.convert_npy_to_zarr(...)``
    Converts full-model real/imaginary NPY files into compressed Zarr stores and
    can delete converted NPY sources after success.

Runtime
-------

``waven.runtime.performance``
    Centralizes hardware-aware chunk sizes and CPU/GPU resource checks.

``get_gpu_count()``
    Returns the number of CUDA devices.

``gpu_vram_bytes(device_index=0)``
    Returns VRAM bytes for one CUDA device.

``available_ram_bytes()``
    Returns currently available system RAM.

``has_enough_ram(required_bytes, safety_margin)``
    Checks whether a workload fits into available RAM with a margin.

``cpu_worker_count(reserve=1)``
    Chooses a conservative CPU worker count.

``resolve_compute_device(prefer_gpu=True)``
    Returns ``"cuda"`` or ``"cpu"``.

``wavelet_filter_chunk_size()``
    Chooses how many filters to process in one wavelet matmul batch.

``video_downsample_chunk_size()``
    Chooses movie-frame chunk size for video downsampling.

``gpu_neuron_chunk_size(stimulus_bytes_per_neuron)``
    Chooses neuron chunk size for Pearson correlation on GPU.

``model_parallel_jobs()``
    Chooses a joblib worker count for per-neuron model fitting.

``coarse_wavelet_chunk_size(...)`` and ``coarse_wavelet_chunk_size_gpu_or_cpu(...)``
    Choose frame chunks for coarse wavelet cache construction.

``waven.runtime.keep_awake.KeepAwake``
    Prevents the OS from sleeping during long analysis tasks.

``waven.runtime.task_control.OperationCancelled``
    Exception raised when a cooperative cancellation request is detected.

``check_cancelled(cancel_event)``
    Raises ``OperationCancelled`` when a cancellation event is set.

``format_duration(seconds)``
    Formats elapsed/ETA durations for terminal output.

``progress_message(label, completed, total, start_time, unit)``
    Formats consistent progress strings with percent, elapsed time, ETA, and
    throughput.

Neural data and time alignment
------------------------------

``waven.time_alignment``
    Dispatches two-photon and ephys spike loading into one aligned data shape.

``AlignedNeuralData``
    Container for aligned spikes, raw spikes, neuron positions, and frame timing.

``load_two_photon_spikes(...)``
    Loads suite2p/timeline-derived two-photon spikes.

``align_ephys_data(...)``
    Template ephys alignment function for Trodes/DIO-style workflows.

``load_aligned_spikes(workflow, ...)``
    Dispatches to the workflow-specific alignment implementation.

``waven.data.neural``
    Lower-level neural loaders and frame-alignment helpers.

``loadExperiment(...)``
    Loads experiment metadata and timeline files.

``align_rotary_encoder(...)``
    Aligns rotary encoder traces to stimulus timing.

``align_datas(...)``
    Segments neural activity into stimulus trials and resamples onto frame grid.

``loadFluoMesoscope(...)`` and ``loadSPKMesoscope(...)``
    Load fluorescence or deconvolved spike traces from mesoscope/suite2p data.

``correctNeuronPos(neuron_pos, ...)``
    Corrects neuron positions across planes or acquisition geometry.

Analysis
--------

``waven.analysis.receptive_fields``
    RF correlation, retinotopy, repeatability, and signal-processing helpers.

Key public functions:
    ``PearsonCorrelation`` and ``PearsonCorrelationPinkNoise`` compute RF
    correlations and preferred visual features. ``compute_skewness_neurons``
    filters neurons by response skewness. ``repetability_trial*`` variants
    estimate repeatability. ``wavelet_feature_dims`` and
    ``coarse_indices_to_full`` manage wavelet feature indexing.  ``cart2pol*``,
    ``preferedDirection``, ``DirectionSelectivity*``, and ``SinCosPlot*`` support
    orientation/direction tuning calculations and plots.

``waven.analysis.nonlinear_models``
    Nonlinear model fitting, STA/STC utilities, prediction metrics, and tuning
    visualization.

Key public functions:
    ``compute_sta``, ``spikeTrig``, ``compute_stc``, and ``CovspikeTrig*`` build
    spike-triggered summaries. ``getNonLinearModel*``, ``computeNonlin*``,
    ``fitnonlin``, and activation helpers fit nonlinear response mappings.
    ``GetNeuronVisresponse`` and ``PredictNeuronsTest`` prepare model inputs and
    predictions. ``PlotTuningCurve``, ``Plot_RF``, ``PlotSelfCorrelation``, and
    ``PlotR2scoreAnalysis`` create diagnostic visualizations.

``waven.analysis.model_runs``
    Top-level model execution routines.

``signaltonoiseScipy(...)``
    Computes signal-to-noise style response metrics.

``run_Model(...)``
    Fits the fast nonlinear Gabor-wavelet model for every neuron.

``run_Full_Model(...)``
    Runs the full-resolution model path using full wavelet arrays.

``plotNeuralRaster(spks, ...)``
    Plots neurons sorted by the first sparse-SVD component.

``waven.analysis.trial_stats``
    Trial statistics, retinotopy map, and visual sign-map helpers.

Key public functions:
    ``circular_variance`` measures orientation concentration.
    ``compute_signal_related_variance`` estimates signal-related variance.
    ``split_trials`` and ``stimresp_matrix`` reshape trial data.
    ``lowess`` smooths data with uncertainty. ``visualSignMap`` and
    ``getSignMap`` compute visual sign maps. ``filter_nan_gaussian_conserving2``
    smooths NaN-containing arrays.

Plotting
--------

``waven.plotting``
    Figure helpers for scriptable RF analysis.

``finite_abs_max(values, fallback=1.0)``
    Returns a finite nonzero color-scale bound.

``plot_rf_grid(rf, ...)``
    Plots one neuron's RF tensor as orientation-by-size panels.

``plot_retinotopic_maps(neuron_pos, ...)``
    Plots retinotopy and preferred visual feature maps over neuron positions.

``plot_sign_map(sign_map, ...)``
    Plots a visual sign map.

``save_figures(figures, save_dir)``
    Saves named matplotlib figures to disk.

External workflow helpers
-------------------------

``waven.suite2p.utils``
    Vendor/workflow helpers for suite2p, timeline, mpep, and experiment metadata.
    These modules are intentionally isolated because their function names mirror
    upstream data conventions.

``waven.suite_ephys``
    Trodes/DIO helpers for electrophysiology alignment.

Compatibility modules
---------------------

``waven.zebraGUI``, ``waven.WaveletGenerator``, ``waven.LoadPinkNoise``,
``waven.Analysis_Utils``, ``waven.performance``, ``waven.power``,
``waven.task_control``, ``waven.wavelet_io``, and ``waven.zarr_compat`` are
compatibility wrappers.  They should stay small and should not receive new
implementation logic.

