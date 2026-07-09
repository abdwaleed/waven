.. _methods_transparency:

Methods and Transparency Guide
==============================

This page documents the current analysis workflow so researchers can inspect how
the GUI results are produced from raw stimulus and neural data.  It is intended
to make the pipeline auditable: each section names the relevant code, the data
arrays being created, the assumptions made, and the files that connect one step
to the next.

The main GUI implementation is ``src/waven/app/gui.py``.  The legacy
``src/waven/zebraGUI.py`` module remains as a compatibility wrapper.  The core
scientific operations live in:

* ``src/waven/wavelets/filters.py`` and
  ``src/waven/wavelets/decomposition.py`` for Gabor filter construction, video
  downsampling, and wavelet projection.
* ``src/waven/stimulus/wavelet_cache.py`` for coarse wavelet cache
  loading/building, with two-photon data support in ``src/waven/data/neural.py``.
* ``src/waven/time_alignment.py`` for two-photon and electrophysiology alignment
  dispatch.
* ``src/waven/analysis/`` for receptive-field estimation, tuning curves, simple
  model fitting, and the full model.
* ``src/waven/storage/wavelet_zarr.py`` for conversion of full-model wavelet
  ``.npy`` arrays into compressed Zarr stores.


Workflow Summary
----------------

The GUI is staged.  A typical analysis goes:

1. Select a workflow: two-photon or electrophysiology.
2. Build or load a Gabor filter library.
3. Decompose the visual stimulus into Gabor wavelet responses.
4. Align neural activity to stimulus frames or load pre-aligned ``spikes.npy``.
5. Run coarse receptive-field analysis.
6. Inspect individual neurons and population retinotopy maps.
7. Run the simple model or full model for selected neurons.
8. Export displayed figures and reusable data.

The central data convention is:

* ``spikes`` has shape ``(n_trials, n_frames, n_neurons)``.
* ``neuron_pos`` has shape ``(n_neurons, 2)`` or ``(n_neurons, 3)``.
* Coarse wavelets are stored as time-first arrays used for fast receptive-field
  screening.
* Full-model wavelets are stored at full spatial resolution for higher-resolution
  model fitting.


Stage 1: Gabor Filter Library
-----------------------------

GUI buttons:

* ``Build Coarse Library``
* ``Build Fine Library``
* ``Build Coarse + Fine Libraries``

Primary code:

* ``zebraGUI.create_gabor``
* ``WaveletGenerator.makeGaborFilter``
* ``WaveletGenerator.makeFilterLibrary``
* ``WaveletGenerator.makeFilterLibrary2``

What is built:

The library contains spatial Gabor filters positioned across the downsampled
stimulus grid.  Each filter is defined by:

* x/y grid position
* orientation
* size, represented by Gaussian ``sigma`` in pixels
* spatial frequency
* phase, usually 0 and 90 degrees

The filter itself is generated with ``skimage.filters.gabor_kernel``.  The
real-valued Gabor kernel is inserted into a larger image canvas at the requested
position, then cropped back to the stimulus grid and flattened for storage.

The GUI now creates two explicit library types:

* The coarse library is built on ``coarse_grid_dimensions(NX, NY)``, currently
  20 percent of the configured full grid.  It uses ``makeFilterLibrary`` with
  ``freq=False``, so size and spatial frequency are coupled by the historical
  WavEn relationship in ``makeGaborFilter``.  This is the paper-style coarse RF
  search library.
* The fine library is built on the full ``NX x NY`` grid.  When nonzero
  frequencies are configured, it uses ``makeFilterLibrary2`` and stores size and
  spatial frequency as independent axes.  This is the paper-style fine model
  library.

Typical output:

* ``Coarse Library Path`` as ``.npy`` or ``.zarr``
* ``Fine Library Path`` as ``.npy`` or ``.zarr``

Important assumptions:

* Orientations are sampled from 0 to 180 degrees, not 0 to 360 degrees.
* Sigma values are in stimulus pixels at the configured analysis resolution.
* Frequency values are cycles per pixel when the library is generated.
* The coarse library must not contain an independent frequency axis.  The code
  raises an error rather than silently using frequency index 0.
* The fine library must match the full analysis grid size, orientation count,
  requested full-model sigma list, and frequency list used later.


Stage 2: Stimulus Downsampling and Wavelet Decomposition
-------------------------------------------------------

GUI button: ``Run Wavelet Decomposition``

Primary code:

* ``zebraGUI.run_wavelet``
* ``WaveletGenerator.downsample_video_binary``
* ``WaveletGenerator.waveletDecomposition``
* ``WaveletGenerator.waveletDecompositionFull``
* ``LoadPinkNoise.coarseWavelet``
* ``wavelet_io.convert_npy_to_zarr``

The stimulus movie is converted into separate coarse and fine binary arrays:

``<movie>_downsampled.npy``

``<movie>_coarse_downsampled.npy``

The current binary conversion uses the first channel of each video frame and a
threshold of ``> 100``.  Frames are cropped according to ``Visual Coverage`` and
``Analysis Coverage``, resized to the requested grid, and written directly to
disk as a memory-mapped ``.npy`` file to avoid holding the entire movie in RAM.
The fine array uses ``NY x NX``.  The coarse array uses
``coarse_ny x coarse_nx``.

Coarse wavelets:

The GUI computes two phases for coarse receptive-field analysis:

* ``dwt_videodata_0.npy`` for the real/cosine phase
* ``dwt_videodata_1.npy`` for the imaginary/sine phase

These are temporary intermediate files generated directly from the coarse
library and coarse stimulus grid.  ``LoadPinkNoise.coarseWavelet`` computes the
complex power component and saves the durable cache:

``dwt_downsampled_videodata.npy``

This cache contains three arrays on the first axis:

* index 0: real wavelet response
* index 1: imaginary wavelet response
* index 2: complex/amplitude-like response used for RF correlation

The temporary real and imaginary phase files are deleted after the durable coarse
cache is written.

Full-model wavelets:

The GUI also prepares full-resolution wavelets for the full model:

* ``dwt_videodata2_r.npy`` or ``dwt_videodata2_r.zarr``
* ``dwt_videodata2_i.npy`` or ``dwt_videodata2_i.zarr``

These arrays have shape:

``(n_frames, nx, ny, n_orientations, n_sigmas_full_model, n_frequencies)``

If Zarr is selected, temporary full-model ``.npy`` files are first written in the
recovery folder, converted to compressed Zarr, and then deleted.

Important assumptions:

* The movie frame count should match the configured ``Number of Frames`` used by
  downstream analysis.
* The downsampled movie is binary, not grayscale.
* The coarse RF grid is currently derived as 20 percent of the configured
  ``NX x NY`` grid, matching the paper's default ``27 x 11`` grid when
  ``NX x NY`` is ``135 x 54``.
* The coarse decomposition refuses independent-frequency libraries so coarse RF
  analysis cannot accidentally use only frequency index 0.


Stage 3: Neural Time Alignment
------------------------------

GUI behavior:

* If ``Spks Path`` points to an existing ``spikes.npy``, the GUI loads it and
  loads sibling ``pos.npy`` from the same folder.
* If ``Spks Path`` is blank, ``None``, or ``null``, the GUI runs the workflow
  alignment code and saves reusable aligned files beside the experiment data.

Primary code:

* ``zebraGUI.plot_data``
* ``time_alignment.load_aligned_spikes``
* ``time_alignment.load_two_photon_spikes``
* ``time_alignment.align_ephys_data``
* ``LoadPinkNoise.loadSPKMesoscope`` for the two-photon path

Saved aligned files:

* ``<experiment folder>/spikes.npy``
* ``<experiment folder>/pos.npy``

The GUI then updates ``Spks Path`` to the saved ``spikes.npy`` so the next run
can reuse the aligned data rather than repeating alignment.

Two-photon path:

The two-photon path delegates to the suite2p/Cortex Lab helper code through
``LoadPinkNoise.loadSPKMesoscope``.  Positions are corrected with
``correctNeuronPos`` when appropriate, using the configured microscope
resolution.

Electrophysiology path:

The ephys path reads a pickle file containing units, extracts DIO photodiode
events, detects trial onset/offset edges, repairs missing frame edges when
possible, and bins spike times into stimulus-frame bins.  The output activity is
stored as firing rate in Hz per frame bin.  Ephys neuron positions preserve up to
three coordinates, so ``neuron_pos`` may be ``(n_neurons, 3)``.

Important assumptions:

* ``spikes`` must be frame-aligned to the same stimulus movie used for wavelet
  decomposition.
* Trials must have consistent frame counts after photodiode-edge repair. Frame drops are handled based no this assumption.
* For SEM error bars, the trial axis is axis 0 of ``spikes``.


Stage 4: Coarse Receptive-Field Analysis
----------------------------------------

GUI button: ``Run Coarse RF Analysis``

Primary code:

* ``zebraGUI.plot_data``
* ``Analysis_Utils.repetability_trial3``
* ``Analysis_Utils.compute_skewness_neurons``
* ``Analysis_Utils.PearsonCorrelationPinkNoise``
* ``Analysis_Utils.PlotTuningCurve``

The GUI first computes two quality-control summaries:

* Repeatability: trial-to-trial response consistency.
* Skewness: skewness of the trial-averaged neural activity.

The current display mask treats neurons with repeatability >= 0.2 and skewness
<= 20 as higher-confidence points.  In the population retinotopy maps, neurons
outside the mask are shown with lower alpha rather than removed.

The RF estimate is computed by correlating each candidate stimulus feature time
course with each neuron's trial-averaged response.  In code:

* the stimulus wavelet array is reshaped to ``(time, features)``;
* neural responses are averaged across trials to ``(time, neurons)``;
* chunked Pearson correlation is computed between every feature and neuron;
* the correlation vector for each neuron is reshaped to:

``(coarse_x, coarse_y, n_orientations, n_sigmas, n_frequencies)``

For each neuron, the preferred feature is the maximum absolute correlation over
that feature grid.  The selected indices are converted into interpretable units:

* azimuth in visual degrees
* elevation in visual degrees
* orientation in degrees
* size in visual degrees
* spatial frequency in cycles/degree

The orientation is corrected for anisotropic visual-field scaling using
``orientation_correction_for_stretches``.

Durable outputs:

* ``dwt_downsampled_videodata.npy`` is reused here but created earlier.
* ``plot_cache.pkl.gz`` stores rendered plot data and analysis state for reuse.
* If alignment was run, ``spikes.npy`` and ``pos.npy`` are saved in the experiment
  folder.


Displayed Coarse RF Plots
-------------------------

All-neurons tab:

``Neuron Layout``
    Each point is one neuron.  The x/y axes are neuron positions in microns.  If
    ``neuron_pos`` has three columns, the plot becomes 3D and includes a z axis.

``Population Retinotopy Maps``
    Each point is one neuron at its anatomical position.  Color encodes the
    preferred azimuth, elevation, orientation, or size selected from the maximum
    absolute RF correlation.  Alpha encodes the repeatability/skewness quality
    mask.

Individual-neuron tab:

``Spike Train``
    The x axis is frame index.  The y axis is trial-averaged neural activity.
    If SEM is enabled and there is more than one trial, error bars are standard
    error of the mean across trials:

    ``SEM = std(trials, ddof=1) / sqrt(n_trials)``

``Selected Neuron Tuning``
    The receptive-field image shows correlation over visual azimuth/elevation.
    Tuning panels show correlation against elevation, azimuth, orientation,
    size, and spatial frequency.  These are slices through the neuron's RF
    correlation tensor.


Stage 5: Simple Model
---------------------

GUI button: ``Run Simple Model Plots``

Primary code:

* ``zebraGUI.plot_run_model_outputs``
* ``Analysis_Utils.run_Model``
* ``Analysis_Utils._process_single_neuron``
* ``Analysis_Utils.GetNeuronVisresponse``

The simple model uses the coarse RF result to select one excitatory wavelet and
one inhibitory/alternative wavelet for the selected neuron.  It then derives
polar wavelet variables from the real and imaginary wavelet responses:

* ``rho = sqrt(real^2 + imaginary^2)``: wavelet amplitude
* ``phi = atan2(imaginary, real)``: wavelet phase
* ``dphi = diff(unwrap(phi))``: phase drift over time

The model estimates how neural activity depends on ``rho``, ``phi``, and
``dphi``.  ``GetNeuronVisresponse`` trains on the GUI's ``Train Trial Indices``,
evaluates on ``Test Trial Indices``, and reports metrics such as FEVE, explained
variance, and correlation.  If ``Test Trial Indices`` is ``auto``, all trials not
used for training are held out.  If ``Use Last Minute Holdout`` is true, the
model also evaluates the minute immediately after the training window when
enough frames are available.  The GUI stores the returned model payload in the
plot cache and in exportable figure payloads.

Displayed simple-model plots:

* Spike activity versus frame index.
* Cosine/real wavelet value versus frame index.
* Sine/imaginary wavelet value versus frame index.
* Amplitude ``rho`` versus frame index.
* Phase ``phi`` in radians versus frame index.
* Drift ``dphi`` versus frame index.
* A 3D trajectory in ``rho``, ``phi``, and ``dphi`` colored by spike activity.

Durable outputs:

* ``plot_cache.pkl.gz`` only.  The simple model does not write standalone model
  result ``.npy`` files.


Stage 6: Full Model
-------------------

GUI button: ``Run Full Model Plots``

Primary code:

* ``zebraGUI.plot_run_full_model_outputs``
* ``Analysis_Utils.run_Full_Model``
* ``Analysis_Utils.GetNeuronVisresponse``

The full model starts from the coarse RF coordinates, then searches at full
stimulus resolution around the coarse position.  It refines:

* x/y position
* orientation
* size
* spatial frequency

It first computes Pearson correlations for the real and imaginary phases
separately inside the local full-resolution search window, then selects the
phase and feature with the largest absolute correlation.  It alternates between
position search and orientation/size/frequency search with the same
phase-specific criterion.  Then it extracts the real and imaginary wavelet time
courses at that refined feature and computes:

* amplitude ``rho``
* phase ``phi``
* drift ``dphi``

The same nonlinear response machinery is then used to predict neural activity.

Displayed full-model plots include:

* local orientation/size/frequency correlation searches;
* local x/y correlation searches;
* RF/tuning summaries for azimuth, elevation, orientation, size, frequency,
  amplitude, phase, and drift;
* time-series diagnostics comparing spike activity, wavelet variables, and model
  prediction.

Durable outputs:

The full model writes files in:

``<Full Model Save Path>/model_results/``

The filenames are:

* ``RPdp_predictions_<n_orientations>_noneigh_c_smoothpos_5.npy``
* ``RPdp_nonlinparams_<n_orientations>_noneigh_c_smoothpos_5.npy``
* ``RPdp_rhophiparams_<n_orientations>_noneigh_c_smoothpos_5.npy``
* ``RPdp_metrics_<n_orientations>_noneigh_c_smoothpos_5.npy``
* ``RPdp_os_<n_orientations>_noneigh_c_smoothpos_5.npy``
* ``RPdp_params_<n_orientations>_noneigh_c_smoothpos_5.npy``
* ``interpolators_5.pkl``

It also updates ``plot_cache.pkl.gz``.


Paper Reproduction Status
-------------------------

The current implementation is intended to make the paper-style WavEn path
explicit while still preserving practical GUI behavior.

Implemented as direct paper reproduction:

* separate coarse and fine Gabor libraries;
* coarse size/frequency coupling;
* fine independent size and frequency axes;
* two phases, real/cosine and imaginary/sine;
* coarse phase pooling as ``real^2 + imaginary^2``;
* coarse RF selection by Pearson correlation;
* fine local refinement initialized from the coarse RF;
* phase-specific fine search using the maximum absolute correlation across the
  two phases;
* explicit train/test repeat selection and optional last-minute holdout.

Implementation adaptations still present:

* the coarse grid is derived as 20 percent of ``NX x NY`` rather than hard-coded
  to ``27 x 11``.  With the paper default ``135 x 54`` grid, this gives
  ``27 x 11``.
* the stimulus movie is thresholded into a binary movie during downsampling;
* the GUI exposes configurable sigma/frequency lists rather than forcing the
  exact paper values;
* sign-map and visual-area segmentation helpers exist, but a full watershed
  visual-area workflow is not the central GUI path;
* external Zebra stimulus generation remains outside this repository.


Plot Cache and Exports
----------------------

Primary code:

* ``zebraGUI._put_cached_entry``
* ``zebraGUI._figure_records``
* ``zebraGUI._export_figure_record``

The plot cache is a compressed pickle:

``plot_cache.pkl.gz``

If ``Plot Cache Path`` is blank, the GUI places it in ``Full Model Save Path``.
If that is blank too, the GUI uses the movie folder.

The cache stores:

* analysis state needed by later plot/model buttons;
* PNG snapshots of figures;
* export payloads such as spike arrays, RF arrays, retinotopy arrays, and model
  outputs;
* extracted Matplotlib artist data;
* SEM captions when present.

The cache fingerprint includes GUI parameters, Gabor parameters, selected neuron,
and the SEM setting.  This prevents plots generated with different settings from
being silently reused.

Export buttons create per-graph bundles.  Each graph export includes:

* ``.png``: current visual image
* ``.svg``: vector plot
* ``_data.npz``: reusable numeric arrays when extractable
* ``_data.pkl``: richer Python object bundle
* ``_figure.pkl``: Matplotlib figure object when pickleable
* ``_manifest.json``: file list, axes metadata, and array keys

The export is meant to preserve both the display state and the reusable data
needed for downstream inspection.


File Lifecycle
--------------

Files intended to persist between steps:

* coarse and fine Gabor libraries ``.npy`` or ``.zarr``
* ``<movie>_downsampled.npy``
* ``<movie>_coarse_downsampled.npy``
* ``dwt_downsampled_videodata.npy``
* full-model wavelets ``dwt_videodata2_r/i.npy`` or ``.zarr``
* aligned ``spikes.npy`` and ``pos.npy``
* ``plot_cache.pkl.gz``
* full-model files in ``model_results/``
* explicit user exports

Temporary files removed after successful completion:

* recovery checkpoint folders
* ``dwt_videodata_0.npy``
* ``dwt_videodata_1.npy``
* low-RAM coarse scratch files:
  ``dwt_r_downsampled.mmap``, ``dwt_i_downsampled.mmap``,
  ``dwt_c_downsampled.mmap``
* temporary full-model NPY files used only for Zarr conversion
* ``plot_cache.pkl.gz.tmp`` after it is atomically replaced by
  ``plot_cache.pkl.gz``

Recovery checkpoint folders are intentionally kept when a task fails, because
they contain the latest completed step and are useful for debugging.


Known Interpretation Notes
--------------------------

The RF step is a correlation screen, not a causal model.  A neuron's preferred
feature is the feature with the largest absolute correlation between the
wavelet-derived stimulus time course and the trial-averaged neural response.

The simple and full models are fitted after the RF screen.  They depend on the
selected wavelet feature and on the train/test trial split.  Model outputs should
therefore be interpreted as feature-conditioned predictive fits, not as a
complete search over all possible stimulus explanations.

SEM error bars are across trials.  They are only plotted when there are at least
two trials; with one trial, SEM is undefined and no SEM caption is shown.

For ephys data, activity values are firing rates in Hz per stimulus-frame bin.
For two-photon data, activity units depend on the upstream suite2p/alignment
output and should be treated as arbitrary activity units unless the preprocessing
pipeline defines a physical calibration.


Where to Audit or Modify
------------------------

Use these code locations when checking or changing specific behavior:

* GUI button flow and file lifecycle: ``src/waven/app/gui.py``
* Workflow-specific parameter defaults: ``src/waven/config.py``
* Gabor construction: ``src/waven/wavelets/filters.py``
* Coarse RF cache creation: ``src/waven/stimulus/wavelet_cache.py``
* Alignment: ``src/waven/time_alignment.py``
* RF/tuning/model math: ``src/waven/analysis/``
* Zarr conversion: ``src/waven/storage/wavelet_zarr.py``
