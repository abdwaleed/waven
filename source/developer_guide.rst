waven Developer Guide
=====================

This guide describes the package layout, naming conventions, and the intended
ownership boundaries between modules.  It is written for developers who need to
extend or debug the codebase.

Package layout
--------------

``waven.app``
    Interactive application entry points.  ``waven.app.gui`` owns the Tkinter /
    CustomTkinter GUI, button callbacks, terminal output, task cancellation,
    recovery checkpoints, and embedded matplotlib views.

``waven.pipeline``
    Programmatic orchestration layer.  This is the best entry point for scripts,
    notebooks, tests, and batch processing because it calls the same scientific
    stages without GUI state.

``waven.config``
    Typed configuration models and parsing helpers.  This module converts JSON
    and GUI strings into strongly shaped paths, tuples, booleans, and numeric
    values used by the pipeline.

``waven.wavelets``
    Gabor filter-bank construction and wavelet decomposition.  ``filters`` builds
    Gabor libraries; ``decomposition`` down-samples movies and writes coarse or
    full-resolution wavelet coefficient arrays.

``waven.stimulus``
    Stimulus wavelet loading and cache construction.  ``wavelet_cache`` builds
    the coarse RF cache consumed by the fast RF/model code.  ``full_model`` loads
    full-resolution wavelets for the high-granularity model path.

``waven.analysis``
    Receptive-field analysis, nonlinear model fitting, tuning curves, trial
    statistics, and plotting-oriented numerical utilities.

``waven.data`` and ``waven.time_alignment``
    Neural data loading and alignment.  These modules bridge suite2p or ephys
    acquisition formats into frame-aligned arrays used by analysis.

``waven.storage``
    Storage adapters for large arrays.  ``array_store`` abstracts NumPy/Zarr
    loading.  ``wavelet_zarr`` converts full-model wavelet NPY outputs into
    chunked Zarr when conversion is needed.

``waven.runtime``
    Runtime services that are not scientific algorithms: hardware-aware chunk
    sizing, cooperative cancellation, progress formatting, and keep-awake support
    for long GUI tasks.

``waven.gui_support``
    Small reusable helpers for GUI labels, file-size display, tooltips, path
    naming, and parsing.

``waven.suite2p`` and ``waven.suite_ephys``
    Vendor/workflow-specific loader utilities kept in their own namespaces.

Compatibility modules
---------------------

These modules remain so existing notebooks and scripts continue to run, but new
code should prefer the lowercase domain modules above:

``waven.zebraGUI``
    Compatibility wrapper for ``waven.app.gui``.

``waven.WaveletGenerator``
    Compatibility wrapper for ``waven.wavelets``.

``waven.LoadPinkNoise``
    Compatibility wrapper for ``waven.stimulus`` and ``waven.data.neural``.

``waven.Analysis_Utils``
    Compatibility wrapper for ``waven.analysis``.

``waven.performance``, ``waven.power``, ``waven.task_control``
    Compatibility wrappers for ``waven.runtime`` modules.

``waven.wavelet_io`` and ``waven.zarr_compat``
    Compatibility wrappers for ``waven.storage`` modules.

Naming conventions
------------------

Use lowercase module names and domain-focused package names.  Prefer explicit
names such as ``storage.wavelet_zarr`` or ``runtime.task_control`` over broad
legacy names.  Public functions should describe their action and data domain:
``prepare_stimulus_wavelets``, ``load_coarse_wavelets``, ``run_rf_analysis``.

Private helpers should start with ``_`` and stay inside the module that owns
their invariants.  If a private helper becomes useful in multiple modules, move
it into an explicit domain module rather than importing across unrelated files.

Large-file lifecycle
--------------------

Wavelet and Gabor outputs can be larger than available RAM or disk.  Code that
creates large files should follow these rules:

* Stream to disk or Zarr when possible instead of materializing duplicate arrays.
* Register partial outputs with the GUI cancel cleanup registry before writing.
* Delete intermediate NPY files after a successful Zarr conversion when they are
  no longer a durable user-selected output.
* Keep legacy loaders able to read existing NPY paths and sibling Zarr stores.

Progress and cancellation
-------------------------

Long-running loops should accept an optional ``cancel_event`` and call
``runtime.task_control.check_cancelled`` at chunk, sigma, frequency, or frame
boundaries.  Progress text should use
``runtime.task_control.progress_message`` so the GUI terminal consistently shows
percent complete, elapsed time, ETA, and speed.

Documentation standard
----------------------

For industry-style documentation, each public function should have:

* a one-line summary that starts with an imperative verb or clear noun phrase;
* parameter descriptions with expected shape/unit when arrays are involved;
* return value descriptions;
* raised exceptions for user-actionable failures;
* notes about disk outputs and side effects;
* examples for pipeline entry points and loaders.

Use the code reference as a checklist.  When a function is listed there without
a docstring-derived summary, it should be a priority for docstring cleanup.

