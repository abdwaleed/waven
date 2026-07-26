# `pipeline_config.json` and scientific settings

`pipeline_config.json` is an optional, portable snapshot of the GUI. The GUI
can launch without it: missing keys use normal defaults and each action
validates what it needs. Fill out a working session in the GUI, save it, and
load it later to restore paths, checkboxes, dropdowns, and export choices.

1. Fill out one working session in the GUI. Feel free to cross-reference the input descriptions below as needed.
2. Click **Save pipeline_config.json** in Session Configuration.
3. Store that file as a replacement for the dummy `pipeline_config.json` downloaded with this repo.
4. Use **Load pipeline_config.json** to restore it later.

JSON paths can use `{PROJECT_ROOT}` to avoid hardcoding absolute paths. WavEn
resolves it relative to the repository directory that contains `ui.py`. If you
edit a file manually, values may be strings because they populate text fields;
WavEn parses numbers, lists, tuples, booleans, and paths when an action runs.
The file is organized as follows:

```json
{
  "workflow": "ephys",
  "gui": {
    "coarse_rf_frequency_mode": "coupled",
    "neural_source": "data_dir",
    "downsample_percent": 11,
    "downsample_format": "zarr",
    "neural_cache_format": "zarr",
    "wavelet_format": "zarr",
    "prepare_run_model_cache": false,
    "prepare_full_model_cache": false,
    "force_2d_graphs": false,
    "run_model_on_inspect": false,
    "run_full_model_on_inspect": false,
    "suite2p_subject_dirs": "",
    "suite2p_output_dir": "",
    "performance": {
      "ram_acceleration_cache": false,
      "multi_gpu": false,
      "torch_compile": false,
      "amp": false
    },
    "export_path": "{PROJECT_ROOT}/your_experiment/output/exports",
    "export_files": {"png": true, "svg": true, "data_pickle": false},
    "export_selections": {"current_all": {}, "all_individual": {}},
    "repeat_individual_export": false,
    "guided_export": {"all_neurons": false, "individual_neurons": false}
  },
  "gabor_param": {
    "N_thetas": "12",
    "Sigmas": "[2, 4, 6, 8, 12, 16]",
    "Frequencies": "[0.0024, 0.0049, 0.0098, 0.0196, 0.0391, 0.0783, 0.1223, 0.1957]",
    "Phases": "[0, 90]"
  },
  "common": {
    "Project Root": "{PROJECT_ROOT}/your_experiment",
    "Experiment Info": "('CnL43', '2026-05-23', 3)",
    "Block End": "0",
    "Train Trial Indices": "[0, 2]",
    "Test Trial Indices": "auto",
    "Use Last Minute Holdout": "False",
    "Visual Coverage": "[-69, 69, 56, -56]",
    "Analysis Coverage": "[-69, 69, 56, -56]",
    "Sigmas Full Model": "[2, 3, 4, 6, 8, 12, 16]",
    "Dir": "{PROJECT_ROOT}/your_experiment/input/raw_data",
    "Movie Path": "{PROJECT_ROOT}/your_experiment/input/stimulus_movie",
    "Spks Path": "{PROJECT_ROOT}/your_experiment/input/neural_cache",
    "Path Directory": "{PROJECT_ROOT}/your_experiment/cache/wavelets/coarse",
    "Full Model Wavelet Path": "{PROJECT_ROOT}/your_experiment/cache/wavelets/full",
    "Full Model Save Path": "{PROJECT_ROOT}/your_experiment/output/models",
    "Plot Cache Path": "{PROJECT_ROOT}/your_experiment/output/plots",
    "Recovery Cache Directory": "{PROJECT_ROOT}/your_experiment/output/recovery_cache",
    "Model Fit Minutes": "5"
  },
  "two_photon": {
    "Resolution": "1.3671",
    "Number of Planes": "1"
  },
  "ephys": {
    "Sampling Rate (samples / sec)": "30000",
    "Photodiode Port": "3"
  }
}
```

Use forward slashes and `{PROJECT_ROOT}` in `pipeline_config.json`:

```json
{
  "common": {
    "Project Root": "{PROJECT_ROOT}/your_experiment",
    "Movie Path": "{PROJECT_ROOT}/your_experiment/input/stimulus_movie",
    "Spks Path": "{PROJECT_ROOT}/your_experiment/input/neural_cache"
  }
}
```

At launch, Waven replaces `{PROJECT_ROOT}` with the repository directory that
contains `ui.py`.

## Top-level sections

| Section | Type | Description |
| --- | --- | --- |
| `workflow` | `"2p"` or `"ephys"` | Selects workflow-specific fields. |
| `gui` | object | Saved GUI choices: sampling, cache format, optional model flags, hardware controls, export folder, graph selections, and guided-export selections. |
| `gabor_param` | object | Orientation, scale, frequency, and phase inputs. Kernel folders follow the project layout. |
| `common` | object | Shared paths, coverage, training/holdout options, and full-model settings. |
| `two_photon` / `ephys` | object | Workflow-specific acquisition values. |

## Shared paths and acquisition inputs (`common`)

| Field | Type | Required? | Description |
| --- | --- | --- | --- |
| Project Root | directory | Yes | Owns the conventional experiment tree. |
| Dir | directory | Fresh alignment only | Raw acquisition root or conventional reference folder. |
| Movie Path | directory | Yes | Contains the stimulus movie used to derive metadata. |
| Spks Path | directory | Continue cache only; fresh output location | Existing or generated aligned neural cache pair. |
| Path Directory | directory | Yes for Coarse RF work | Coarse stimulus/wavelet cache folder. |
| Full Model Wavelet Path | directory | Only when preparing model phase caches | Folder for Run Model/Full Model phase products. |
| Full Model Save Path | directory | Optional | Intended model-result location. |
| Plot Cache Path | directory | Optional | Folder for the GUI plot cache. |
| Recovery Cache Directory | directory | Recommended | Checkpoint location for long tasks. |
| Experiment Info | tuple/list text | Fresh alignment when the workflow uses it | Experiment identifier, for example `('mouse01', '2026-05-23', 3)`. |
| Block End | integer | Workflow-dependent | Last acquisition block considered during alignment. |
| Train Trial Indices | `"auto"` or zero-based list | Optional | Model fitting trial selection. |
| Test Trial Indices | `"auto"` or zero-based list | Optional | Held-out trial selection. |
| Use Last Minute Holdout | boolean | Optional | Reserves the final minute for model evaluation when enabled. |

### Coverage

| Field | Type / units | Required? | Description |
| --- | --- | --- | --- |
| Visual Coverage | `[left, right, top, bottom]` in visual degrees | Yes | Angular bounds of the full movie. |
| Analysis Coverage | same four-number order | Yes | Angular crop to analyse; must lie within Visual Coverage. |

For example, a movie spanning azimuth `-69` to `69` degrees and elevation `56` to `-56` degrees is written as `[-69, 69, 56, -56]`. Preserve this order. Changing either coverage invalidates dependent stimulus and wavelet caches.

## Sampling: choose the grid before choosing filters

The movie metadata, Analysis Coverage, and selected sampling mode determine the analysis grid. Waven derives movie width, height, and frame rate as metadata from the movie.

Please note that it is good practice for your monitor's refresh rate (Hz) to match or be an integer multiple of your video's frame rate. Violating this rule may affect your results, though the pipeline will run.

| GUI setting (`gui`) | Type | Use when | Reasoning |
| --- | --- | --- | --- |
| `sampling_mode: "Target degrees/pixel"` + `target_degrees_per_pixel` | positive degrees/pixel | You know the visual-angle resolution required. | Smaller degrees/pixel means more pixels and a higher Nyquist limit, with larger caches. |
| `sampling_mode: "Retain up to cpd"` + `maximum_spatial_frequency_cpd` | positive cycles/degree | You know the highest meaningful stimulus frequency. | Waven requests a grid with a 1.5× margin above that frequency's Nyquist boundary. |
| `sampling_mode: "Compatibility percent"` + `downsample_percent` | 1–100 | You need a direct legacy-style sampling fraction. | Useful for matching an earlier grid, but not a physical resolution target. |

Use the GUI's displayed Nyquist value to check the choice. If the highest frequency of interest approaches Nyquist, increase sampling or lower the frequency list. A larger grid raises storage and convolution roughly in proportion to its number of pixels.

## Gabor settings (`gabor_param`)

| Field | Type / units | Required? | What it controls | Practical starting rule |
| --- | --- | --- | --- | --- |
| N_thetas | positive integer | Yes | Orientation samples on `[0°, 180°)`. | `12` gives 15° spacing; use more only for a finer orientation hypothesis. |
| Sigmas | list of analysis pixels | Yes | Coarse RF Gaussian-envelope sizes. | Use a short increasing list covering expected RF scales. |
| Frequencies | list of cycles per analysis pixel | Full Model only | Full-model spatial-frequency samples. | Keep values below the grid's Nyquist limit; use a sparse roughly logarithmic list for broad ranges. |
| Phases | list of degrees | Yes | Gabor phase offsets. | `[0, 90]` provides phase-quadrature information for real/imaginary model products. |

`Sigmas Full Model` lives in `common`. It is a list of analysis pixels used for the full-resolution phase bank and is merged with the coarse sigma values. Every extra orientation, sigma, frequency, or phase expands the feature space; start with the smallest scientifically defensible set and refine after a pilot run.

## GUI state (`gui`)

| Field | Type | Description |
| --- | --- | --- |
| `coarse_rf_frequency_mode` | `"coupled"` or `"frequency_list"` | Whether Coarse RF couples size/frequency or uses an independent frequency list. |
| `downsample_format` | `"npy"` or `"zarr"` | Format for the prepared stimulus cache. |
| `neural_source` | `"data_dir"` or `"spks_path"` | Fresh/raw alignment or an existing neural cache. |
| `neural_cache_format` | `"npy"` or `"zarr"` | Format written for fresh aligned neural data. |
| `wavelet_format` | `"npy"` or `"zarr"` | Preferred durable Coarse RF correlation-result storage; convolution phase/power products remain Zarr. |
| `run_model_on_inspect` | boolean | Opt-in Run Model curves after basic individual-neuron inspection. |
| `run_full_model_on_inspect` | boolean | Opt-in Full Model curves after basic inspection. |
| `prepare_run_model_cache`, `prepare_full_model_cache` | boolean | Whether Prepare Analysis Caches also prepares the corresponding optional phase pair. |
| `force_2d_graphs` | boolean | Use two-dimensional display variants where available. |
| `performance` | object | Optional hardware/speed controls described below. |
| `suite2p_subject_dirs`, `suite2p_output_dir` | strings | Optional two-photon discovery overrides. Leave blank unless needed. |
| `export_files` | object | Selected `png`, `svg`, and `data_pickle` output types. |
| `export_path` | path | Folder used for all graph exports; exports do not open a save-location dialog. |
| `export_selections` | object | Saved checked graph types for the All Neurons and Individual Neurons export sections. |
| `repeat_individual_export` | boolean | Repeats the selected individual-neuron graphs for every analyzed neuron. |
| `guided_export` | object | Whether Guided Pipeline exports the saved all-neuron and/or individual-neuron selections after analysis. |

## Optional performance controls (`gui.performance`)

These choices change scheduling or precision, not scientific parameter axes. They are available under **Performance & Hardware** in Session Configuration.

| Field | Default | Benefit | Use with care |
| --- | --- | --- | --- |
| `ram_acceleration_cache` | `false` | Retains safely sized, reusable model-phase and PSTH/STA inputs in RAM for later analysis. | Leave off if other software needs RAM; oversized arrays remain disk-backed either way. |
| `multi_gpu` | `false` | Batch-parallel convolution across compatible CUDA GPUs. | Enable only for comparable GPUs with sufficient free memory. |
| `torch_compile` | `false` | May accelerate repeated stable convolution kernels after a one-time warm-up. | Experimental; Waven falls back safely if compilation fails. |
| `amp` | `false` | Uses CUDA float16 autocast for wavelet convolution. | Faster on appropriate Tensor Core hardware; Coarse RF statistics retain their established precision. |

## Workflow-specific sections

| Workflow | Field | Type / units | Required? | Description |
| --- | --- | --- | --- | --- |
| `two_photon` | Resolution | positive micrometers/pixel | Fresh 2-photon | Spatial scale for neural positions. |
| `two_photon` | Number of Planes | positive integer | Fresh 2-photon | Number of imaging planes. |
| `ephys` | Sampling Rate (samples / sec) | positive Hz | Fresh ephys | Acquisition sampling rate, for example `30000`. |
| `ephys` | Photodiode Port | non-negative integer | Fresh ephys | Trodes digital port carrying the photodiode/TTL signal. |

See [Cache Storage and Disk Planning](../reference/storage.md) before preparing large phase products, and [GUI Workflow](../reference/gui.md) for the exact buttons that consume these values.
