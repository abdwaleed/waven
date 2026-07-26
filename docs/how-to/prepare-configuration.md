# `pipeline_config.json` and scientific settings

`pipeline_config.json` is an optional, portable record of GUI inputs. The GUI can launch without it; missing keys use the normal GUI defaults and each action validates the fields it needs. The safest workflow is:

1. Fill out one working session in the GUI.
2. Click **Save pipeline_config.json** in System Configuration.
3. Store that file with the experiment or copy it as a starting point for the next one.
4. Use **Load pipeline_config.json** to restore it later. Save and load stay enabled while a long task is running.

The checked-in root-level file is an example of the current save format. JSON paths can use `{PROJECT_ROOT}`; Waven resolves it relative to the repository directory that contains `ui.py`.

```json
{
  "workflow": "ephys",
  "gui": {"downsample_percent": 11, "wavelet_format": "zarr"},
  "gabor_param": {"N_thetas": "12", "Sigmas": "[2, 4, 8]"},
  "common": {"Project Root": "{PROJECT_ROOT}/your_experiment"},
  "ephys": {"Sampling Rate (samples / sec)": "30000"}
}
```

Values may be stored as strings because they populate text fields. Waven parses numbers, lists, tuples, booleans, and paths when the relevant action runs.

## Top-level sections

| Section | Type | Required? | Description |
| --- | --- | --- | --- |
| `workflow` | `"2p"` or `"ephys"` | Optional at launch | Selects workflow-specific fields. |
| `gui` | object | Optional | Saved GUI choices: sampling, cache format, optional model flags, hardware controls, and export file types. |
| `gabor_param` | object | Required before kernel preparation | Orientation, scale, frequency, phase, and kernel-cache folder inputs. |
| `common` | object | Required before a complete analysis | Shared paths, coverage, training/holdout options, and full-model settings. |
| `two_photon` / `ephys` | object | Required for fresh data in the chosen workflow | Workflow-specific acquisition values. |

Do not add a `wavelet_backend` field: the GUI is convolution-only. Older files may load compatibility values, but the current GUI neither displays nor uses a legacy backend choice.

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

The movie metadata, Analysis Coverage, and selected sampling mode determine the analysis grid. Never enter movie width, height, frame rate, `NX`, or `NY` in the GUI: Waven derives them.

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
| Coarse Library Path | directory | Yes | Coarse kernel/cache folder setting. | Keep it under `cache/gabor/coarse`. |
| Fine Library Path | directory | Full-model preparation | Full kernel/cache folder setting. | Keep it under `cache/gabor/full`. |

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
| `performance` | object | Optional hardware/speed controls described below. |
| `suite2p_subject_dirs`, `suite2p_output_dir` | strings | Optional two-photon discovery overrides. Leave blank unless needed. |
| `export_files` | object | Selected `png`, `svg`, and `data_pickle` output types. |

## Optional performance controls (`gui.performance`)

These choices change scheduling or precision, not scientific parameter axes. They are available under **Performance & Hardware (advanced)** in System Configuration.

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
