# Restore GUI Settings with `pipeline_config.json`

`pipeline_config.json` is a backup and restore file for a working GUI session.
The GUI is the normal place to enter an experiment. Use this file when you want
to reopen the same paths, dropdowns, checkboxes, Gabor settings, selected cache
versions, and export choices later.

If you are setting up a new experiment, start with [First GUI Analysis](../tutorials/first-gui-analysis.md)
and [GUI Onboarding](../tutorials/gui-onboarding.md). If you prefer to prepare a
JSON file by hand, use the mapping below; every description has the same
meaning as its GUI control.

## Save and load from the GUI

1. Configure the session in the GUI and verify the selected cache version.
2. Click **Save `pipeline_config.json`** and choose a destination.
3. In a later session, click **Load `pipeline_config.json`**.
4. Review the restored paths before starting work, especially the movie folder,
   raw/neural source choice, and active cache versions.

Loading changes settings for future actions. It does not modify a background
task that has already captured its settings.

## What the file stores

```json
{
  "workflow": "ephys",
  "gui": {
    "active_coarse_cache_path": "{PROJECT_ROOT}/my_experiment/cache/wavelets/coarse/cache-20260726-153012",
    "active_full_model_cache_path": "",
    "sampling_mode": "Target degrees/pixel",
    "target_degrees_per_pixel": "0.75",
    "prepare_run_model_cache": false,
    "prepare_full_model_cache": false,
    "export_path": "{PROJECT_ROOT}/my_experiment/output/exports"
  },
  "gabor_param": {
    "N_thetas": "12",
    "Sigmas": "[2, 4, 8]",
    "Frequencies": "[0.02, 0.05]",
    "Phases": "[0, 90]"
  },
  "common": {
    "Project Root": "{PROJECT_ROOT}/my_experiment",
    "Movie Path": "{PROJECT_ROOT}/my_experiment/input/stimulus_movie",
    "Dir": "{PROJECT_ROOT}/my_experiment/input/raw_data",
    "Spks Path": "{PROJECT_ROOT}/my_experiment/input/neural_cache",
    "Path Directory": "{PROJECT_ROOT}/my_experiment/cache/wavelets/coarse",
    "Full Model Wavelet Path": "{PROJECT_ROOT}/my_experiment/cache/wavelets/full",
    "Visual Coverage": "[-69, 69, 56, -56]",
    "Analysis Coverage": "[-69, 69, 56, -56]"
  },
  "ephys": {
    "Sampling Rate (samples / sec)": "30000",
    "Photodiode Port": "3"
  }
}
```

| JSON section | What it restores | Same GUI meaning |
| --- | --- | --- |
| `workflow` | `"2p"` or `"ephys"` | Chooses the session-specific neural-data fields. |
| `common` | Paths, visual coverage, model defaults, and other shared fields | The values entered across System Configuration, Stimulus Video, Cache folders, and Session Setup. |
| `gabor_param` | Orientation, sigma, frequency, and phase fields | The Gabor Filters values used to define the feature bank. |
| `gui` | Dropdowns, checkboxes, sampling mode, active cache versions, export choices, and hardware preferences | The GUI state surrounding the typed fields. |
| `two_photon` or `ephys` | Modality-specific acquisition fields | The values shown when that workflow is selected. |

## Field mapping

These descriptions intentionally match [GUI Onboarding](../tutorials/gui-onboarding.md).

| GUI control | JSON location | Type | Description |
| --- | --- | --- | --- |
| Project Root Folder | `common["Project Root"]` | Directory | The root for `input/`, `cache/`, and `output/`. Set this before selecting generated-cache folders. |
| Stimulus Movie Folder | `common["Movie Path"]` | Directory | This is the source of truth for time and pixel dimensions. Keep the folder unambiguous. |
| Visual Field Coverage | `common["Visual Coverage"]` | Four-number text list | The angular extent of the full movie. Use the stimulus display's measured bounds. |
| Analysis Field Coverage | `common["Analysis Coverage"]` | Four-number text list | Limits the visual field analysed. Changing it changes dependent stimulus and wavelet caches. |
| Raw Data Folder | `common["Dir"]` | Directory | Required only for fresh alignment. Use the layouts described in First GUI Analysis. |
| Neural Cache Folder | `common["Spks Path"]` | Directory | Receives fresh `spikes`/`pos` output or contains the existing pair to validate. |
| Coarse RF Cache Library Folder | `common["Path Directory"]` | Directory | The parent folder for Coarse RF products. Leave at the project default unless cache storage belongs elsewhere. |
| Full-Model Cache Library Folder | `common["Full Model Wavelet Path"]` | Directory | Used only for optional Full Model phase products. |
| Active Coarse RF Cache | `gui["active_coarse_cache_path"]` | Path | Choose the cache version used by Run Coarse RF Analysis. The GUI validates its movie, crop, grid, and Gabor provenance before analysis. |
| Active Full Model Cache | `gui["active_full_model_cache_path"]` | Path | Choose the phase-cache version used by optional Full Model inspection. |
| Sampling mode | `gui["sampling_mode"]` and its associated setting | String + number | Recommended modes are Target degrees/pixel when you know resolution or Retain up to cpd when you know the highest meaningful frequency. |
| Orientation Count, Sigmas, Frequencies, Phases | `gabor_param` | Integer and text lists | Define the feature bank. Use the GUI recommender as the starting point when possible. |
| Fresh / raw data versus Continue / existing cache | `gui["neural_source"]` | `"data_dir"` or `"spks_path"` | Choose raw alignment for a new recording or cache validation for an already aligned pair. |
| Resolution / Number of Planes | `two_photon` | Number + integer | Required for fresh 2-photon data. The requested number of Suite2p planes must exist. |
| Sampling Rate / Photodiode Port | `ephys` | Number + integer | Required for fresh ephys data. Use Find ports, then confirm the port from wiring. |
| Prepare optional model caches | `gui["prepare_run_model_cache"]`, `gui["prepare_full_model_cache"]` | Booleans | Select only when you want the corresponding optional model graphs. |
| Export folder and selections | `gui["export_path"]`, `gui["export_files"]`, `gui["export_selections"]` | Path + objects | Set once at the top of Export. The GUI does not ask again for a save location. |

## Manual editing

Manual editing is supported, but use it sparingly. Save once from the GUI and
edit that generated file rather than building a schema from memory.

- Paths may use `{PROJECT_ROOT}`; it resolves to the repository folder that
  contains `ui.py`.
- Most text-entry values remain strings because they are restored into GUI text
  fields. WavEn parses lists, numbers, booleans, and paths when an action runs.
- Omit values you do not need; the GUI supplies its normal defaults.
- If a saved active cache path no longer belongs to the selected cache library,
  the GUI falls back safely to that library's top-level cache.

For cache compatibility, versioning, and storage planning, see
[Reusable Caches](../reference/storage.md). For the GUI workflow, return to
[GUI Onboarding](../tutorials/gui-onboarding.md).
