# Prepare configuration

!!! note "Optional for the GUI"
    `pipeline_config.json` is not required to launch `python ui.py`. Missing
    keys and values written as `""` appear as blank GUI fields. The typed
    scripted pipeline still validates required values when it is constructed,
    and GUI actions validate the fields they consume when clicked.

The easiest starting point is `pipeline_config.json`. Most values in the example
file are strings because the same values can be loaded into the GUI text fields.
The parser then converts those strings into integers, floats, booleans, lists,
tuples, or paths.

## Project root placeholder

Use `{PROJECT_ROOT}` when you want paths to be portable across machines:

```json
"Project Root": "{PROJECT_ROOT}/your_experiment",
"Movie Path": "{PROJECT_ROOT}/your_experiment/input/stimulus_movie"
```

When `PipelineConfig.from_json()` loads the file, it replaces `{PROJECT_ROOT}`
with the current working directory of the Python process. In practice, run your
script from the repository or project folder you want to be treated as the root:

```bash
cd C:/path/to/waven_jul1
python my_analysis_script.py
```

Use forward slashes in JSON paths even on Windows. Python accepts them, and they
avoid escaping every backslash.

## Typing rules

| What you need | Safe value to type | Notes |
| --- | --- | --- |
| folder path | `"{PROJECT_ROOT}/your_experiment/cache/wavelets/coarse"` | Use quotes. GUI path fields are folders; the code discovers files inside the final folder. |
| integer | `"18000"` | Quotes are fine; the parser converts to `int`. |
| float | `"1.3671"` | Quotes are fine; the parser converts to `float`. |
| list of numbers | `"[2, 4, 8, 12]"` | Keep brackets and commas. |
| tuple/list with labels | `"('mouse01', '2026-05-23', 3)"` | Used by `Experiment Info`. |
| boolean | `"True"` or `"False"` | Also accepts `true`/`false`, `1`/`0`, `yes`/`no`. |
| intentionally missing optional path | `"None"` or `""` | Use only for optional fields such as `Spks Path`. |

Required input folders should contain the relevant input before you run
analysis. Output/cache folders can be missing; `waven` creates them when it
writes outputs.

## Project layout

The GUI uses a strict layout under `Project Root`. The most common tree is:

```text
your_experiment/
  input/raw_data/
  input/stimulus_movie/
  input/neural_cache/
  cache/gabor/coarse/
  cache/gabor/full/
  cache/gabor/kernels/
  cache/wavelets/coarse/
  cache/wavelets/full/
  output/plots/
  output/recovery_cache/
  output/models/
```

Path parameters point to these folders. If you browse to an external source
folder, the GUI stores a `.waven_reference.json` pointer inside the conventional
folder and keeps the GUI field pointed at the conventional folder.

## File structure

The JSON file has four main sections:

| Section | Required? | Meaning |
| --- | --- | --- |
| `workflow` | optional for GUI | `"2p"` for two-photon or `"ephys"` for electrophysiology. |
| `gui` | optional | Initial scale, backend, fresh/continue source, and output-format selectors. |
| `gabor_param` | optional for launch | Filter axes and coarse/full cache folders. Grid dimensions are read from movie metadata. |
| `common` | optional for launch | Stimulus, timing, coverage, and shared folder settings. |
| `two_photon` / `ephys` | optional for launch | Workflow-specific acquisition settings. |

The checked-in `pipeline_config.json` uses the exact schema written by the GUI's
**Save Configuration** button. Older `analysis`, `param_defaults`, `gabor`, and
`save_options` sections remain loadable for compatibility.

## `gui`

| Field | Accepted values | Effect |
| --- | --- | --- |
| `analysis_scale` | `"coarse"`, `"full"` | Chooses which scale-specific folders and parameters are visible. |
| `wavelet_backend` | `"legacy"`, `"convolution"` | Selects the decomposition implementation. |
| `neural_source` | `"data_dir"`, `"spks_path"` | `data_dir` means fresh/raw acquisition; `spks_path` means continue from an existing cache. |
| `downsample_percent` | number from 0 through 100 | Spatial percentage applied to source movie width and height. |
| format fields | `"npy"` or `"zarr"` | Initial cache formats for the corresponding stage. |

## `gabor_param`

These fields define the Gabor filters before they are applied to the movie.

| Field | Type to type | Units | File or dir? | Element meaning |
| --- | --- | --- | --- | --- |
| `N_thetas` | integer string, e.g. `"18"` | orientation bins | no | Number of orientations between 0 inclusive and 180 exclusive. `18` means 10-degree spacing. |
| `Sigmas` | list string, e.g. `"[2, 4, 8, 12]"` | analysis pixels | no | Gaussian envelope sizes for the coarse Gabor filters. Larger values mean broader filters. |
| `Frequencies` | list string, e.g. `"[0.02, 0.06, 0.1]"` | cycles per analysis pixel | no | Spatial-frequency bins for the fine/full model. Multiple values create the frequency axis. |
| `Phases` | list string, e.g. `"[0, 90]"` | degrees | no | Gabor phase offsets. The GUI converts degrees to radians immediately before Gabor-kernel construction. |
| `Coarse Library Path` | folder path | path | directory | Folder shown and used in coarse mode. |
| `Fine Library Path` | folder path | path | directory | Folder shown and used in full mode. |

## `common` paths

| Field | Type to type | File or dir? | Required? | Meaning |
| --- | --- | --- | --- | --- |
| `Project Root` | path string | directory | yes | Root that owns the strict `input/`, `cache/`, and `output/` tree. |
| `Dir` | path string | directory | yes | Conventional raw-data folder, usually `input/raw_data`. It may contain a `.waven_reference.json` pointing to external raw data. |
| `Path Directory` | path string | directory | yes | Coarse wavelet cache folder, usually `cache/wavelets/coarse`. |
| `Movie Path` | path string | directory | yes | Stimulus movie folder, usually `input/stimulus_movie`. Put exactly one compatible movie file in this folder. |
| `Spks Path` | path string | directory | yes | Neural cache folder, usually `input/neural_cache`. Exact `spikes`/`pos` names are preferred, but semantic stems such as `my_spikes.npy` and `cell_positions.npy` are accepted when unambiguous. |
| `Full Model Wavelet Path` | path string | directory | yes for full mode | Full-model wavelet folder, usually `cache/wavelets/full`. |
| `Full Model Save Path` | path string | directory | yes for full model outputs | Model output folder, usually `output/models`. |
| `Plot Cache Path` | path string | directory | optional | Plot cache folder, usually `output/plots`; the GUI writes `plot_cache.pkl.gz` inside it. |
| `Recovery Cache Directory` | path string | directory | optional | Checkpoint directory for resumable long GUI tasks, usually `output/recovery_cache`. |

## `common` identifiers and acquisition timing

| Field | Type to type | Units | Element meaning |
| --- | --- | --- | --- |
| `Experiment Info` | tuple/list string, e.g. `"('CnL43', '2026-05-23', 3)"` | labels | `(subject_or_mouse_id, date_string, experiment_number)`. The date is a label used in paths; use your lab's folder convention. |
| `Block End` | integer string | acquisition block index | Last block considered during alignment. Use `"0"` when your workflow does not need a later block. |
| `Train Trial Indices` | `"auto"` or list string | zero-based trial indices | Training trials for model fitting. Example `"[0, 2]"` means first and third retained trials. |
| `Test Trial Indices` | `"auto"` or list string | zero-based trial indices | Held-out trials. `"auto"` means every retained trial not used for training. |
| `Use Last Minute Holdout` | boolean string | yes/no | If true, model evaluation reserves the final minute of frames. |

## `common` spatial fields

| Field | Type to type | Units | Element meaning |
| --- | --- | --- | --- |
| `Visual Coverage` | four-number list string | visual degrees | `[azimuth_left, azimuth_right, elevation_top, elevation_bottom]` for the full stimulus field. Preserve this order. |
| `Analysis Coverage` | four-number list string | visual degrees | Same four-element order, but for the crop you want to analyze. It should lie inside `Visual Coverage`. |
| `Sigmas Full Model` | list string | analysis pixels | Full-model Gabor sizes. These are merged with `gabor_param.Sigmas` when building the fine library. |

`NX`, `NY`, coarse `Sigmas`, and `Frequencies` have one owner in the GUI:
`gabor_param`. Legacy duplicates in `common` are accepted by the typed parser but
are intentionally not displayed.

Coverage lists are easy to mistype. Keep the same left/right/top/bottom order in
both fields. If your full stimulus spans `-69` to `69` degrees azimuth and `56`
to `-56` degrees elevation, type:

```json
"Visual Coverage": "[-69, 69, 56, -56]"
```

If you only want to analyze the central visual field, type a smaller crop:

```json
"Analysis Coverage": "[-35, 35, 35, -35]"
```

## Workflow-specific sections

| Workflow | Section | Field | Type to type | Units | Meaning |
| --- | --- | --- | --- | --- | --- |
| `2p` | `two_photon` | `Resolution` | float string | micrometers per pixel | Imaging-plane spatial scale used for neuron positions. |
| `2p` | `two_photon` | `Number of Planes` | integer string | imaging planes | Number of imaging planes in the recording. |
| `ephys` | `ephys` | `Sampling Rate (samples / sec)` | float string | samples per second | Electrophysiology acquisition sampling rate, e.g. `"30000"`. |

Programmatic loading:

```python
from pathlib import Path

import waven

config = waven.PipelineConfig.from_json(Path("pipeline_config.json"))
```

## Shape consequences

Configuration fields are not just metadata. They directly determine array size:

| Axis | Comes from | Appears in |
| --- | --- | --- |
| `n_frames` | movie length and `Number of Frames` | downsampled movies, wavelets, spikes |
| `n_trials` | repeats discovered during neural alignment | `spikes` |
| `n_neurons` | Suite2p/ephys input | `spikes`, RF tensors, plots |
| `NX`, `NY` | config grid | fine library and full wavelets |
| coarse `nx`, `ny` | derived from `NX`, `NY` | coarse library, coarse wavelets, RF tensor |
| `n_orientations` | `N_thetas` | Gabor libraries, wavelets, RF tensor, OSI/gOSI |
| `n_sigmas` | `Sigmas` or `Sigmas Full Model` | Gabor libraries, wavelets, RF tensor |
| `n_frequencies` | `Frequencies` | fine library, full wavelets, RF tensor when present |

When a run is expensive, start by estimating the largest output. Full-model
wavelets are usually the limiting object because each phase is a dense
`float32` tensor:

```text
bytes_per_phase = n_frames * NX * NY * n_orientations * n_sigmas * n_frequencies * 4
```

Coarse RF search is intentionally smaller because it uses the derived coarse
grid and usually one coupled size/frequency relationship.
