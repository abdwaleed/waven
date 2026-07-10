# Prepare configuration

The easiest starting point is `pipeline_config.json`. Most values in the example
file are strings because the same values can be loaded into the GUI text fields.
The parser then converts those strings into integers, floats, booleans, lists,
tuples, or paths.

## Project root placeholder

Use `{PROJECT_ROOT}` when you want paths to be portable across machines:

```json
"Movie Path": "{PROJECT_ROOT}/your_experiment/input/zebra_movie/zebra_movie.mp4"
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
| file or directory path | `"{PROJECT_ROOT}/your_experiment/output"` | Use quotes. Empty strings are allowed only for optional paths. |
| integer | `"18000"` | Quotes are fine; the parser converts to `int`. |
| float | `"1.3671"` | Quotes are fine; the parser converts to `float`. |
| list of numbers | `"[2, 4, 8, 12]"` | Keep brackets and commas. |
| tuple/list with labels | `"('mouse01', '2026-05-23', 3)"` | Used by `Experiment Info`. |
| boolean | `"True"` or `"False"` | Also accepts `true`/`false`, `1`/`0`, `yes`/`no`. |
| intentionally missing optional path | `"None"` or `""` | Use only for optional fields such as `Spks Path`. |

Required paths should point to existing inputs before you run analysis. Output
directories can be missing; `waven` creates them when it writes outputs.

## File structure

The JSON file has four main sections:

| Section | Required? | Meaning |
| --- | --- | --- |
| `workflow` | yes | `"2p"` for two-photon or `"ephys"` for electrophysiology. |
| `gabor_param` | yes | Filter-library axes and library output path. |
| `common` | yes | Stimulus, trial, coverage, and shared path settings. |
| `two_photon` / `ephys` | one matching `workflow` | Workflow-specific acquisition settings. |

## `gabor_param`

These fields define the Gabor filters before they are applied to the movie.

| Field | Type to type | Units | File or dir? | Element meaning |
| --- | --- | --- | --- | --- |
| `N_thetas` | integer string, e.g. `"18"` | orientation bins | no | Number of orientations between 0 inclusive and 180 exclusive. `18` means 10-degree spacing. |
| `Sigmas` | list string, e.g. `"[2, 4, 8, 12]"` | analysis pixels | no | Gaussian envelope sizes for the coarse Gabor filters. Larger values mean broader filters. |
| `Frequencies` | list string, e.g. `"[0.02, 0.06, 0.1]"` | cycles per analysis pixel | no | Spatial-frequency bins for the fine/full model. Multiple values create the frequency axis. |
| `Phases` | list string, e.g. `"[0, 1.57079632679]"` | radians | no | Gabor phase offsets. `1.57079632679` is pi/2, the common quadrature phase. Do not type `90` unless you intentionally mean 90 radians. |
| `NX` | integer string | analysis pixels | no | Full-grid horizontal samples. Must match `common.NX`. |
| `NY` | integer string | analysis pixels | no | Full-grid vertical samples. Must match `common.NY`. |
| `Save Path` | path string ending in `.npy` | path | file | Base Gabor library path. GUI-derived coarse/fine paths are usually made from this path. |

## `common` paths

| Field | Type to type | File or dir? | Required? | Meaning |
| --- | --- | --- | --- | --- |
| `Dir` | path string | directory | yes | Root raw-data directory. For two-photon, this should contain the experiment folder tree. For ephys, this should point to the ephys input root. |
| `Path Directory` | path string | directory | yes | Working directory for coarse wavelet outputs and RF cache files. |
| `Movie Path` | path string ending in `.mp4` or compatible video | file | yes | Stimulus movie used for downsampling and wavelet decomposition. |
| `Library Path` | path string ending in `.npy` or `.zarr` | file or Zarr directory | yes | Backward-compatible library path. The GUI also tracks explicit coarse/fine library paths. |
| `Spks Path` | path string, `"None"`, or `""` | file or Zarr directory | optional | Existing aligned `spikes.npy` or `spikes.zarr`. The loader looks for a sibling `pos.npy` or `pos.zarr`; use `"None"` to search the experiment cache first and then run workflow-specific alignment if no valid pair exists. |
| `Full Model Wavelet Path` | path string or `""` | directory | optional but recommended for full mode | Directory containing or receiving `dwt_videodata2_r/i.npy` or `.zarr`. |
| `Full Model Save Path` | path string or `""` | directory | optional but recommended | Directory for model outputs, plot cache, and exports. |
| `Plot Cache Path` | path string ending in `.pkl.gz` or `""` | file | optional | GUI cache file for expensive plot payloads. |
| `Recovery Cache Directory` | path string or `""` | directory | optional | Checkpoint directory for resumable long GUI tasks. |

## `common` identifiers and acquisition timing

| Field | Type to type | Units | Element meaning |
| --- | --- | --- | --- |
| `Experiment Info` | tuple/list string, e.g. `"('CnL43', '2026-05-23', 3)"` | labels | `(subject_or_mouse_id, date_string, experiment_number)`. The date is a label used in paths; use your lab's folder convention. |
| `Block End` | integer string | acquisition block index | Last block considered during alignment. Use `"0"` when your workflow does not need a later block. |
| `Hz` | integer string | frames per second | Stimulus frame rate after movie presentation. Used for frames-per-minute and holdout windows. |
| `Number of Frames` | integer string | frames | Expected or maximum stimulus frames. RF analysis uses the minimum of config frames, spike frames, and wavelet frames. |
| `Number of Trials to Keep` | integer string | trials | Number of repeated stimulus trials retained in the aligned spike tensor. |
| `Train Trial Indices` | `"auto"` or list string | zero-based trial indices | Training trials for model fitting. Example `"[0, 2]"` means first and third retained trials. |
| `Test Trial Indices` | `"auto"` or list string | zero-based trial indices | Held-out trials. `"auto"` means every retained trial not used for training. |
| `Use Last Minute Holdout` | boolean string | yes/no | If true, model evaluation reserves the final minute of frames. |

## `common` spatial fields

| Field | Type to type | Units | Element meaning |
| --- | --- | --- | --- |
| `screen_x` | integer string | display pixels | Native horizontal display resolution of the stimulus movie/presentation. |
| `screen_y` | integer string | display pixels | Native vertical display resolution. |
| `NX` | integer string | analysis pixels | Horizontal size of the full analysis grid and fine wavelet grid. |
| `NY` | integer string | analysis pixels | Vertical size of the full analysis grid and fine wavelet grid. |
| `Visual Coverage` | four-number list string | visual degrees | `[azimuth_left, azimuth_right, elevation_top, elevation_bottom]` for the full stimulus field. Preserve this order. |
| `Analysis Coverage` | four-number list string | visual degrees | Same four-element order, but for the crop you want to analyze. It should lie inside `Visual Coverage`. |
| `Sigmas` | list string | analysis pixels | Coarse RF Gabor sizes. Should match `gabor_param.Sigmas`. |
| `Sigmas Full Model` | list string | analysis pixels | Full-model Gabor sizes. These are merged with `gabor_param.Sigmas` when building the fine library. |
| `Frequencies` | list string | cycles per analysis pixel | Spatial-frequency bins. Should match `gabor_param.Frequencies`. |

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
| `n_trials` | `Number of Trials to Keep` and alignment | `spikes` |
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
