# Configuration

Configuration is the boundary between editable GUI/JSON values and typed Python
objects. The GUI stores most fields as strings because users edit them in text
boxes; `waven.config` parses those strings into validated dataclasses.

## Core dataclasses

| Class | Owns | Important derived values |
| --- | --- | --- |
| `GaborConfig` | filter-library axes and output paths | `theta_radians`, `sigmas_array`, `frequencies_array` |
| `AnalysisConfig` | stimulus paths, neural-data paths, coverage, trial settings | `sigmas_deg` |
| `PipelineConfig` | paired `GaborConfig` and `AnalysisConfig` | `from_json()` with `{PROJECT_ROOT}` expansion |

`{PROJECT_ROOT}` is replaced with `os.getcwd()` at load time. It is not the
folder containing `pipeline_config.json` unless you launch Python from that
folder. For reproducible scripts, either `cd` to the project root first or use
absolute paths.

## JSON typing contract

The example config stores GUI-compatible strings, then `waven.config` parses
them with safe literal parsing:

| Intended type | Example in JSON | Parsed as |
| --- | --- | --- |
| integer | `"18000"` | `int` |
| float | `"1.3671"` | `float` |
| list of floats | `"[2, 4, 8, 12]"` | `tuple[float, ...]` |
| path | `"{PROJECT_ROOT}/your_experiment/output"` | `pathlib.Path` |
| optional path | `"None"` or `""` | `None` |
| boolean | `"True"` / `"False"` | `bool` |
| experiment tuple | `"('CnL43', '2026-05-23', 3)"` | `(str, str, int)` |

Use the same form in the GUI text boxes. If you construct config dictionaries in
Python, native integers, floats, booleans, lists, and `Path` objects are also
accepted.

## Element meanings

| Field | Element order or unit |
| --- | --- |
| `Experiment Info` | `(subject_or_mouse_id, date_string, experiment_number)` |
| `Visual Coverage` | `[azimuth_left, azimuth_right, elevation_top, elevation_bottom]` in visual degrees |
| `Analysis Coverage` | same order as `Visual Coverage`, but for the analyzed crop |
| `Phases` | GUI degrees; use `[0, 90]` for quadrature phases. Conversion to radians occurs at filter construction. |
| `Frequencies` | cycles per analysis pixel |
| `Sigmas`, `Sigmas Full Model` | Gaussian envelope sizes in analysis pixels |
| `Train Trial Indices`, `Test Trial Indices` | zero-based trial indices after trial retention |

## Fields with size impact

| Field | Type | Unit | Affects |
| --- | --- | --- | --- |
| movie dimensions + downsample percentage | metadata + percentage | pixels | derived Gabor grid, wavelets, and downsampled movie |
| `N_thetas` | `int` | bins over 180 degrees | orientation axes and OSI/gOSI curves |
| `Sigmas` | tuple of `float` | pixels | coarse library, coarse wavelets, RF tensor |
| `Sigmas Full Model` | tuple of `float` | pixels | fine library, full wavelets |
| `Frequencies` | tuple of `float` | cycles/pixel | fine library and full-model frequency axis |
| movie frame count | metadata | frames | RF/model time axis and ephys alignment |
| `Number of Trials to Keep` | `int` | trials | first axis of `spikes` |

Changing any of these fields can invalidate previously generated libraries or
wavelet caches. Shape validation catches many mistakes, but keep output folders
organized by configuration so same-shaped-but-different-meaning files do not get
mixed.

::: waven.config
