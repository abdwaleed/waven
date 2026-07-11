# GUI

The GUI wraps pipeline calls in cancellable, resumable tasks and renders the
resulting arrays for inspection. It is an orchestration layer, not a separate
analysis method.

`pipeline_config.json` is optional for GUI startup. When it is absent, empty, or
omits a field, the corresponding GUI entry is left blank for the user. Empty
strings are valid GUI configuration values; type and required-value validation
occurs only when an action that needs the field is run.

## Staged workflow

The left side uses six tabs directly below Session Configuration: **Setup**,
**Stimulus**, **Gabor**, **Wavelets**, **Analysis**, and **Export**. A successful
stage enables dependent actions and advances to the next chronological tab.

Stimulus downsampling is explicit and percentage-driven. The selected movie is
the source of truth for width, height, frame count, FPS, and duration; users do
not enter `NX`, `NY`, or stimulus frame rate. The displayed analysis grid is
the movie width and height multiplied by the selected percentage. Wavelet
decomposition requires its cache and does not silently repeat downsampling.

The Stage 2 NPY/Zarr selector is always shown. The Wavelets tab also always
shows a mutually exclusive NPY/Zarr selector beside **Run Wavelet
Decomposition**, so the selected storage is visible before a run. Wavelet startup reconciles
current and legacy downsample names, accepts either storage format when the
array shape matches, and prints the exact reused path.

The Export tab always preserves PNG/SVG and metadata formats. Its array selector
chooses NPY, Zarr, or both for reusable numerical payloads. **Export All
Individual Neurons** renders the selected-neuron spike and tuning figures for
every loaded cell or unit.

## Conditional inputs

The GUI keeps values when a control is hidden, but displays only fields relevant
to the current choice:

- **coarse** shows the coarse Gabor and wavelet-cache folders;
- **full** shows the fine Gabor folder, full-model sigmas, full wavelet folder,
  full wavelet format, and model-output folder;
- **Fresh / raw data** shows only the raw acquisition folder and neural-cache
  output format;
- **Continue / existing cache** shows only the folder containing an existing
  `spikes`/`pos` pair.

Non-path text fields include a visible example beneath the entry. These examples
distinguish integers, floats, booleans, tuples, and numeric lists, and state the
units where applicable.

## Folder contract

GUI path fields are folders. `Project Root` anchors the strict project layout:
`input/raw_data`, `input/stimulus_movie`, `input/neural_cache`,
`cache/gabor/{coarse,full,kernels}`, `cache/wavelets/{coarse,full}`, and
`output/{plots,recovery_cache,models}`. Browse buttons keep the visible field
pointed at the conventional folder. If you choose an external source folder,
the GUI stores a lightweight `.waven_reference.json` pointer inside the
conventional folder.

The stimulus movie field is a folder containing one movie. When the GUI opens
that movie, it reads frame count, FPS, width, and height. Those values are used
directly throughout downsampling, Gabor construction, wavelet decomposition,
model timing, and ephys alignment. The computed analysis-grid dimensions are
shown read-only in the Gabor tab.

## Long-task contract

| GUI action | Durable resume point |
| --- | --- |
| select analysis scale | saved GUI state and cache fingerprint |
| create neural cache | `spikes`/`pos` cache pair plus selected source |
| prepare downsampled video cache | selected-format path, sibling format, and legacy movie-adjacent names validated by shape |
| build Gabor library | selected coarse/fine library file shape |
| run wavelet decomposition | selected coarse cache or full-model phases |
| run RF analysis | plot cache and RF payload |
| run model plots | selected coarse `run_Model` or full `run_Full_Model` plot cache |
| export figures | image files plus numeric metadata bundle |

Status text should reflect the currently running stage. If a stage completed
successfully before interruption, rerunning the button should skip that valid
artifact and continue with the next missing one.

Completion is tracked per scale. A coarse downsample or Gabor cache does not
unlock full decomposition. Editing a consumed movie, grid, coverage, filter, or
neural input relocks only the dependent buttons; unrelated completed stages stay
available.

The `Analysis scale` selector drives the large buttons. In `coarse` mode, the
Gabor and wavelet buttons build the RF screening artifacts and the model button
runs `run_Model`. In `full` mode, they build the fine/full artifacts and the
model button runs `run_Full_Model`. Coarse RF analysis remains separate because
the full model still uses coarse RF preferred features as seeds.

The `Neural Spike/Position Cache` section groups mutually exclusive neural
inputs. Choose **Fresh / raw data** to build `spikes` and `pos` from the
acquisition data under `Dir`, or choose **Continue / existing cache** to
validate an existing `spikes.npy` or `spikes.zarr` and the matching `pos` cache
beside it. The `Create as` selector is displayed only for fresh processing and
controls whether the button writes `spikes.npy`/`pos.npy` or
`spikes.zarr`/`pos.zarr`.

The `Stimulus Downsample Cache` section controls movie downsampling before
Gabor projection. Its percentage slider scales dimensions read from the movie:
`20%` creates each axis at roughly one fifth of the source, while `100%`
preserves source dimensions. Both NPY and Zarr cache formats are available.
Artifacts carry small `.waven.json` sidecars, so reruns reuse only outputs whose
shape and parameter fingerprint still match the current movie and GUI settings.

## Backend-aware displays

The Gabor and Wavelets tabs deliberately describe different work for the two
backends. **Legacy** builds a large flattened Gabor library, so it exposes the
library NPY/Zarr choice and a library-size estimate. **Convolution** builds a
compact kernel cache instead; those legacy-library controls are hidden because
they would describe an artifact that is not created. Both backends show the
Wavelet storage-format selector, and the run button names the active backend.

::: waven.gui

::: waven.app.gui
