# GUI

The GUI wraps pipeline calls in cancellable, resumable tasks and renders the
resulting arrays for inspection. It is an orchestration layer, not a separate
analysis method.

## Long-task contract

| GUI action | Durable resume point |
| --- | --- |
| select analysis scale | saved GUI state and cache fingerprint |
| build Gabor library | selected coarse/fine library file shape |
| run wavelet decomposition | selected coarse cache or full-model phases |
| run RF analysis | plot cache and RF payload |
| run model plots | selected coarse `run_Model` or full `run_Full_Model` plot cache |
| export figures | image files plus numeric metadata bundle |

Status text should reflect the currently running stage. If a stage completed
successfully before interruption, rerunning the button should skip that valid
artifact and continue with the next missing one.

The `Analysis scale` selector drives the large buttons. In `coarse` mode, the
Gabor and wavelet buttons build the RF screening artifacts and the model button
runs `run_Model`. In `full` mode, they build the fine/full artifacts and the
model button runs `run_Full_Model`. Coarse RF analysis remains separate because
the full model still uses coarse RF preferred features as seeds.

The `Neural Spike/Position Cache` section groups the aligned-neural-cache
controls in chronological order before Gabor and wavelet work. `Spks Path` is
the optional "I already calculated these" path; point it at `spikes.npy` or
`spikes.zarr` when a cache already exists. The `Create as` selector controls
whether the `Create pos/spikes Cache (.npy or .zarr)` button writes
`spikes.npy`/`pos.npy` or `spikes.zarr`/`pos.zarr`. If `Spks Path` is blank or
`None`, the GUI first searches the experiment cache for either supported pair.
If none is found, alignment runs and writes the selected format.

::: waven.gui

::: waven.app.gui
