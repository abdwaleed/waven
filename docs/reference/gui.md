# GUI

The GUI wraps pipeline calls in cancellable, resumable tasks and renders the
resulting arrays for inspection. It is an orchestration layer, not a separate
analysis method.

`pipeline_config.json` is optional for GUI startup. When it is absent, empty, or
omits a field, the corresponding GUI entry is left blank for the user. Empty
strings are valid GUI configuration values; type and required-value validation
occurs only when an action that needs the field is run.

## Staged workflow

The left side uses six tabs directly below Session Configuration: **Stimulus &
Metadata**, **Session Setup**, **Gabor**, **Wavelet Products**, **Analysis**,
and **Export**. A successful stage enables dependent actions and advances to
the next chronological tab.

**Stimulus & Metadata is deliberately first.** Selecting and preparing the
movie establishes the authoritative width, height, frame count, FPS, and
duration used by every later stage. In particular, ephys alignment cannot run
until this stage completes: it uses the movie duration to validate photodiode
trial boundaries and the movie frame count to construct the firing-rate cache.

Stimulus downsampling is explicit and percentage-driven. The selected movie is
the source of truth for width, height, frame count, FPS, and duration; users do
not enter `NX`, `NY`, or stimulus frame rate. The percentage sets the horizontal
analysis resolution, then the vertical resolution is derived from the selected
visual coverage so both axes have the same degrees per pixel. Wavelet
decomposition requires its cache and does not silently repeat downsampling.

The Stage 2 NPY/Zarr selector controls the user-visible downsampled movie
cache. The movie is cropped from `Visual Coverage` to `Analysis Coverage` by
converting visual degrees to pixel bounds, then resized to the selected grid.
The cache records this crop provenance, so a cache made with older crop logic
or different coverage values is regenerated rather than silently reused. Large
internal Wavelets products are always chunked Zarr caches, which keeps their
reuse disk-backed. The Wavelets tab identifies their format and size before a
run and prints the exact reused path.

The Export tab always preserves PNG/SVG and metadata formats. Its array selector
chooses NPY, Zarr, or both for reusable numerical payloads. **Section A —
Current Display** has independent all-neuron and individual-neuron graph-type
checkboxes, used by the combined and scope-specific current-display actions.
**Section B — Every Analyzed Neuron** exports only the selected graph axes for
every cell or unit: its spike train, RF map, profiles, and tuning curves are
separate folders with separate data bundles rather than a multi-panel figure.

## Conditional inputs

The GUI keeps values when a control is hidden, but displays the shared-grid
settings required by the current backend:

- Coarse RF and Run Model share a metadata-derived grid and have separate
  consumer-specific wavelet products;
- Run Full Model uses the same spatial grid and adds its full-model sigma and
  frequency feature axes;
- **Fresh / raw data** shows only the raw acquisition folder and neural-cache
  output format;
- **Continue / existing cache** shows only the folder containing an existing
  `spikes`/`pos` pair.
- **Ephys** adds a **Photodiode Port** selector. **Find ports** scans the raw
  data tree for Trodes `Din<port>.dat` filenames and narrows the menu to ports
  actually present; it intentionally does not require a folder named `.DIO`.
  The user must choose the port wired to the photodiode/TTL signal.

Non-path text fields include a visible example beneath the entry. These examples
distinguish integers, floats, booleans, tuples, and numeric lists, and state the
units where applicable.

## Performance & Hardware controls

Session Configuration includes a **Performance & Hardware** card. Adaptive
batch tuning, bounded input prefetch, asynchronous cache writing, time-major
coarse convolution, parallel Suite2p plane loading, and Coarse RF GPU
statistics are core execution paths: they are always enabled and are not saved
as GUI or environment preferences. Coarse RF uses CUDA automatically when it
is available and safely falls back to the CPU for unavailable or oversized
work.

The remaining controls are optional hardware choices, not scientific
parameters: changing one never changes movie dimensions, Gabor values, cache
shape, or fitted-model inputs. They apply immediately to the current GUI
process and are included in **Save Current GUI Inputs / Parameters** under
`gui.performance`.

| GUI control | Runtime flag | Default | Applies to |
| --- | --- | --- | --- |
| RAM acceleration cache | `WAVEN_RAM_ACCELERATION_CACHE` | off | Safely sized reused Run Model and PSTH/STA inputs |
| Use compatible GPUs | `WAVEN_MULTI_GPU` | off | Convolution wavelets only; mismatched cards automatically fall back to one GPU |
| Compile stable convolution kernels | `WAVEN_TORCH_COMPILE` | off | Experimental `torch.compile`; useful for repeated long fixed-shape jobs after its warm-up cost, with automatic eager fallback if setup or execution fails |
| Tensor Core convolution | `WAVEN_AMP` | off | CUDA float16 autocast for convolution only; retain default precision for scientific-equivalence runs |

The hardware status line reports the detected CUDA count and whether multi-GPU
can be effective with the selected backend. When enabled, the app uses only
cards with matching CUDA architecture and broadly similar estimated convolution
throughput and VRAM; otherwise it reports why it selected one GPU instead.
Choosing multi-GPU on a one-GPU or CPU-only machine is safe and persists the
preference, but it has no effect until the app runs on a compatible multi-GPU
convolution system.

Cancelling a convolution wavelet action retains its resumable output tiles and
the task's recovery checkpoint. After stimulus downsampling completes, both
**Create pos/spikes Cache** and **Validate Existing Neural Cache** become
available; changing between the two sources does not change that prerequisite.

The **Advanced 2-photon data discovery** card exposes the other optional
runtime setting used by the application: Suite2p timeline dataset roots. Enter
one or more roots separated by the platform path separator (`;` on Windows,
`:` on Linux/macOS) and select **Apply Suite2p Folders**. This field is needed
only for two-photon timeline discovery outside `Dir`; an empty value removes
the setting from the current process so a saved configuration never forces a
path from another computer.

Two-photon cache creation reads completed Suite2p output; it does not segment
raw TIFF recordings itself. Each configured plane must contain `spks.npy`,
`iscell.npy`, and `stat.npy` under a folder such as `suite2p/plane0`. When that
output lives outside the default experiment layout, enter its parent `suite2p`
folder in **Completed Suite2p output folder**. A Cortex Lab `Timeline.mat` is
also required to align the neural recording to the stimulus; a standalone
`timestamps.npy` does not provide the photodiode/TTL stimulus events needed for
that alignment.

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

All task settings are captured on the GUI thread before a worker begins.
Workers report progress and completion through a main-thread callback queue;
they do not call Tk/Tcl directly. This keeps the window interactive during
cache creation and avoids the Windows minimize/restore deadlock caused by
cross-thread Tk calls. The task status, terminal view, Cancel button, and
normal window minimize/restore controls remain usable while processing.

| GUI action | Durable resume point |
| --- | --- |
| create neural cache | `spikes`/`pos` cache pair plus selected source |
| prepare downsampled video cache | selected-format path, sibling format, and legacy movie-adjacent names validated by shape |
| prepare Gabor assets | legacy coarse/fine libraries or convolution coarse/fine kernel caches |
| prepare Coarse RF power cache | `coarse_rf_power.zarr` |
| prepare Run Model phase caches | `coarse_model_real.zarr` and `coarse_model_imag.zarr` |
| prepare Run Full Model phase caches | `dwt_videodata2_r.zarr` and `dwt_videodata2_i.zarr` |
| run RF analysis | plot cache and RF payload |
| run model plots | selected coarse `run_Model` or full `run_Full_Model` plot cache |
| export figures | image files plus numeric metadata bundle |

Status text should reflect the currently running stage. If a stage completed
successfully before interruption, rerunning the button should skip that valid
artifact and continue with the next missing one.

Completion is tracked per product, not by a coarse/full session selector.
Preparing RF power unlocks Coarse RF analysis; preparing model phases unlocks
Run Model; preparing full-model phases unlocks Run Full Model. Coarse RF
analysis remains the common prerequisite because both model paths use its
preferred feature locations as seeds. Editing a consumed movie, percentage,
coverage, filter, or neural input relocks only dependent products.

At startup and after loading a configuration, the GUI also scans compatible
stimulus, neural, Gabor, and wavelet artifacts. A validated existing product
unlocks the same next action as a product created in the current session; an
existing neural cache is additionally checked against the movie-derived frame
count before it is accepted.

The `Neural Spike/Position Cache` section groups mutually exclusive neural
inputs. Choose **Fresh / raw data** to build `spikes` and `pos` from the
acquisition data under `Dir`, or choose **Continue / existing cache** to
validate an existing `spikes.npy` or `spikes.zarr` and the matching `pos` cache
beside it. The `Create as` selector is displayed only for fresh processing and
controls whether the button writes `spikes.npy`/`pos.npy` or
`spikes.zarr`/`pos.zarr`.

When an individual neuron is inspected after **Run Coarse RF Analysis**, the
same tab also shows its **PSTH-weighted Spike-Triggered Averages (0–300 ms)**.
The analysis uses the trial-averaged, frame-aligned firing rate together with
the prepared coarse stimulus movie; it computes one standard STA map per valid
lag and marks the map with the greatest pixel variance. See
[PSTH-weighted Spike-Triggered Averaging](psth-spike-triggered-averaging.md)
for the method and exported arrays.
The `Stimulus & Metadata` section controls movie downsampling before neural
alignment and Gabor projection. Its only spatial control is a percentage slider that scales
dimensions read from the movie:
`20%` creates each axis at roughly one fifth of the source, while `100%`
preserves source dimensions. The usable slider range is **1–100%**: zero is
not a meaningful stimulus grid and older saved zero values are normalized to
1% before cache creation. Both NPY and Zarr cache formats are available.
Artifacts carry small `.waven.json` sidecars, so reruns reuse only outputs whose
shape and parameter fingerprint still match the current movie and GUI settings.

## Backend-aware displays

The Gabor and Wavelets tabs deliberately describe different work for the two
backends. **Legacy** builds large flattened coarse and fine Gabor libraries, so
it exposes library storage and estimates. **Convolution** builds compact coarse
and fine kernel caches instead; legacy-library controls are hidden because they
would describe an artifact that is not created. Both backends then prepare the
same three named Zarr wavelet products.

::: waven.gui

::: waven.app.gui
