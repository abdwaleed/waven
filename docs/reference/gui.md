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
not enter `NX`, `NY`, or stimulus frame rate. The displayed analysis grid is
the movie width and height multiplied by the selected percentage. Wavelet
decomposition requires its cache and does not silently repeat downsampling.

The Stage 2 NPY/Zarr selector controls the user-visible downsampled movie
cache. Large internal Wavelets products are always chunked Zarr caches, which
keeps their reuse disk-backed. The Wavelets tab identifies their format and
size before a run and prints the exact reused path.

The Export tab always preserves PNG/SVG and metadata formats. Its array selector
chooses NPY, Zarr, or both for reusable numerical payloads. **Export All
Individual Neurons** renders the selected-neuron spike and tuning figures for
every loaded cell or unit.

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

Non-path text fields include a visible example beneath the entry. These examples
distinguish integers, floats, booleans, tuples, and numeric lists, and state the
units where applicable.

## Performance & Hardware controls

Session Configuration includes a **Performance & Hardware** card. These are
runtime scheduling controls, not scientific parameters: changing one never
changes movie dimensions, Gabor values, cache shape, or fitted-model inputs.
They apply immediately to the current GUI process and are included in **Save
Current GUI Inputs / Parameters** under `gui.performance`.

| GUI control | Runtime flag | Default | Applies to |
| --- | --- | --- | --- |
| Adaptive batch tuning | `WAVEN_AUTOTUNE` | on | Convolution wavelet actions |
| Prefetch input chunks | `WAVEN_PREFETCH` | on | Downsampling and convolution wavelet reads |
| Asynchronous cache writing | `WAVEN_ASYNC_WRITER` | on | Convolution wavelet outputs |
| Read movie chunks once across filter groups | `WAVEN_TIME_MAJOR_CONV` | on | Direct coarse-RF power convolution; preserves the group-major fallback when kernel banks cannot safely coexist |
| GPU Coarse RF statistics | `WAVEN_RF_GPU` | on | Coarse RF sufficient-statistics cross-products |
| Use all available GPUs | `WAVEN_MULTI_GPU` | off | Convolution wavelets only, with two or more CUDA GPUs |
| Compile stable convolution kernels | `WAVEN_TORCH_COMPILE` | off | Experimental `torch.compile`; useful for repeated long fixed-shape jobs after its warm-up cost |
| Tensor Core convolution | `WAVEN_AMP` | off | CUDA float16 autocast for convolution only; retain default precision for scientific-equivalence runs |
| FFT STA actual maps | `WAVEN_STA_FFT` | off | Experimental bounded CUDA FFT backend for in-memory, many-lag STA runs; shuffle testing remains exact batched GEMM |

The hardware status line reports the detected CUDA count and whether multi-GPU
can be effective with the selected backend. Choosing multi-GPU on a one-GPU or
CPU-only machine is safe and persists the preference, but it has no effect
until the app runs on a compatible multi-GPU convolution system. GPU Coarse RF
also remains safe on a constrained device: each tile falls back to CPU if it
does not fit.

The **Advanced 2-photon data discovery** card exposes the other optional
runtime setting used by the application: Suite2p timeline dataset roots. Enter
one or more roots separated by the platform path separator (`;` on Windows,
`:` on Linux/macOS) and select **Apply Suite2p Folders**. This field is needed
only for two-photon timeline discovery outside `Dir`; an empty value removes
the setting from the current process so a saved configuration never forces a
path from another computer.

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

For an **ephys** session using the **convolution** backend, neural alignment
writes a sibling raw `spike_counts` cache without changing the standard
firing-rate `spikes` cache. The **Neural & RF Analysis** panel then offers
**Run Spike-Triggered Averaging (STA)**, which streams the prepared stimulus
cache through lagged STA and a circular shuffle test. STA has its own **STA
Neuron Index** selector, separate from the normal coarse-RF Neuron ID. The
**STA Receptive Fields** results tab contains one nested tab
per lag and only fits its phase-sensitive Gabor model after the image passes
the selected 3–5 SD shuffle threshold. See [Spike-Triggered
Averaging](spike-triggered-averaging.md) for the method and artifacts.

The STA card also has a **STA cache format** radio selector. **NPY** is the
default for simple portable arrays and direct memory mapping; **Zarr
(compressed)** writes chunked compressed result arrays to reduce disk use. The
same choice is used by the explicit STA data export and persists in saved GUI
state.

After STA finishes, the GUI asks for a valid STA Neuron Index instead of
plotting an arbitrary cell (a valid index already typed is used directly). The
normal selected-neuron RF figure retains correlation tuning and adds a separate
orientation curve in firing-rate units. Export includes **Export All STA Data
and Lag Graphs** and **Export Every Individual Graph Type for Every Neuron**;
the latter includes the firing-rate orientation graph and, when available, STA
lag plots.

The `Stimulus & Metadata` section controls movie downsampling before neural
alignment and Gabor projection. Its only spatial control is a percentage slider that scales
dimensions read from the movie:
`20%` creates each axis at roughly one fifth of the source, while `100%`
preserves source dimensions. Both NPY and Zarr cache formats are available.
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
