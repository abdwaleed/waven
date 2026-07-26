# First GUI analysis

This is the recommended first-run path. Complete the stages in order once;
afterward Waven reuses matching caches rather than recalculating them. The GUI
uses one convolutional wavelet implementation—there is no backend choice.

```mermaid
flowchart LR
    A["Configure experiment"] --> B["Prepare stimulus cache"]
    A --> C["Create or validate neural cache"]
    B --> D["Prepare convolution kernels"]
    B --> E["Prepare Coarse RF power cache"]
    C --> F["Run Coarse RF analysis"]
    D --> E --> F
    F --> G["Inspect single neuron"]
    G --> H["Optional Run Model or Run Full Model"]
    F --> I["Export selected results"]
    H --> I
```

## 1. Before starting: computer and storage

Use **32 GB RAM or more** for experiment-sized work. Waven writes stimulus,
neural, and wavelet products to disk in chunks; it does not require every
product to fit in RAM. More RAM still helps operating-system caching and the
optional **RAM acceleration cache**, but it is not a substitute for free disk.

Use a local SSD or NVMe drive for `your_experiment`. Keep at least **twice the
GUI estimate of the largest cache product free**, in addition to the raw movie,
neural cache, and any cache products you want to retain. The Run Full Model
real/imaginary phase pair is normally the largest product. The GUI reports
product-specific estimates before cache preparation; use those estimates rather
than a fixed number of gigabytes.

An NVIDIA CUDA GPU is optional. Waven falls back to CPU when CUDA is absent.
When two or more GPUs have **comparable performance and available memory**,
enable **Use compatible GPUs** in System Configuration to batch convolution
work across them. A markedly slower or smaller GPU can make synchronous work
slower; Waven deliberately excludes unsuitable mixed sets. See
[System Configuration](../reference/gui.md#system-configuration) for the other
optional speed settings.

## 2. Install and launch

Follow [Install and Launch](../how-to/install.md). In short, activate the
`waven` conda environment and run `python ui.py` from the repository root.
The application opens with `pipeline_config.json` as optional defaults; a
missing or partially filled file does not prevent the GUI from opening.

## 3. Make the experiment folder

Create or choose one writable `your_experiment` directory. The conventional
layout is:

```text
your_experiment/
├── input/
│   ├── raw_data/             raw ephys/2-photon source or reference pointer
│   ├── stimulus_movie/       exactly one stimulus movie
│   └── neural_cache/         aligned spikes + positions
├── cache/
│   ├── gabor/                compact convolution-kernel caches
│   └── wavelets/             stimulus, Coarse RF, and model phase caches
└── output/
    ├── plots/                reusable GUI plot cache
    ├── models/               optional model-result location
    └── recovery_cache/       resumable task checkpoints
```

The folders that hold **inputs** should already contain the relevant data.
Waven creates cache and output folders when it writes to them. Folder fields in
the GUI point to directories, not individual files. If raw data live elsewhere,
the GUI can retain a reference inside the conventional project tree. See the
[Experiment Folder reference](../reference/project-layout.md) for ownership and
file names.

## 4. Provide inputs

Enter these fields in **System Configuration** before starting the pipeline.

| Input | Type | Required? | Description | Output / use |
| --- | --- | --- | --- | --- |
| Project Root | writable directory | Yes | The `your_experiment` directory. | Owns the conventional input/cache/output tree. |
| Movie Path | directory containing one movie | Yes | Visual stimulus source. Waven reads frame count, dimensions, and FPS from it. | Metadata and the binary stimulus cache. |
| Visual Coverage | four-number list in degrees | Yes | Full movie field: `[left, right, top, bottom]`. | Maps source pixels to visual angle. |
| Analysis Coverage | four-number list in degrees | Yes | Crop to analyze, in the same order and inside Visual Coverage. | Defines the analysed visual field and grid. |
| Fresh/raw data or existing cache | radio choice | Yes | Choose raw alignment or a previously aligned cache. | Determines how `spikes`/`pos` are obtained. |
| Dir | directory | Fresh/raw only | Raw acquisition root. | Input to alignment. |
| Spks Path | directory | Existing-cache only; output location for fresh data | Folder containing or receiving the neural cache pair. | `spikes` and `pos` cache. |
| 2-photon: Resolution, Number of Planes | positive number, integer | Fresh 2-photon only | Imaging spatial scale and plane count. | Aligned neural positions/activity. |
| Ephys: Sampling Rate, Photodiode Port | positive Hz, integer port | Fresh ephys only | Acquisition rate and the physical TTL/photodiode channel. | Frame-aligned firing-rate cache. |

For ephys, **Find ports** discovers candidate `Din<port>.dat` files. Confirm
the selected port from the acquisition wiring; discovery cannot determine which
wire was the photodiode.

## 5. Choose scientifically meaningful sampling and Gabor axes

Use the physical sampling controls in **Stimulus & Metadata**, not a guessed
percentage whenever possible:

- **Target degrees/pixel** chooses the desired visual-angle size of one
  analysis pixel. Smaller values retain more spatial detail and create larger,
  slower caches.
- **Retain up to cpd** starts from the highest spatial frequency you need. The
  GUI uses a 1.5× safety margin above the Nyquist boundary, so it targets
  `degrees/pixel = 1 / (2 × requested_cpd × 1.5)`.
- **Compatibility percent** is a direct 1–100% horizontal-source sampling
  setting. It is useful for reproducing earlier work but has no direct physical
  interpretation.

After selecting a valid movie and Analysis Coverage, the GUI reports the
derived grid, degrees per pixel, and Nyquist limit. Choose a grid whose Nyquist
limit is comfortably above the largest frequency you intend to analyse. Larger
grids increase space and convolution roughly with the number of pixels.

Enter Gabor values in **Gabor**:

| Input | Type / units | Required? | Scientific reasoning |
| --- | --- | --- | --- |
| N_thetas | positive integer | Yes | Number of orientation bins from 0° inclusive to 180° exclusive. `12` means 15° spacing; increase it only when the expected tuning needs finer angular resolution. |
| Sigmas | list of analysis pixels | Yes | Gaussian envelope sizes for Coarse RF. Use a short, increasing set that spans plausible receptive-field scales; more values multiply cache size and search time. |
| Frequencies | list of cycles per analysis pixel | Required for Full Model; otherwise available for its preparation | Spatial-frequency axis for full-resolution model work. Keep values below the displayed Nyquist limit and use a sparse, approximately logarithmic progression when the plausible range is broad. |
| Phases | list of degrees | Yes | Gabor phase offsets. `[0, 90]` supplies quadrature-style real/imaginary phase information for the phase-aware model caches. |
| Sigmas Full Model | list of analysis pixels | Required for Full Model | Optional finer size grid for local full-model refinement. It is merged with the coarse sizes for the full phase bank. |

The choices describe the hypotheses Waven tests: *where* in visual space,
*which orientation*, *which spatial scale*, *which spatial frequency*, and
*which phase* best relate to a neuron's response. Start with few values, verify
the analysis, then add resolution only where it changes the scientific question.
The detailed field and `pipeline_config.json` reference is in
[Configuration and Scientific Settings](../how-to/prepare-configuration.md).

## 6. Build the durable inputs

| GUI action | Required input | Output type | Output and purpose |
| --- | --- | --- | --- |
| Prepare Stimulus Cache | movie + coverage + sampling | NPY or Zarr `(frames, y, x)` binary array | Common cropped/downsampled movie used by later cache products. |
| Create pos/spikes Cache | raw workflow inputs | NPY or Zarr arrays | Aligned `spikes` `(trials, frames, neurons)` plus `pos`; ephys values are frame-bin firing rates in Hz. |
| Validate Existing Neural Cache | `spikes`/`pos` folder | validated disk-backed arrays | Reuses an already aligned matching pair without raw-data alignment. |
| Prepare Convolution Kernels | movie-derived grid + Gabor axes | kernel-cache files | Compact reusable filters for Coarse RF and Full Model work. |
| Prepare Coarse RF Cache | stimulus cache + kernels | Zarr power array | Power features required by Coarse RF correlation. |

The **Run Guided Coarse RF Pipeline** button runs stimulus preparation, neural
cache preparation, Coarse RF cache preparation, and Coarse RF analysis in that
order. It obeys the currently visible settings. Use individual buttons if you
already have a valid upstream cache or want to stop after a specific stage.

## 7. Analyse and inspect

Click **Run Coarse RF Analysis** after the neural cache and Coarse RF power
cache are ready. It writes/loads the correlation result, chooses preferred
feature locations, and produces all-neuron plus individual-neuron views.

**Inspect Single Neuron** is intentionally fast by default: it renders the
prepared Coarse RF result and a PSTH-weighted spike-triggered-average (STA)
panel. Recently inspected STAs are kept in a small in-memory cache.

Two optional checkboxes add model tuning curves:

| Option | Additional required cache | Adds | Cost |
| --- | --- | --- | --- |
| Also run Run Model tuning curves | `coarse_model_real.zarr` + `coarse_model_imag.zarr` | amplitude, phase, and drift graphs | Extra fitting after basic inspection. |
| Also run Full Model | `dwt_videodata2_r.zarr` + `dwt_videodata2_i.zarr` | full-resolution amplitude, phase, and drift graphs | Largest optional computation; uses the full sigma/frequency phase bank. |

Prepare those phase caches only if you need the corresponding graphs. Both
models use the Coarse RF result as their seed; they are refinements, not a
replacement for Coarse RF analysis.

## 8. Export and recover

The Export tab has two scopes: **All Neurons** and **Individual Neurons**.
Choose graph types, then export to one parent folder. Files are flat and named
with a unit prefix such as `shank1_unit7_orientation_tuning.svg`.

- **PNG** is a presentation-ready raster image.
- **SVG** is an editable publication-quality vector image.
- **PKL** stores graph data and analysis values for trusted Python workflows;
  it is optional because it can be considerably larger and slower.

NPY and Zarr are cache formats, not export formats. See
[Export Results](../how-to/export-results.md) for the exact behavior.

Long tasks keep the interface responsive, report progress, flash the taskbar,
and play a system sound when they finish, fail, or cancel. **Cancel** requests
safe cooperative stopping at the next chunk/iteration boundary; completed cache
tiles remain reusable. Save the current inputs with **Save pipeline_config.json**
and restore them with **Load pipeline_config.json** at any time.

Next: [Configuration and Scientific Settings](../how-to/prepare-configuration.md)
for a field-by-field reference, or [Scientific Intuition](../explanation/pipeline-intuition.md)
for how to interpret the analysis.
