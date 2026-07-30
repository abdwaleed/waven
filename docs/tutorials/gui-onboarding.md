# GUI Onboarding

This is the primary way to run WavEn. Feel free to enter your experiment settings in the GUI
as you read this guide, and save `pipeline_config.json` only when you want a
restorable snapshot of those choices.

If you prefer a JSON-first workflow of this guide, use
[Restore GUI Settings](../how-to/prepare-configuration.md); the field meanings
there are the same as the descriptions below.

!!! note "App Overview"
    ![App Overview](images\app_overview.png)

## 1. System Configuration

This section will walk through pointing to your `my_experiment` folder, your Recovery folder, loading and saving GUI 
inputs as JSON, and optional opt-in optimizations.

### a. Project Root Folder & Recovery Checkpoint Directory

!!! note "Project Root and Recovery Folder"
    ![Project Root and Recovery Folder](images\step-1.png)

| Input | Description |
| --- | --- |
| Project Root Folder | This is the root for `input/`, `cache/`, and `output/`. Set this folder ideally in an NVME SSD or at least any SSD. USB peripherals may cause read/write bottlenecks, however, so ensure the used disk drive is ideally installed as an internal system drive. |
| Recovery Checkpoint Directory | If you set `Project Root Folder` above, this field auto-updates to point to the `output/` subfolder, if you recall from the [First GUI Analysis](first-gui-analysis.md) section. Otherwise, you can hardcode a specific desired recovery folder path. In both cases, the program will populate either folder with the same subfolders and files. |

### b. Save / Load `pipeline_config.json`

!!! note "Loading and Saving JSON"
    ![Loading and Saving JSON](images\step-2.png)

Optional convenience for saving or loading the input configurations of a known-good session. It is not required to enter or run an experiment.

### c. Performance & Hardware Speedup Options

!!! note "Performance & Hardware Speedup Options"
    ![Performance & Hardware Speedup Options](images\step-3.png)

Multiple features here tend to speedup the code tremendously. For safety, all these features have fallbacks in case they fail.

| Input | Description |
| --- | --- |
| RAM acceleration cache (later analysis) | Keeps safely sized, repeatedly used model-phase and PSTH/STA arrays in memory to accelerate later analysis. Large arrays remain disk-backed automatically. |
| Use compatible GPUs | Enables parallel convolution across multiple CUDA GPUs only when their capabilities are sufficiently compatible. Incompatible or substantially slower GPUs are excluded to avoid slowing the run. The hardware status text at the bottom of this panel reports whether CUDA GPUs are detected and whether multi-GPU processing can have any effect on the current computer. |
| Compile stable convolution kernels (experimental) | Uses torch.compile after a one-time warm-up to optimize repeated convolution operations. If compilation fails, processing falls back safely to normal execution. Apparently, this is only supported on Linux systems. |
| Tensor Core convolution (fast precision) | Uses CUDA float16 autocasting for wavelet convolution on supported GPUs. Coarse-RF statistics retain their existing precision. |
| Release RAM acceleration cache | Immediately frees arrays held by the optional RAM cache without changing whether the RAM-cache option is enabled for future work. |

### d. Automate Pipeline

!!! note "Automate Pipeline"
    ![Automate Pipeline](images\step-4.png)

After configuring all the experimental configurations the stimulus, neural data source, visual coverage, sampling, Gabor filter bank, cache folders, and export selections throughout the GUI, use this section to **automate** "clicking the buttons" in the GUI. **GO THROUGH
THE FOLLOWING SECTIONS THEN RETURN TO THIS ONE.**

The automated pipeline prepares the stimulus cache, validates or creates the neural cache, builds the Coarse RF cache, and runs Coarse RF analysis. Optionally, it can export the all-neuron and individual-neuron graphs selected in the Export section after analysis completes. Review and save your pipeline_config.json before running, especially when creating a new cache version or processing a large dataset.

### E. Ephys or 2-Photon

!!! note "Ephys or 2-Photon"
    ![Ephys or 2-Photon](images\step-5.png)

Switch between ephys and 2-photon to change the `2 Session Setup` section of the GUI to suit your needs.

## 2. Stimulus Video

!!! note "Stimulus Video"
    ![Stimulus Video](images\step-6.png)

Choose the movie, define its visual extent, then choose the analysis visual coverage.
The GUI reads movie width, height, frame count, and FPS automatically from metadata.

| Input | Input type | Output | Output type | Relevance and configuration |
| --- | --- | --- | --- | --- |
| Stimulus Movie Folder | Directory containing one intended movie | Movie metadata | Width, height, frames, FPS | This is the source of truth for time and pixel dimensions. Keep the folder unambiguous. |
| Visual Field Coverage | `[left, right, top, bottom]` in degrees | Pixel-to-angle mapping | Calibration values | The angular extent of the full movie. Use the stimulus display's measured bounds. |
| Analysis Field Coverage | Same four-number order, inside Visual Field Coverage | Cropped analysis field | Derived spatial grid | Limits the visual field analysed. Changing it changes dependent stimulus and wavelet caches. |
| Sampling mode: Target degrees/pixel | Positive degrees/pixel | Analysis grid and Nyquist limit | Derived metadata | Recommended when you know the spatial resolution you need. Smaller values retain more detail and create larger caches. |
| Sampling mode: Retain up to cpd | Positive cycles/degree | Analysis grid and Nyquist limit | Derived metadata | Recommended when you know the highest meaningful spatial frequency. The GUI leaves a safety margin above Nyquist. |
| Sampling mode: Compatibility percent | Integer 1–100 | Legacy-style grid fraction | Derived metadata | Use only to reproduce a prior percentage-based grid; it has no direct visual-angle interpretation. |
| Prepare Stimulus Cache | Button | Downsampled/cropped movie | NPY or Zarr binary movie cache | Required before wavelet preparation. This stimulus cache is shared by matching cache versions. |

## 3. Gabor Filters

!!! note "Gabor Filters"
    ![Gabor Filters](images\step-7.png)

Set the Gabor feature bank that the cache will contain. Start from the GUI
recommendation when possible (based on inputs you provide above), then adjust as needed to answer a specific scientific
question.

| Input | Description |
| --- | --- |
| Recommend and Apply Filter Bank | Recommended starting point. It converts visual-angle sampling and a requested cpd range into a physically meaningful filter bank. |
| Orientation Count (`N_thetas`) | Number of evenly spaced orientations from 0° to 180°. Use more only when finer orientation hypotheses are necessary. |
| Sigmas | Gaussian envelope sizes at half maximum (define size of Gabor filter). Use a short list spanning plausible receptive-field sizes; each extra size increases cache cost. |
| Frequencies | Use values below the displayed Nyquist limit. Required for Full Model work and optionally used by Coarse RF frequency-list mode. |
| Phases | Quadrature-style phase offsets used by phase-aware model products (run model and run full model). List of **degrees**, normally `[0, 90]` |
| Coarse RF frequency mode | Coupled mode pairs each size with its corresponding frequency; independent mode evaluates every size/frequency combination and costs more. Idependent mode makes your cache significantly larger. Usually, coupling is good to do. |
| Prepare Gabor Assets | Build once for a matching grid and Gabor bank. Kernel preparation is much smaller than wavelet-cache preparation. |

## 4. Creating or Reusing Decomposition Cache

!!! note "Gabor Filters"
    ![Creating or Reusing Decomposition Cache](images\step-8.png)

The cache folders are libraries. A library retains the shared downsampled
stimulus cache and can hold several named wavelet-cache versions. This is what
makes reuse across new raw neural recordings practical.

| Input | Relevance and configuration |
| --- | --- |
| Coarse RF Cache Library Folder | The parent folder for Coarse RF products. Leave at the project default unless cache storage belongs elsewhere. |
| Full-Model Cache Library Folder | Used only for optional Full Model phase products. |
| Active Coarse RF Cache | ONLY IF you are reusing a cache, consider this input. Choose the cache version used by **Run Coarse RF Analysis**. The GUI validates its movie, crop, grid, and Gabor provenance before analysis. |
| Active Full Model Cache | ONLY IF you are reusing a cache, consider this input. Choose the phase-cache version used by optional Full Model inspection. |
| Prepare Analysis Caches | Builds or reuses `coarse_rf_power.zarr`, the required feature cache for Coarse RF analysis. |
| Prepare Run Model Phase Caches | Select only when you want optional Run Model graphs after basic Coarse RF inspection. Please note that run model by definition linearly couples sigma and frequency, even if you chose the "Independent Frequency List" in the previous step. |
| Prepare Run Full Model Phase Caches | Select only when you want Full Model graphs; this is the largest optional preparation. Unlike run model. run full model uses an independent frequency list and, in fact, requires it. |

Reuse a cache only when the stimulus video, visual/analysis coverage, derived
sampling grid, Gabor orientations, sigmas, frequencies, phases, and frequency
mode match. New raw neural data is compatible with an existing stimulus-side
cache when those conditions remain true. See [Reusable Caches](../reference/storage.md)
for the complete rule and storage map.

## 5. Session Setup: create or validate neural data

!!! note "Session Setup: create or validate neural data"
    ![Session Setup: create or validate neural data](images\step-9.png)

Choose whether you have fresh raw data or an already aligned neural cache.
The source choice changes which path is used; it does not change the selected
stimulus-side cache.

| Input | Relevance and configuration |
| --- | --- |
| Fresh / raw data | Choose for a new recording. WavEn aligns neural activity to the movie frame axis. |
| Continue / existing cache | Choose when a matching aligned pair already exists; raw-data alignment is skipped. |
| Raw Data Folder | Required only for fresh alignment. Use the layouts described in [First GUI Analysis](first-gui-analysis.md). |
| Neural Cache Folder | Receives fresh `spikes`/`pos` output or contains the existing pair to validate. |
| Create / Validate Neural Cache | Produces or verifies the neural inputs consumed by Coarse RF analysis. The result either way should be frame-aligned neural response arrays: `spikes` `(trials, frames, neurons)` and `pos` |

| Input | Relevance and configuration |
| --- | --- |
| Resolution and Number of Planes | Required for fresh 2-photon data. The requested number of Suite2p planes must exist. |
| Sampling Rate and Photodiode Port | Required for fresh ephys data. Use **Find ports**, then confirm the port from wiring. |

## 6. Run Analysis

!!! note "Run Analysis"
    ![Run Analysis](images\step-10.png)

Analysis combines the selected compatible Coarse RF cache with the aligned
neural cache. It uses the cache created or specified in the previous step.

If you did not prepare run model or run full model caches, ignore the `Model Evaluation Settings` panel.

### a. Running the Analysis Itself

| Input | Relevance and configuration |
| --- | --- |
| Analysis Results Folder | Keeps reusable GUI result state separate from stimulus and neural input caches. |
| Run Coarse RF Analysis | Requires a validated neural cache and a selected compatible `coarse_rf_power.zarr`. |

### b. Inspecting a Single Neuron

| Input | Relevance and configuration |
| --- | --- |
| Inspect Single Neuron | Uses the existing Coarse RF result; it does not rerun population analysis. |
| Also run Run Model tuning curves | Requires the selected Coarse cache version's real/imaginary Run Model phase pair. |
| Also run Full Model | Requires the selected Full Model phase-cache version. |

| Input | Relevance and configuration |
| --- | --- |
| Model Evaluation Settings | Only affects optional model fitting. The labels explain save location, sigma list, fitting duration, trial split, and holdout behavior. |

## 7. Export

!!! note "Export's Main Section"
    ![Export's Main Section](images\step-11.png)

Set a persistent export folder once, choose the desired graph types, then write
files without another save-location prompt.

| Input | Relevance and configuration |
| --- | --- |
| Export folder | Set once at the top of Export. The GUI does not ask again for a save location. |
| PNG / SVG / PKL | PNG is presentation-ready; SVG is editable; PKL can be much larger and is for trusted Python reuse. |

!!! note "Export's All or Individual Neurons"
    ![Export All or Individual Neurons](images\step-12.png)

| Input | Relevance and configuration |
| --- | --- |
| All Neurons selections | Exports selected current all-neuron figures. |
| Individual Neurons selections | Exports selected panels for the displayed neuron. |
| Repeat selections for every analyzed neuron | Explicitly rerenders the selected individual graphs for every analysed neuron. **This is the most time-consuming export. Usually, exporting only pkl files is fastest.** |

The Guided Pipeline's export checkboxes use these same Export-tab selections.
For filenames, overwrite behavior, and format trade-offs, see
[Export Results](../how-to/export-results.md).

## Optional: restore this GUI setup later

When the GUI is configured and working, choose **Save `pipeline_config.json`**
to preserve the current paths, dropdowns, checkboxes, and fields. Load it later
with **Load `pipeline_config.json`**. This is a settings backup, not the normal
way to enter a new experiment.

Next: [Restore GUI Settings (`pipeline_config.json`)](../how-to/prepare-configuration.md),
or return to [Reusable Caches](../reference/storage.md) when planning reuse.
