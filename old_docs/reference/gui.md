# GUI workflow and System Configuration

The GUI is designed as a staged, cache-aware workflow. It is convolution-only:
the Gabor action prepares compact convolution kernels, and later actions create
the product required by their consumer.

## System Configuration

System Configuration contains the current experiment inputs, save/load actions,
the guided workflow, and advanced performance settings.

| Control | Input | Result |
| --- | --- | --- |
| **Load pipeline_config.json** | a JSON configuration file | Restores workflow, paths, sampling, Gabor values, cache formats, optional model choices, speed controls, and export format selections. |
| **Save pipeline_config.json** | destination JSON path | Writes the current GUI inputs in the current schema. |
| **Run Guided Coarse RF Pipeline** | current valid inputs | Runs Prepare Stimulus Cache → Create/Validate Neural Cache → Prepare Coarse RF Cache → Run Coarse RF Analysis. |
| **Performance & Hardware (advanced)** | optional switches | Applies process-local speed/memory choices before the next action. |

Save and load are intentionally available while a background task runs. They do
not change the immutable settings snapshot already captured by that task; they
affect subsequent actions.

### Optional speed and hardware settings

| Switch | What it does | When to enable |
| --- | --- | --- |
| RAM acceleration cache | Holds only safely sized, repeatedly used model-phase and PSTH/STA inputs in RAM. | You have spare RAM and revisit model/inspection work. |
| Use compatible GPUs | Splits eligible convolution batches across comparable CUDA GPUs. | Two or more comparable cards with sufficient free memory. |
| Compile stable convolution kernels (experimental) | Uses `torch.compile` after warm-up and retries eagerly on failure. | Repeated same-shape convolution work where warm-up cost is acceptable. |
| Tensor Core convolution (fast precision) | Uses CUDA float16 autocast for convolution only. | Tensor Core GPU and you want speed; Coarse RF statistics retain their normal precision. |

The hardware summary reports whether CUDA is available. Multi-GPU has no effect
with one or no CUDA GPU, and Waven falls back safely rather than waiting for an
unsuitable device.

## Stage actions

| Stage | Action | Required input | Durable output | Enables |
| --- | --- | --- | --- | --- |
| Stimulus & Metadata | Prepare Stimulus Cache | movie, coverage, sampling, NPY/Zarr selection | binary movie cache | Neural alignment validation, wavelets, STA. |
| Session Setup | Create pos/spikes Cache | raw 2-photon/ephys inputs | aligned neural pair | Coarse RF. |
| Session Setup | Validate Existing Neural Cache | existing matching `spikes`/`pos` folder | validated disk-backed pair | Coarse RF without raw alignment. |
| Gabor | Prepare Convolution Kernels | sampling grid and Gabor axes | compact kernel cache | Wavelet-product preparation. |
| Stimulus Wavelet Pipeline | Prepare Coarse RF Cache | stimulus cache + kernels | `coarse_rf_power.zarr` | Coarse RF analysis. |
| Stimulus Wavelet Pipeline | Prepare Run Model Phase Caches | stimulus cache + kernels | `coarse_model_real.zarr`, `coarse_model_imag.zarr` | Optional Run Model tuning graphs. |
| Stimulus Wavelet Pipeline | Prepare Run Full Model Phase Caches | stimulus cache + full settings | `dwt_videodata2_r.zarr`, `dwt_videodata2_i.zarr` | Optional Full Model tuning graphs. |
| Analysis | Run Coarse RF Analysis | neural pair + Coarse RF power | RF correlation result and plot state | Population plots and individual-neuron inspection. |

The NPY/Zarr selectors describe **cache storage**, not graph export. See
[Cache Storage and Disk Planning](storage.md) for their purpose.

## Individual-neuron inspection

**Inspect Single Neuron** always uses the existing Coarse RF result and creates
the spike train, RF/tuning, orientation/selectivity, and PSTH-weighted STA
views. It does not rerun Coarse RF analysis.

| Checkbox | Default | Requirement | Additional output |
| --- | --- | --- | --- |
| Also run Run Model tuning curves (slower) | Off | Coarse model real/imaginary phase pair | amplitude, phase, and drift graphs from the coarse phase bank. |
| Also run Full Model | Off | Full model real/imaginary phase pair | amplitude, phase, and drift graphs from the full sigma/frequency phase bank. |

Keeping both off gives the fastest routine inspection. Both model actions use
the Coarse RF result as their starting point; they are optional refinements.

## Export

The streamlined Export tab separates two scopes:

| Scope | Select | Result |
| --- | --- | --- |
| All Neurons | available population graph types | Export the selected current all-neuron figures. |
| Individual Neurons | available individual graph types; optional repeat for every analysed neuron | Export the current neuron or the selected graphs for every analysed neuron. |

Run Model/Full Model graph choices appear only when their corresponding phase
caches are available. Exports are written directly into the selected parent
folder with names such as `shank1_unit7_orientation_tuning.png`.

| Format | Purpose | Cost |
| --- | --- | --- |
| PNG | raster figure for slides/review | Fast, moderate image memory. |
| SVG | editable vector figure for publication | Can be slower for complex dense figures. |
| PKL | graph data and analysis values for trusted Python reuse | Optional; can be much larger and slower. |

See [Export Results](../how-to/export-results.md) for precise selection and
file behavior.

## Responsiveness, completion, and cancellation

Long actions run outside the Tk event loop so the window can repaint and report
progress. When a normal background task, configuration action, or export
finishes, fails, or cancels, Waven flashes the taskbar and plays a system sound.

**Cancel** is cooperative: Waven checks for cancellation between durable
chunks, neurons, or model iterations. It stops as soon as the current
interruptible operation finishes rather than corrupting a cache. Completed
tiles and valid caches remain reusable; a cancellation request is preferable to
force-closing the process.
