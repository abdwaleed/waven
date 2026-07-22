# Export results

Export is the final, optional GUI stage. It writes files from completed
analysis results and does not make an incomplete upstream stage valid. The
every-neuron model-curve choices are the exception: they fit the selected Run
Model or Run Full Model for each neuron before writing those requested curves.

## Before clicking Export

| Input/control | Type | Required? | Effect | Output type |
| --- | --- | --- | --- | --- |
| completed Coarse RF result | analysis result | Yes | Supplies current population/selected-neuron plots and numerical payloads | figures plus arrays/metadata |
| Run Model phase pair | `coarse_model_real.zarr` + `coarse_model_imag.zarr` | Only for every-neuron Run Model curves | Fits amplitude, phase, and drift curves for each exported neuron | figures plus arrays/metadata |
| Run Full Model phase pair | `dwt_videodata2_r.zarr` + `dwt_videodata2_i.zarr` | Only for every-neuron Run Full Model curves | Fits the full-resolution amplitude, phase, and drift curves for each exported neuron | figures plus arrays/metadata |
| optional displayed model result | analysis result | Only for current-display model plots | Adds its currently displayed diagnostics to eligible exports | figures plus arrays/metadata |
| graph-type checkboxes | Boolean selections | Yes for each action | Choose visible graph/data payloads to write | selected graph bundles |
| array format | `npy`, `zarr`, or `both` | Optional; defaults shown in GUI | Format for reusable numerical arrays only | `.npy`, `.zarr`, or both |
| export preset | Quick review, Data bundle, Full archive | Optional | Controls packaging and metadata density | folders and optional archives |
| delivery | folder, ZIP, or both | Optional | Chooses how a finished package is delivered | directory and/or `.zip` |

The Export tab has two clearly separated sections.

**Section A — Current Display** exports the graphs that are currently visible:

- **Export Current GUI: All + Individual**;
- **Export Current Display: All-Neuron Graphs**;
- **Export Current Display: Individual-Neuron Graphs**.

Tick the graph-type checkboxes above each action to control the contents. The
current all-neuron and individual-neuron selections are independent, so a
combined export includes only the checked graph types from each view.

**Section B — Every Analyzed Neuron** has **Export Selected Single-Graph Files
for Every Neuron**. Its checkboxes select spike trains, RF maps, elevation,
azimuth, orientation/size/frequency tuning, PSTH-weighted STA lag maps, and
independent Run Model and Run Full Model amplitude, phase, and drift curves.
Selecting a model curve fits that model for every neuron; Run Full Model is
slower because it uses the larger full-resolution sigma/frequency phase bank.
It creates one folder per graph: a multi-panel tuning dashboard is never stored
as a single graph-data bundle.

Each exported graph can include:

- PNG image;
- SVG vector file;
- `.npy`, `.zarr`, or both for reusable arrays;
- Python pickle with richer payload data;
- JSON manifest describing files, axes, and array keys.

Use exports when you need a figure and the numerical data that produced it.
The selected array format does not alter PNG, SVG, JSON, or pickle outputs.
The all-neuron single-graph export creates a directory for every selected graph
of every neuron and can therefore be large for recordings with many cells.

## Fast export presets

The Export tab provides three output presets:

- **Quick review** writes cropped PNG graphs only. Choose this for visual QA or
  a fast result review. Images are stored directly in each neuron's `graphs/`
  folder to avoid one directory operation per graph.
- **Data bundle** writes PNG graphs plus one uncompressed `data.npz` and one
  `data_manifest.json` per neuron. Its images are also flat under `graphs/`,
  and it avoids repeating small arrays, pickles, and manifests in every graph
  directory.
- **Full archive** retains the established PNG, SVG, pickle, reusable-array,
  and per-graph-manifest layout.

For all-neuron exports, **Per-neuron .npz** is the compact numeric layout. Its
NPZ member names are prefixed by the matching graph directory name, while the
neuron-level manifest maps those names to graph titles and axes. The array
format selector applies only to the legacy per-graph layout.

For a selected STA export, Waven calculates a small RAM-bounded group of
neurons at a time using vectorized matrix products, then writes each neuron's
usual PNG, SVG, pickle, array, and manifest files. When model curves are
selected, Waven fits Run Model and/or Run Full Model one neuron at a time before
that neuron's export snapshot is queued; this keeps Tk responsive and avoids
holding all model figures in RAM. Batch export avoids unneeded Tk canvas
refreshes and keeps a bounded queue of up to four prepared neurons with two
writer workers by default, so rendering and storage can overlap without
accumulating every neuron in RAM. Set `WAVEN_EXPORT_WORKERS` to a value from
`1` to `4` only when you have measured that your CPU and destination drive
benefit from it.

The terminal reports per-neuron export timings for restoring figures, drawing,
PNG/SVG output, numeric arrays, pickles, manifests, file count, and bytes. Use
those timings to choose the appropriate preset rather than assuming that a
particular format is the bottleneck.

## Output structure and when to use it

| File/folder | Type | Written by | Required? | Purpose |
| --- | --- | --- | --- | --- |
| `graphs/` | directory | all presets | Yes | Image output grouped by selected view/neuron. |
| `*.png` | raster image | all presets | Yes | Fast visual review and paper-draft figures. |
| `*.svg` | vector image | Full archive | Optional | Editable/publication vector figure. |
| `data.npy` or `data.zarr` | numerical array | selected array format | Optional | Reusable array values behind an exported graph. |
| `data.npz` | compressed array bundle | Data bundle | Optional | One compact numeric package per neuron/unit. |
| `manifest.json` / `data_manifest.json` | JSON metadata | Data bundle and Full archive | Optional | Graph title, axes, array keys, and generated files. |
| `*.pkl` | Python pickle | Full archive | Optional | Richer Python-specific payload; use only in trusted Python workflows. |
| `export_manifest.json` | JSON summary | batch exports | Yes for batch output | Records package contents and any per-neuron failures. |

`npy` is straightforward for Python/NumPy interchange. `zarr` is useful for
larger chunked data. Choose `both` only when two downstream consumers actually
need it because it duplicates numeric storage. PNG/SVG and metadata output are
independent of this numeric-array selection.

Export folders use compact, collision-safe directory names so batch exports
remain below typical Windows path-length limits. The complete graph title and
the list of generated files remain in each graph's `manifest.json`; a batch
`export_manifest.json` records any individual neurons that could not be
exported without interrupting the remaining neurons.
