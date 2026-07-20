# Export results

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
azimuth, orientation/size/frequency tuning, and PSTH-weighted STA lag maps. It creates one
folder per graph: a multi-panel tuning dashboard is never stored as a single
graph-data bundle.

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
usual PNG, SVG, pickle, array, and manifest files. Batch export also avoids
unneeded Tk canvas refreshes; the saved graph contents are unchanged. It keeps
a bounded queue of up to four prepared neurons with two writer workers by
default, so rendering and storage can overlap without accumulating every
neuron in RAM. Set `WAVEN_EXPORT_WORKERS` to a value from `1` to `4` only when
you have measured that your CPU and destination drive benefit from it.

The terminal reports per-neuron export timings for restoring figures, drawing,
PNG/SVG output, numeric arrays, pickles, manifests, file count, and bytes. Use
those timings to choose the appropriate preset rather than assuming that a
particular format is the bottleneck.

Export folders use compact, collision-safe directory names so batch exports
remain below typical Windows path-length limits. The complete graph title and
the list of generated files remain in each graph's `manifest.json`; a batch
`export_manifest.json` records any individual neurons that could not be
exported without interrupting the remaining neurons.
