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

For a selected STA export, Waven calculates a small RAM-bounded group of
neurons at a time using vectorized matrix products, then writes each neuron's
usual PNG, SVG, pickle, array, and manifest files. Batch export also avoids
unneeded Tk canvas refreshes; the saved graph contents are unchanged.

Export folders use compact, collision-safe directory names so batch exports
remain below typical Windows path-length limits. The complete graph title and
the list of generated files remain in each graph's `manifest.json`; a batch
`export_manifest.json` records any individual neurons that could not be
exported without interrupting the remaining neurons.
