# Export results

Exports write selected results directly into one folder; they do not create nested package directories. Choose the graph scope, select file types, then click the matching **Export** button.

## Scopes

| Section | Select | Export behavior |
| --- | --- | --- |
| All Neurons | Available population/all-neuron graph types | Writes selected current all-neuron figures. |
| Individual Neurons | Available individual graph types | Writes selected graph panels for the displayed neuron. |
| Individual Neurons + Repeat selections for every analyzed neuron | Same graph types | Recreates and writes those selected panels for each analysed neuron. |

The individual-neuron choices for Run Model and Run Full Model are disabled until their phase caches exist. Selecting them for every neuron can require a model fit per neuron, so it is intentionally explicit.

## File formats

| Format | File type | Purpose | Resource impact |
| --- | --- | --- | --- |
| PNG | `.png` | Presentation-ready raster graph. | Fast visual export; rendered at export resolution. |
| SVG | `.svg` | Editable vector graph suitable for publication. | Can be slower/larger for dense figures. |
| PKL | `.pkl` | Graph data and analysis values for a trusted Python environment. | Optional; can be much larger and slower because it retains numerical payloads. |

PNG and SVG are selected by default. PKL is opt-in so a normal visual export does not duplicate large numerical arrays in memory and on disk. NPY and Zarr are not export choices: they describe durable analysis caches; see [Cache Storage and Disk Planning](../reference/storage.md).

## Names and orientation files

Files are written directly in the destination folder with a stable prefix:

```text
shankX_unitY_graph-name.ext
```

For example:

```text
shank2_unit14_orientation_tuning.png
shank2_unit14_orientation_tuning.svg
shank2_unit14_orientation_tuning.pkl
```

All-neuron graphs use an `all_neurons_` prefix when no individual unit applies. Orientation exports preserve the `_orientation_tuning` suffix. Existing files with the same name are replaced by a subsequent export to that folder, so use a new destination or rename a completed set when you want to retain variants.

## Performance and memory

For the fastest, lowest-memory export, choose only the required graph types and leave PKL unchecked. The exporter renders PNG/SVG in background workers, bounds the serialized-figure queue to active writers, and releases restored figures after each write. Complex SVGs and all-neuron repeated exports still take time because every requested figure must be constructed and rendered.

The normal task completion notification applies to exports as well: Waven flashes the taskbar and plays a system sound on completion, failure, or cancellation.
