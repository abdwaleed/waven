# Experiment folder reference

The canonical experiment-folder layout, the difference between inputs and
GUI-created outputs, and the required ephys/2-photon source layouts are in
[First GUI Analysis](../tutorials/first-gui-analysis.md).

In short, `input/` contains raw data, a stimulus movie, and optionally an
existing neural cache; `cache/` contains reusable stimulus-side products; and
`output/` contains results for the current neural-data analysis. The GUI can
also store a reference to a source folder outside the project instead of
copying that source data.

See [Reusable Caches](storage.md) for cache-library versioning.
