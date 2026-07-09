# waven documentation

`waven` analyzes neural responses to visual stimuli with Gabor-wavelet features.
It supports a GUI workflow for exploratory analysis and a scriptable Python
pipeline for reproducible runs.

The documentation is organized with the Divio structure:

- **Tutorials** walk through complete workflows for new users.
- **How-to guides** solve focused tasks such as installing the package, building
  Gabor libraries, or exporting plots.
- **Explanation** pages describe the intuition behind the stimulus transform,
  receptive-field search, model fitting, and file lifecycle.
- **Reference** pages are generated from Google-style docstrings with
  `mkdocstrings`.

## Core data model

Most analysis stages share a small set of array conventions:

| Object | Shape | dtype | Meaning |
| --- | --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | floating numeric | Neural activity aligned to stimulus frames. Axis 1 is always the stimulus-frame axis used for correlations and models. |
| `neuron_pos` | `(n_neurons, 2)` or `(n_neurons, 3)` | floating numeric | Anatomical positions. Two-photon data usually stores imaging-plane coordinates; electrophysiology data can include shank/depth metadata. |
| downsampled movie | `(n_frames, ny, nx)` | `bool` | Binary stimulus after cropping to `Analysis Coverage` and resizing to the analysis grid. Movie files keep image order as row, column. |
| Gabor library | coarse: `(nx, ny, n_orientations, n_sigmas, n_phases, nx * ny)` | `float16` | Spatial filters flattened over pixels. Libraries are large but reusable when the grid and feature axes match. |
| fine Gabor library | `(NX, NY, n_orientations, n_sigmas, n_frequencies, n_phases, NX * NY)` | `float16` | Full-model filter bank when frequencies are independent from size. |
| coarse wavelet cache | `(3, n_frames, nx, ny, n_orientations, n_sigmas)` | `float32` | Durable cache containing real phase, imaginary phase, and combined energy-like coefficients. |
| full wavelets | `(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)` | `float32` | Full-resolution wavelet arrays for nonlinear model fitting; written separately for real and imaginary phases. |
| RF tensor | `(n_neurons, nx, ny, n_orientations, n_sigmas, n_frequencies)` | `float32` or `float64` | Pearson-correlation receptive fields. Tuning curves are slices through this tensor around each neuron's preferred feature. |

The docs use uppercase `NX`/`NY` for the full analysis grid from the
configuration and lowercase `nx`/`ny` for a concrete array after any coarse
downsampling. Movie arrays use image order `(y, x)`, while Gabor and RF tensors
use feature order `(x, y)`. When a shape looks transposed, check which family the
array belongs to before assuming it is wrong.

For uncompressed arrays, size is approximately:

```text
number_of_values * dtype_bytes
```

For example, a full wavelet phase with
`n_frames=54000`, `NX=135`, `NY=54`, `n_orientations=8`, `n_sigmas=6`, and
`n_frequencies=4` contains about 75.6 billion `float32` values, or roughly
302 GB before filesystem and compression effects. This is why `waven` reuses
existing files, shape-validates caches, and supports Zarr for full-model
wavelets.

## Recommended first path

1. Read [Install waven](how-to/install.md).
2. Follow [First GUI Analysis](tutorials/first-gui-analysis.md).
3. Review [Pipeline Intuition](explanation/pipeline-intuition.md).
4. Use the [Reference](reference/index.md) when extending code or writing scripts.

!!! note
    Authored documentation lives in `docs/`. The rendered static site is written
    to `site/`, and the previous Sphinx `source/` tree has been retired in favor
    of MkDocs plus `mkdocstrings`.
