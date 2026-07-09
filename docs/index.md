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

| Object | Shape | Meaning |
| --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | Neural activity aligned to stimulus frames. |
| `neuron_pos` | `(n_neurons, 2)` or `(n_neurons, 3)` | Anatomical neuron positions. |
| coarse wavelets | `(3, n_frames, nx, ny, n_orientations, n_sigmas)` | Real, imaginary, and magnitude-like wavelet cache. |
| full wavelets | `(n_frames, nx, ny, n_orientations, n_sigmas, n_frequencies)` | Full-resolution wavelet arrays for model fitting. |

## Recommended first path

1. Read [Install waven](how-to/install.md).
2. Follow [First GUI Analysis](tutorials/first-gui-analysis.md).
3. Review [Pipeline Intuition](explanation/pipeline-intuition.md).
4. Use the [Reference](reference/index.md) when extending code or writing scripts.

!!! note
    The older Sphinx `source/` folder is kept as historical source material.
    New authored documentation lives in `docs/`; the built site is written to
    `site/`.
