# Wavelets

Wavelet modules handle the most expensive data transformations in the project:
building Gabor filter banks and projecting every stimulus frame onto those
filters.

## Library shapes

| Function family | Output shape | dtype | Purpose |
| --- | --- | --- | --- |
| `makeFilterLibrary` | `(coarse_x, coarse_y, n_orientations, n_sigmas, n_phases, coarse_x * coarse_y)` | `float16` | coupled size/frequency library for coarse RF |
| `makeFilterLibrary2` | `(analysis_x, analysis_y, n_orientations, n_sigmas, n_frequencies, n_phases, analysis_x * analysis_y)` | `float16` | independent-frequency fine library |
| `waveletDecomposition` | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `float32` | one coarse phase at a time |
| `waveletDecompositionFull` | `(n_frames, analysis_x, analysis_y, n_orientations, n_sigmas, n_frequencies)` | `float32` | one full-model phase at a time |

Libraries store filters as flattened pixel vectors. Wavelet outputs store time
courses after multiplying movie frames by those filters. This is why library
size grows quadratically with grid area, while wavelet size grows linearly with
frames and feature count.

## Axis convention

Downsampled movies use image order `(frame, y, x)`. Wavelet outputs use feature
order `(frame, x, y, orientation, sigma, frequency)`. The transposition is
intentional and follows the filter-library indexing.

::: waven.wavelets.filters

::: waven.wavelets.decomposition
