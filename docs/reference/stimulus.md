# Stimulus

Stimulus helpers load cached wavelet arrays and convert historical files into
the shapes expected by RF analysis and nonlinear models.

## Cache files

| File | Shape | dtype | Consumed by |
| --- | --- | --- | --- |
| `dwt_videodata_0.npy` | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `float32` | coarse cache builder |
| `dwt_videodata_1.npy` | same as real phase | `float32` | coarse cache builder |
| `dwt_downsampled_videodata.npy` | `(3, n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `float32` | RF analysis and simple model |
| `dwt_videodata2_r.npy` / `.zarr` | `(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)` | `float32` | full model |
| `dwt_videodata2_i.npy` / `.zarr` | same as real phase | `float32` | full model |

`coarseWavelet()` returns three arrays: real, imaginary, and combined
coefficients. The combined array is the feature tensor used for coarse RF
correlation.

## Practical interpretation

Use this page when a file exists on disk and you need to know how it will be
loaded. If you are creating the files from a movie, start with the
[Wavelets](wavelets.md) or [Pipeline](pipeline.md) reference pages instead.

::: waven.stimulus.wavelet_cache

::: waven.stimulus.full_model
