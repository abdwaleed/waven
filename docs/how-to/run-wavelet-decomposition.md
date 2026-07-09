# Run wavelet decomposition

In the GUI, choose **Analysis scale** first, then click **Run Wavelet
Decomposition**. The button changes its work according to the selected scale.

With `coarse`, it prepares:

- coarse downsampled stimulus movie;
- coarse real phase wavelets;
- coarse imaginary phase wavelets;
- durable coarse RF cache.

With `full`, it prepares:

- full-resolution downsampled stimulus movie;
- full-model real and imaginary wavelets as `.npy` or `.zarr`.

The same work can be run from Python:

```python
import waven

config = waven.PipelineConfig.from_json("pipeline_config.json")

waven.prepare_stimulus_wavelets(
    config.analysis,
    library_path=config.gabor.coarse_save_path,
)
waven.prepare_full_model_wavelets(
    config.analysis,
    config.gabor,
    library_path=config.gabor.fine_save_path,
)
```

If a long run is interrupted, keep the same **Analysis scale** selected and run
the GUI button again. Existing outputs are shape-validated and reused.

## Outputs and shapes

The wavelet stage writes several large arrays. The names are historical, but the
axis meanings are stable:

| Output | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `<movie>_coarse_downsampled.npy` | `(n_frames, coarse_ny, coarse_nx)` | `bool` | Cropped binary movie for coarse RF analysis. Image arrays keep row/column order. |
| `dwt_videodata_0.npy` | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `float32` | Coarse real-phase coefficients. |
| `dwt_videodata_1.npy` | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `float32` | Coarse imaginary-phase coefficients. |
| `dwt_downsampled_videodata.npy` | `(3, n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | `float32` | Durable RF cache: real, imaginary, and combined coefficients. |
| `<movie>_downsampled.npy` | `(n_frames, NY, NX)` | `bool` | Full-grid binary movie for full-model wavelets. |
| `dwt_videodata2_r.npy` or `.zarr` | `(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)` | `float32` | Full-model real phase. |
| `dwt_videodata2_i.npy` or `.zarr` | `(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)` | `float32` | Full-model imaginary phase. |

The downsampled movie axes are `(frame, y, x)` because they are image arrays.
The wavelet axes are `(frame, x, y, feature...)` because they follow the Gabor
library convention. This difference is expected.

## Memory and disk intuition

Downsampled movies are compact because they are boolean:

```text
movie_bytes = n_frames * ny * nx * 1
```

Wavelets are dense `float32`, so every value is 4 bytes. A full-model phase uses:

```text
full_phase_bytes =
  n_frames * NX * NY * n_orientations * n_sigmas * n_frequencies * 4
```

The real and imaginary phases are separate, so keeping both as `.npy` requires
twice that amount. Zarr stores the same logical array in chunks and can compress
repeated structure. It is usually easier to resume, inspect, and move in pieces,
but the uncompressed logical shape is unchanged.

## Resume points

The GUI checks completed artifacts before doing expensive work again. A rerun
can resume after:

- coarse movie downsampling;
- coarse real phase;
- coarse imaginary phase;
- durable coarse cache creation;
- full-grid movie downsampling;
- full real phase;
- full imaginary phase.

If a file exists with the wrong shape, regenerate it rather than using it. A
same-name file from a different `NX`, `NY`, orientation, sigma, or frequency
configuration can silently distort downstream plots if it is not rejected.
