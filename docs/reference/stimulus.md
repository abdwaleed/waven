# Stimulus

The selected movie is the authoritative source for stimulus dimensions and
timing. **Stimulus & Metadata** reads width, height, frame count, FPS, and
duration, then prepares a binary movie cache with shape `(time, y, x)` using
the chosen percentage.

The same metadata is required by ephys alignment: frame count fixes the neural
cache frame axis and duration validates stimulus trial boundaries. The GUI does
not use manual `NX`, `NY`, stimulus FPS, frame-count, or duration fields.

Wavelet products use feature order `(time, x, y, ...)` and are split by their
consumer:

| Product | Shape | Consumer |
| --- | --- | --- |
| `coarse_rf_power.zarr` | `(time, x, y, orientation, sigma)` | Coarse RF |
| `coarse_model_real.zarr` + `coarse_model_imag.zarr` | same coarse shape | Run Model |
| `dwt_videodata2_r.zarr` + `dwt_videodata2_i.zarr` | `(time, x, y, orientation, full_sigma, frequency)` | Run Full Model |

All internal wavelet products are chunked Zarr arrays and are shape-validated
before reuse.
