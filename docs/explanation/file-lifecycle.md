# File lifecycle

The GUI keeps generated data inside the project layout and records a small
shape/parameter sidecar beside reusable artifacts. A product is reused only when
its logical shape and fingerprint match the current movie metadata, percentage,
backend, and filter settings.

| Artifact | Shape | Storage | Lifecycle |
| --- | --- | --- | --- |
| stimulus cache | `(time, y, x)` | user-selected NPY or Zarr | reusable input to all later stimulus work |
| neural `spikes` + `pos` cache | `(trials, frames, units)` + positions | selected NPY or Zarr | reusable aligned neural source |
| legacy Gabor library | feature-grid lookup table | NPY or Zarr | reusable only by Legacy backend |
| convolution kernel cache | compact filters | cache artifact | reusable only by Convolution backend |
| Coarse RF power | `(time, x, y, orientation, sigma)` | `coarse_rf_power.zarr` | used only by Coarse RF analysis |
| Run Model phases | same coarse phase shape, real + imaginary | named Zarr pair | used only by Run Model |
| Run Full Model phases | `(time, x, y, orientation, full_sigma, frequency)`, real + imaginary | named Zarr pair | used only by Run Full Model |

RF-only real/imaginary phases are temporary Zarr stores. They are converted to
the power product in bounded chunks and removed after successful completion.
This prevents the prior three-plane combined NPY cache from being created and
keeps hidden large cache work disk-backed.

The GUI never treats a same-named cache as valid solely because it exists. A
movie with a changed frame count, a new percentage, or a changed feature axis
must regenerate the dependent product.
