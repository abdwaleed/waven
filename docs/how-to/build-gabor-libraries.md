# Build Gabor libraries

Build the coarse and fine libraries before decomposing the stimulus.

```python
import waven

config = waven.PipelineConfig.from_json("pipeline_config.json")

coarse = waven.create_coarse_gabor_library(config.gabor)
fine = waven.create_fine_gabor_library(
    config.gabor,
    extra_sigmas=config.analysis.sigmas_full_model,
)
```

The coarse library is used for fast receptive-field screening. The fine library
is used for full-resolution model fitting.

In the GUI, first complete **1 Stimulus & Metadata**, then click **Prepare
Gabor Assets (Coarse RF + Full Model)**. The displayed grid is read-only: it is
calculated from movie metadata and the selected percentage. Enter phase offsets in
degrees (for example `[0, 90]`); the application converts them to radians only
when constructing Gabor kernels.

## Legacy library versus convolution kernels

The active wavelet backend changes what the Gabor action does. With **legacy**,
the action creates the flattened Gabor library described below; therefore the
GUI displays the NPY/Zarr library selector and its size estimate. With
**convolution**, the action builds a compact convolution-kernel cache instead.
The library format and large-library estimate are intentionally hidden because
no flattened library is produced. This prevents a displayed size estimate from
being mistaken for the size of the convolution cache.

| Product | Built library | Folder field used |
| --- | --- | --- |
| Coarse RF / Run Model | coarse coupled library | `Coarse Library Path` |
| Run Full Model | fine independent-frequency library | `Fine Library Path` and `Library Path` |

Existing `.npy` or `.zarr` libraries are reused only when their shape matches the
current configuration. The GUI fields are folders. Generated libraries use
deterministic names inside those folders, but when loading an existing library
the code also accepts a single unambiguous `.npy` or `.zarr` artifact in the
expected folder.

## What gets written

Gabor libraries are lookup tables of spatial filters. Each filter is flattened
over the analysis grid so the decomposition step can multiply a movie frame by a
large bank of filters.

| Library | Conventional folder/file | Shape | dtype | Used by |
| --- | --- | --- | --- | --- |
| Coarse | `cache/gabor/coarse/gabor_library_coarse.npy` | `(coarse_nx, coarse_ny, n_orientations, n_sigmas, n_phases, coarse_nx * coarse_ny)` | `float16` | coarse RF search |
| Fine with independent frequencies | `cache/gabor/full/gabor_library_fine.npy` | `(x, y, n_orientations, n_sigmas_total, n_frequencies, n_phases, x * y)` | `float16` | full-model wavelets |
| Fine without independent frequencies | `cache/gabor/full/gabor_library_fine.npy` | `(x, y, n_orientations, n_sigmas_total, n_phases, x * y)` | `float16` | full-model wavelets with one coupled frequency |

`n_sigmas_total` is the union of `Sigmas` and `Sigmas Full Model` when the fine
library is built from the high-level pipeline. This lets the same file serve
both historical settings and newer full-model size choices.

## Size estimates

Libraries are stored as `float16`, so each value is 2 bytes:

```text
coarse_bytes =
  coarse_nx * coarse_ny * n_orientations * n_sigmas * n_phases
  * (coarse_nx * coarse_ny) * 2

fine_bytes =
  NX * NY * n_orientations * n_sigmas_total * n_frequencies * n_phases
  * (NX * NY) * 2
```

The final flattened-pixel axis is why libraries become large quickly as the grid
increases. Doubling both `NX` and `NY` roughly multiplies the fine library by
16, because both the number of filter centers and the number of pixels per
filter grow.

## Reuse rules

It is safe to reuse a library only when these axes match the analysis:

- grid size (`NX`, `NY`, or the derived coarse grid);
- `N_thetas`;
- `Sigmas` and, for fine libraries, `Sigmas Full Model`;
- `Frequencies`;
- `Phases`.

The GUI validates shapes before reuse and writes small metadata sidecars for
new artifacts. It cannot infer whether a same-shaped external library was made
with a different scientific convention, so keep library folders organized by
experiment family.
