# Prepare wavelet products

The GUI uses one analysis grid for the whole session. It reads the stimulus
movie metadata (width, height, frame count, FPS, and duration) in **Stimulus &
Metadata** and applies the selected percentage to both image dimensions. There are no
manual `NX`, `NY`, stimulus-FPS, or separate coarse/full-grid settings.

This stage is also the prerequisite for **Session Setup** neural alignment.
Ephys alignment refuses to guess the stimulus duration: it receives the exact
metadata duration from the selected movie after this stage is completed.

First click **Prepare Stimulus Cache** in **1 Stimulus & Metadata**. It writes one binary movie cache with
image order `(time, y, x)`. The three buttons in **4 Wavelet Products** then prepare
independent, Zarr-backed products for their consumers:

| Button | Consumer | Durable artifact(s) | Shape |
| --- | --- | --- | --- |
| **Prepare Coarse RF Power Cache** | Run Coarse RF Analysis | `coarse_rf_power.zarr` | `(time, x, y, orientation, sigma)` |
| **Prepare Run Model Phase Caches** | Run Model (Coarse RF) | `coarse_model_real.zarr`, `coarse_model_imag.zarr` | `(time, x, y, orientation, sigma)` each |
| **Prepare Run Full Model Phase Caches** | Run Full Model | `dwt_videodata2_r.zarr`, `dwt_videodata2_i.zarr` | `(time, x, y, orientation, sigma_full, frequency)` each |

This split is intentional. Coarse RF correlates neural activity with wavelet
power; it does not need to keep real and imaginary model phases. Run Model
uses only its named real/imaginary pair. Run Full Model uses its own
higher-feature-count pair while taking the preferred coarse RF locations as
seeds. Preparing one product therefore does not create the combined three-plane
`dwt_downsampled_videodata.npy` cache used by older releases.

## Backend-specific behavior

**Legacy** needs the flattened Gabor libraries. **Prepare Gabor Assets** builds
the coarse coupled library and the fine independent-frequency library before
the relevant phase product is made. Its displayed size estimate refers to those
libraries.

**Convolution** does not materialize those flattened libraries. The same button
instead creates compact coarse and fine kernel caches. The Wavelet Products tab still
shows the same three consumer-oriented buttons, but its size estimate describes
the Zarr phase/power products rather than a legacy library.

## Axis convention and safety

The downsampled movie is `(time, y, x)` because it is an image sequence. Every
wavelet product is `(time, x, y, ...)` because it follows the Gabor feature
convention. The GUI derives both from the same metadata-based grid and validates
the full expected shape before reuse. A stale artifact with a different movie,
percentage, backend, orientation count, sigma list, phase list, or frequency
list is regenerated rather than passed downstream.

Internal wavelet artifacts are Zarr. They are chunked and streamed so a large
product remains disk-backed; the application never promotes an entire product
to RAM merely because it is being reused. The terminal reports the exact
artifact, logical shape, and resume decision. Temporary RF phases are Zarr too
and are removed after `coarse_rf_power.zarr` is safely completed.

## Disk intuition

For a float32 coarse product, the uncompressed logical size is

```text
time * x * y * orientations * sigmas * 4 bytes
```

`coarse_rf_power.zarr` stores one such product. Run Model stores two, and Run
Full Model stores two products with an additional frequency axis. Zarr
compression can reduce on-disk use, but planning should use the uncompressed
equivalent shown by the GUI. Higher percentages grow both `x` and `y`, so disk
growth is approximately quadratic in the percentage.
