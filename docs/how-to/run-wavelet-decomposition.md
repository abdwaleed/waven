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

## Performance and responsiveness

The convolution backend fuses the two real/imaginary phases when preparing
**Coarse RF Power**. It performs one convolution bank per frame chunk and
computes `real² + imaginary²` before writing the final power cache. This avoids
the former duplicate movie read, transfer, and temporary phase products while
preserving the same power definition and output shape. As with any reordered
float32 GPU convolution, least-significant-bit roundoff may vary by hardware.

For all three convolution products, Waven overlaps a bounded next-chunk read
with the current GPU convolution and uses one bounded writer for completed
chunks. The application never queues an unbounded number of frames. It also
derives a conservative batch ceiling from currently free RAM/VRAM and adapts
the first measured batches within that ceiling. At the end of the action, the
terminal reports input, compute/transfer, and output throughput so a slow run
can be attributed to decoding, GPU compute, or disk writes.

On a multi-GPU workstation, enable **Use compatible GPUs** in **Session
Configuration → Performance & Hardware** before starting a convolution action.
This is off by default because leaving a display or shared-workload GPU
saturated can reduce desktop responsiveness. Waven only combines cards with
matching CUDA compute capability and at least 75% of the primary card's
estimated convolution throughput and VRAM, preventing a slower card from
holding up every synchronous DataParallel batch. The same choice is saved with
the current GUI inputs (and remains available as `WAVEN_MULTI_GPU=1` for
unattended launches). If the cards are not a suitable group, the terminal
reports why and the action continues on the primary GPU. The optional
compiled-convolution setting also falls back to eager convolution if compiler
setup or the first compiled batch fails, so a driver/compiler mismatch does not
abort the decomposition.

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
