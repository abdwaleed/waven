# Reusable Caches

Use this page when you want to avoid repeating expensive stimulus preparation.
The GUI is responsible for creating, selecting, validating, and versioning
caches; you normally do not need to move cache files by hand.

## The useful rule

It is advisable to reuse a stimulus-derived cache for a new neural recording
when the **stimulus-side experiment is unchanged**. You can create a new neural
cache from new raw data, then analyse it against an existing Coarse RF cache.

A prepared stimulus, Gabor kernel cache, or Coarse RF cache is reusable only
when these inputs match:

1. The underlying stimulus video.
2. Visual Field Coverage and Analysis Field Coverage.
3. The derived sampling grid.
4. Gabor orientations, sigmas, frequencies, phases, and Coarse RF frequency
   mode.

Changing raw neural data alone does not invalidate the stimulus-side cache.
Changing any item above does. WavEn records provenance and validates a selected
cache before analysis, so it will refuse a cache built for a different movie,
crop, grid, or filter bank.

## What is reusable and what is not

| Product | Reuse with new raw neural data? | Reason |
| --- | --- | --- |
| Downsampled stimulus movie | Yes, when the movie, coverage, and grid match. | It is derived only from the visual stimulus. |
| Gabor convolution kernels | Yes, when the grid and Gabor parameters match. | They are independent of neural activity. |
| Coarse RF power cache | Yes, when all stimulus-side compatibility inputs match. | It contains visual features, not neural responses. |
| Run Model / Full Model phase caches | Yes, when their stimulus-side settings match. | They contain phase-aware visual features, not neural responses. |
| Aligned neural `spikes` / `pos` cache | No. Create or validate a cache for the new recording. | It represents the specific neural data and frame alignment. |
| Coarse RF results, plots, and fitted models | Usually no. Re-run for the new neural recording. | They combine the selected visual cache with neural responses. |

## Cache libraries and versions

The **Coarse RF Cache Library Folder** and **Full-Model Cache Library Folder**
are libraries, not single-use folders.

```text
cache/wavelets/coarse/
├── stimulus_coarse_downsampled_p20.zarr     # shared prepared stimulus
├── coarse_rf_power.zarr                     # legacy/top-level cache version
└── cache-20260726-153012/                   # preserved alternative version
    ├── coarse_rf_power.zarr
    ├── coarse_model_real.zarr               # if requested
    └── coarse_model_imag.zarr               # if requested
```

When **Prepare Analysis Caches** finds existing products, the replacement dialog
has three outcomes:

| Choice | What happens |
| --- | --- |
| **Yes** | Allows replacement inside the currently selected cache version. Use only when you intend to update that version. |
| **No** | Creates a new timestamped subfolder, selects it as active, and writes the new cache there. Existing versions stay untouched. |
| **Cancel** | Does not start cache preparation. |

Use **Active Coarse RF Cache** to choose the version that **Run Coarse RF
Analysis** should use. The dropdown lists completed versions in the selected
library. Analysis then validates the selected cache against the current movie,
coverage, grid, and Gabor settings; it never guesses which version you meant or
silently switches to another one.

The shared downsampled movie remains at the library level, so choosing **No**
typically creates only a new wavelet-cache version rather than a duplicate
video cache.

## Cache products and consumers

| Product | Typical location | Type / shape | Created by | Used by |
| --- | --- | --- | --- | --- |
| Downsampled stimulus | `cache/wavelets/coarse/` | Binary NPY/Zarr `(frames, y, x)` | Prepare Stimulus Cache | Wavelets, neural validation, PSTH/STA. |
| Aligned neural cache | `input/neural_cache/` | Matching `spikes` `(trials, frames, neurons)` and `pos` arrays | Create or Validate Neural Cache | Coarse RF and inspection. |
| Convolution kernels | `cache/gabor/` | Compact kernel-cache files | Prepare Gabor Assets | Wavelet-cache preparation. |
| Coarse RF power | `cache/wavelets/coarse[/cache-timestamp]/coarse_rf_power.zarr` | Float32 `(frames, x, y, orientations, sigmas[, frequencies])` | Prepare Analysis Caches | Run Coarse RF Analysis. |
| Run Model phase pair | `cache/wavelets/coarse[/cache-timestamp]/coarse_model_{real,imag}.zarr` | Two float32 phase arrays | Optional preparation checkbox | Run Model inspection. |
| Full Model phase pair | `cache/wavelets/full[/cache-timestamp]/dwt_videodata2_{r,i}.zarr` | Two float32 phase arrays with frequency axis | Optional preparation checkbox | Full Model inspection. |
| Plot cache | `output/plots/plot_cache.pkl.gz` | Compressed GUI result state | Analysis/plot actions | Faster compatible plot restoration. |
| Recovery cache | `output/recovery_cache/` | Task checkpoints and completed tiles | Long task runner | Safe cancellation/restart. |

Image arrays use `(y, x)` order; feature and RF arrays use `(x, y)` after the
time axis. This is intentional: image rendering follows row/column order while
feature arrays follow azimuth/elevation indexing.

## NPY and Zarr

The GUI lets you choose NPY or Zarr for selected durable products. These are
cache-storage formats, not graph-export formats.

| Format | Best use | Benefits | Trade-off |
| --- | --- | --- | --- |
| **NPY** | Simple local arrays and fast sequential scans | One file, easy NumPy interchange, memory-mappable. | No chunk-level compression or resumable tiles. |
| **Zarr** | Large, compressed, or resumable arrays | Chunked I/O, compression, and safer continuation for large tasks. | Many metadata/chunk files; small arrays can have more overhead. |

Convolutional Coarse RF power and phase products remain Zarr because their
chunked layout is required for preparation and bounded analysis reads. PNG,
SVG, and optional PKL are **export** formats; see
[Export Results](../how-to/export-results.md).

## Plan disk space

The GUI shows uncompressed estimates based on the movie metadata, grid, and
filter bank. Treat the estimate as a planning baseline, not a guarantee: Zarr
compression depends on the data, and preparation may need temporary chunks.

For large work, reserve enough space for the raw movie, neural cache, the final
cache version, and working headroom. If you want to keep several scientific
configurations, keep each version rather than overwriting it, and name or record
the corresponding experiment purpose outside the timestamped folder.

Return to [GUI Onboarding](../tutorials/gui-onboarding.md) for the controls that
create and select these caches.
