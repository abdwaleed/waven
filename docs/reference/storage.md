# Cache storage and disk planning

Waven treats durable arrays as caches with provenance. A cache is reused only
when the movie, visual coverage, sampling grid, Gabor axes, and requested
product agree with the current action. This prevents quietly mixing results
from different experiments or resolution settings.

## Choose NPY or Zarr for user-selectable caches

| Format | Best use | Benefits | Trade-off |
| --- | --- | --- | --- |
| **NPY** | Simple local arrays and fast sequential scans | One file, easy NumPy interchange, memory-mappable without loading all data. | No chunk-level compression or recovery; partial writes are less convenient. |
| **Zarr** | Large, chunked, compressed, or resumable caches | Reads/writes only needed chunks, supports compression and safer chunk-level continuation. | Many files/metadata entries; small arrays can have more overhead. |

The GUI offers this choice for the prepared stimulus cache and a freshly
created neural cache. Its **Wavelet storage format** controls the durable Coarse
RF correlation result when that result is written to disk. Convolutional power
and real/imaginary phase products remain Zarr because their chunked layout is
required for their workload. NPY and Zarr do **not** describe graph exports;
the Export tab writes PNG, SVG, and optional PKL only.

For a large experiment, prefer Zarr. For a small, self-contained cache you want
to open directly with NumPy, NPY is often simpler. Changing cache format does
not change the scientific result.

## Cache products and consumers

| Product | Typical path | Type / shape | Created by | Consumed by |
| --- | --- | --- | --- | --- |
| Downsampled stimulus | `cache/wavelets/coarse/` | binary NPY/Zarr `(frames, y, x)` | Prepare Stimulus Cache | All wavelet products; PSTH/STA. |
| Aligned neural cache | `input/neural_cache/` | matching `spikes` `(trials, frames, neurons)` and `pos` | Create pos/spikes Cache or user input | Coarse RF, inspection, models. |
| Convolution kernels | `cache/gabor/` | compact kernel-cache files | Prepare Convolution Kernels | Wavelet cache preparation. |
| Coarse RF power | `cache/wavelets/coarse/coarse_rf_power.zarr` | float32 `(frames, x, y, orientations, sigmas)` | Prepare Coarse RF Cache | Run Coarse RF Analysis. |
| Run Model phase pair | `cache/wavelets/coarse/coarse_model_{real,imag}.zarr` | two float32 arrays `(frames, x, y, orientations, sigmas)` | Prepare Run Model Phase Caches | Optional Run Model inspection graphs. |
| Full Model phase pair | `cache/wavelets/full/dwt_videodata2_{r,i}.zarr` | two float32 arrays `(frames, x, y, orientations, sigmas, frequencies)` | Prepare Run Full Model Phase Caches | Optional Full Model inspection graphs. |
| RF correlation result | Coarse wavelet folder | in-memory or NPY/Zarr when large | Run Coarse RF Analysis | Population and individual-neuron plots. |
| Plot cache | `output/plots/plot_cache.pkl.gz` | compressed plot state | Plot/cache actions | Faster restoration of compatible plots. |
| Recovery cache | `output/recovery_cache/` | task checkpoints and completed tiles | Long task runner | Safe cancellation/restart. |

Image arrays use `(y, x)` order; feature/RF arrays use `(x, y)` after the time
axis. This distinction is intentional: image rendering follows row/column
order while the feature grid follows azimuth/elevation indexing.

## Estimate disk before starting

The GUI shows estimates for the selected product. The following uncompressed
upper bounds explain why full-model products grow quickly. All feature products
are float32, so multiply elements by **4 bytes**.

```text
stimulus_bytes = frames × grid_y × grid_x × 1

coarse_power_bytes = frames × grid_x × grid_y × orientations × sigmas × 4

run_model_phase_pair_bytes =
    2 × frames × grid_x × grid_y × orientations × sigmas × 4

full_model_phase_pair_bytes =
    2 × frames × grid_x × grid_y × orientations × full_sigmas × frequencies × 4

rf_correlation_bytes =
    neurons × grid_x × grid_y × orientations × sigmas × frequencies × 4
```

Here `orientations` is `N_thetas`, `sigmas` is the length of `Sigmas`, and
`full_sigmas` is the merged full-model size list. The actual Zarr footprint can
be smaller because of compression, but it can also require temporary chunks and
metadata. Treat the GUI estimate as the planning baseline, not a guaranteed
compressed size.

### Free-space recommendation

Reserve at least:

```text
2 × estimated largest product
+ raw movie size
+ aligned neural cache size
+ every already-completed cache product you intend to keep
```

The `2 ×` headroom covers partial chunks, retries, recovery information, and
normal filesystem overhead. If you plan to retain both Run Model and Full Model
phase pairs, count both; they are separate products. Avoid network shares for
active cache creation when possible—local SSD/NVMe storage gives much more
predictable chunk throughput.

## RAM and safe reuse

The cache system limits peak RAM by reading arrays in chunks or through memory
maps. It is normal for RAM to rise during a tile calculation, image render, or
BLAS operation, but the complete wavelet tensor is not intentionally loaded as
one RAM array. **32 GB RAM or more** is recommended for large sessions.

The optional **RAM acceleration cache** retains only safely sized, reused
model-phase and PSTH/STA inputs for later actions; preparation itself stays
disk-backed and oversized arrays stay on disk. Disable it if another program
needs memory. See [System Configuration](gui.md#system-configuration) for the
other optional speed controls.
