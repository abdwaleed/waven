# Runtime

Runtime helpers keep long scientific jobs practical. They estimate safe chunk
sizes, detect usable RAM/GPU resources, support cancellation, and keep the host
awake during long GUI tasks.

## Sustained-throughput controls

The convolution wavelet actions use a bounded three-stage schedule. One worker
prefetches the next movie/Zarr chunk while the current chunk is convolved; a
single bounded writer commits completed chunks. The queues intentionally retain
only a few chunks. This overlaps disk/CPU/GPU work without allowing a fast
producer to consume all RAM or make Windows unresponsive.

At the start of each convolution action, the frame batch is derived from
*currently free* RAM/VRAM rather than installed memory. The first real chunks
then adjust the batch within that preflight ceiling: quick batches grow toward
the ceiling; slow batches shrink. Changing a batch boundary changes scheduling
only. A convolution is still evaluated independently for every frame with the
same kernels, so the scientific definition and array layout are unchanged. As
with any reordered float32 GPU/CPU convolution, least-significant-bit roundoff
can vary across hardware or batch plans; it is not a change in the model or
analysis inputs.

The terminal prints a rate-limited live telemetry line during downsampling,
convolution, and Coarse RF correlation, followed by a final summary. It reports
wall-clock time and measured input, compute/transfer, and output throughput.
That line is the reliable way to identify whether a particular run is limited
by decoding, GPU convolution, or storage rather than guessing from CPU/GPU
percentages alone.

### Optional hardware flags

These are process environment flags for advanced deployments; the safe defaults
require no configuration. Set a flag to `0` to disable an optimization while
diagnosing a machine-specific driver or filesystem issue.

| Flag | Default | Effect |
| --- | --- | --- |
| `WAVEN_AUTOTUNE` | on | Measures early convolution chunks and adapts only within the safe batch ceiling. |
| `WAVEN_PREFETCH` | on | Enables one bounded input prefetch worker and downsampling resize overlap. |
| `WAVEN_ASYNC_WRITER` | on | Enables a single bounded wavelet-output writer. |
| `WAVEN_RF_GPU` | on, if CUDA is present | Accumulates Coarse RF feature/response cross-products on GPU when the current tile fits. Each tile falls back to CPU on allocation failure. |
| `WAVEN_MULTI_GPU` | off | Explicitly enables PyTorch batch-parallel convolution across all detected CUDA GPUs. Leave this off unless all GPUs are dedicated to the analysis. |

Multi-GPU is deliberately opt-in: it distributes independent frame batches,
then gathers them in original order. It does not change filter settings or
array layout, and a setup failure automatically continues on the primary GPU.
GPU RF acceleration also preserves the CPU algorithm's sufficient statistics;
only the floating-point matrix multiply location changes.

## Why this matters

Wavelet decomposition streams through arrays that can be hundreds of gigabytes
logically. Chunk sizing controls peak RAM and GPU memory use. The goal is to
process enough frames/filters per batch for speed without forcing the operating
system to page memory or crash the Python process.

| Helper area | Used by | Controls |
| --- | --- | --- |
| performance | wavelet and cache builders | RAM checks, chunk sizes, GPU/CPU route |
| task control | GUI and long loops | cancellation checks, progress messages |
| keep awake | GUI long tasks | prevents sleep during active processing |

::: waven.runtime.performance

::: waven.runtime.task_control

::: waven.runtime.keep_awake
