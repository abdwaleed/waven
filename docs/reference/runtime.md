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

Run `python scripts/benchmark_runtime.py` from the repository root for a small
synthetic comparison of CPU/GPU Coarse RF and the STA backends on the current
machine. It performs no experiment-cache writes and is the recommended way to
decide whether optional FFT or fast-precision modes are worthwhile locally.

### Performance and hardware choices

The GUI exposes every runtime choice in **Session Configuration → Performance &
Hardware**.  The safe defaults require no configuration, apply to the next
action without restarting the GUI, and are saved in `gui.performance` when you
save the current GUI inputs and parameters. The matching environment variables
remain supported for unattended or command-line launches. Set a value to `0`
only when diagnosing a machine-specific driver or filesystem issue.

| Flag | Default | Effect |
| --- | --- | --- |
| `WAVEN_AUTOTUNE` | on | Measures early convolution chunks and adapts only within the safe batch ceiling. |
| `WAVEN_PREFETCH` | on | Enables one bounded input prefetch worker and downsampling resize overlap. |
| `WAVEN_ASYNC_WRITER` | on | Enables a single bounded wavelet-output writer. |
| `WAVEN_TIME_MAJOR_CONV` | on | For direct coarse-RF power, reads/uploads each frame chunk once before applying its bounded filter groups. Falls back to the group-major schedule if the combined kernel banks exceed a safe live-memory budget. |
| `WAVEN_RF_GPU` | on, if CUDA is present | Accumulates Coarse RF feature/response cross-products on GPU when the current tile fits. Each tile falls back to CPU on allocation failure. |
| `WAVEN_MULTI_GPU` | off | Explicitly enables PyTorch batch-parallel convolution across all detected CUDA GPUs. Leave this off unless all GPUs are dedicated to the analysis. |
| `WAVEN_TORCH_COMPILE` | off | Experimental `torch.compile` runner for long, fixed-shape convolution jobs. The first chunks are slower while the runner compiles; disable after a compiler/driver issue. |
| `WAVEN_AMP` | off | Explicit Tensor Core float16 autocast for convolution only. It changes convolution round-off, so use only after validating a representative run against default precision. |
| `WAVEN_STA_FFT` | off | Experimental bounded CUDA FFT calculation of the actual STA maps when the movie is already in RAM and at least eight lags are requested. The circular-shuffle null retains exact batched GEMMs. |

Multi-GPU is deliberately opt-in: it distributes independent frame batches,
then gathers them in original order. It does not change filter settings or
array layout, and a setup failure automatically continues on the primary GPU.
GPU RF acceleration also preserves the CPU algorithm's sufficient statistics;
only the floating-point matrix multiply location changes.

### Optional two-photon timeline roots

The GUI also exposes **Advanced 2-photon data discovery** in Session
Configuration. It supplies the optional `WAVEN_SUBJECT_DIRS` process setting
used by the Suite2p timeline adapter. Enter one or more dataset-root folders,
separated by the operating system's path separator (`;` on Windows, `:` on
Linux/macOS), then choose **Apply Suite2p Folders**. This is only needed when a
two-photon timeline must be discovered outside the project `Dir`; it does not
alter neural data, alignment mathematics, cache content, or any model input.
Leaving it empty removes the temporary override, which avoids carrying paths
from another computer into a new session.

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
