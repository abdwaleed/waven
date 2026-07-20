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

### Responsive exports

Exports first snapshot the selected Matplotlib figures on the GUI thread, then
restore them on a headless Agg canvas in one background writer. PNG/SVG
rendering, per-axis cropping, NPY/Zarr array writing, pickles, and manifests
therefore do not block Tk's event loop or make Windows display **Not
Responding** during large exports. The writer is deliberately single-threaded:
simultaneous SVG rendering and many small-array writes contend for the same
CPU, disk, and Matplotlib global state, usually making a large export slower.
The serialized snapshot preserves the same figures, data payloads, and output
formats as the foreground exporter. When STA single-neuron graphs are included,
their bounded NumPy batch calculation also runs in a worker; only Tk-owned
figure construction and snapshotting remain on the UI thread.

During **Run Coarse RF Analysis**, firing-rate orientation tuning groups neurons
that share the same preferred wavelet feature. Each feature matrix is read once
from a disk-backed cache and multiplied against all matching neural responses
at once using NumPy/BLAS. OSI/gOSI still use the established per-neuron metric
functions, so this only changes data scheduling and not the firing-rate or
selectivity definitions.

Run `python scripts/benchmark_runtime.py` from the repository root for a small
synthetic comparison of CPU/GPU Coarse RF and PSTH-weighted STA timing on the
current machine. It performs no experiment-cache writes and is useful for
checking whether fast-precision convolution modes are worthwhile locally.

### Performance and hardware choices

The GUI exposes optional runtime choices in **Session Configuration →
Performance & Hardware**. Adaptive batch tuning, bounded input prefetch,
asynchronous cache writing, time-major coarse convolution, parallel Suite2p
plane loading, and GPU Coarse RF statistics are automatic core paths. They are
not GUI settings or supported environment overrides. Coarse RF uses CUDA when
available and safely completes the remaining work on CPU when a tile cannot fit.
The optional choices apply to the next action without restarting the GUI and
are saved in `gui.performance` when you save the current GUI inputs and
parameters.

| Flag | Default | Effect |
| --- | --- | --- |
| `WAVEN_RAM_ACCELERATION_CACHE` | off | Retains only safely sized reused later-analysis arrays in RAM. |
| `WAVEN_MULTI_GPU` | off | Enables PyTorch batch-parallel convolution only across compatible CUDA GPUs. Cards with different compute capability, or less than 75% of the primary card's estimated throughput or VRAM, are excluded automatically. |
| `WAVEN_TORCH_COMPILE` | off | Experimental `torch.compile` runner for long, fixed-shape convolution jobs. The first chunks are slower while the runner compiles; if setup or a compiled call fails, Waven retries eagerly and keeps the rest of the action eager. |
| `WAVEN_AMP` | off | Explicit Tensor Core float16 autocast for convolution only. It changes convolution round-off, so use only after validating a representative run against default precision. |

Multi-GPU is deliberately opt-in: it distributes independent frame batches,
then gathers them in original order. It does not change filter settings or
array layout, and a setup failure automatically continues on the primary GPU.
Some PyTorch/CUDA builds do not expose a GPU `clock_rate`; Waven handles that
case with a conservative throughput proxy instead of failing the switch.
GPU RF acceleration also preserves the CPU algorithm's sufficient statistics;
only the floating-point matrix multiply location changes.

### Cancellation and resume

Cancellation is cooperative: the active chunk finishes, then the task stops at
the next safe checkpoint. Convolution-backed **Prepare Coarse RF**, **Prepare
Run Model**, and **Prepare Run Full Model** retain their partial Zarr caches and
their recovery manifest. A rerun reuses completed phase artifacts and, for the
direct Coarse RF cache, skips every completed frame/filter output tile. Partial
caches are never considered analysis-ready until their normal artifact metadata
is written after a successful run.

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
