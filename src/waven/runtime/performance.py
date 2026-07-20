"""Hardware-aware defaults for chunked CPU and GPU workloads.

Chunk sizes and worker counts are derived from currently available system
memory and GPU VRAM so that waven scales across laptops, workstations, and
servers without manual tuning.  All helpers are side-effect free and safe to
call repeatedly (results are cached where profiling would be expensive).
"""
from __future__ import annotations

import math
import os
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, Literal, Optional, Tuple

import psutil
import torch

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional protection for legacy installs
    threadpool_limits = None

ComputeDevice = Literal["cuda", "cpu"]
_reported_multi_gpu_decisions: set[Tuple[int, ...]] = set()
_nvml_lock = threading.Lock()
_nvml_module = None
_nvml_initialized = False
_nvml_unavailable = False


def _nvml():
    """Return an initialized optional NVML module without adding a dependency."""
    global _nvml_module, _nvml_initialized, _nvml_unavailable
    if _nvml_unavailable:
        return None
    with _nvml_lock:
        if _nvml_unavailable:
            return None
        try:
            if _nvml_module is None:
                import pynvml

                _nvml_module = pynvml
            if not _nvml_initialized:
                _nvml_module.nvmlInit()
                _nvml_initialized = True
            return _nvml_module
        except Exception:
            # GPU telemetry is diagnostic only.  PyTorch VRAM metrics remain
            # available when the driver exposes no NVML interface or the
            # optional package is not installed.
            _nvml_unavailable = True
            return None


def gpu_runtime_snapshot() -> list[Dict[str, object]]:
    """Return lightweight live per-GPU telemetry with safe NVML fallback.

    Values are sampled only by the GUI task monitor (every few seconds), not
    in hot numerical loops.  ``compute_utilization`` and PCIe rates are ``None``
    when NVML is unavailable rather than guessed from allocation size.
    """
    if not torch.cuda.is_available():
        return []
    nvml = _nvml()
    devices = []
    for index in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        except Exception:
            free_bytes = gpu_available_vram_bytes(index)
            total_bytes = gpu_vram_bytes(index)
        entry: Dict[str, object] = {
            "index": index,
            "name": str(torch.cuda.get_device_properties(index).name),
            "free_bytes": int(free_bytes),
            "total_bytes": int(total_bytes),
            "compute_utilization": None,
            "memory_utilization": None,
            "pcie_tx_bytes_per_second": None,
            "pcie_rx_bytes_per_second": None,
        }
        if nvml is not None:
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                utilization = nvml.nvmlDeviceGetUtilizationRates(handle)
                entry["compute_utilization"] = float(utilization.gpu)
                entry["memory_utilization"] = float(utilization.memory)
                # NVML PCIe throughput is reported in KiB/s on supported
                # desktop drivers.  Some consumer cards omit these counters.
                entry["pcie_tx_bytes_per_second"] = int(
                    nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_TX_BYTES)
                ) * 1024
                entry["pcie_rx_bytes_per_second"] = int(
                    nvml.nvmlDeviceGetPcieThroughput(handle, nvml.NVML_PCIE_UTIL_RX_BYTES)
                ) * 1024
            except Exception:
                pass
        else:
            # Recent PyTorch builds expose NVML-backed utilization methods even
            # when the optional Python ``pynvml`` package is absent.
            try:
                if hasattr(torch.cuda, "utilization"):
                    entry["compute_utilization"] = float(torch.cuda.utilization(index))
                if hasattr(torch.cuda, "memory_usage"):
                    entry["memory_utilization"] = float(torch.cuda.memory_usage(index))
            except Exception:
                pass
        devices.append(entry)
    return devices


@dataclass(frozen=True)
class ConvolutionWorkload:
    """Live memory requirements for one convolution frame batch.

    ``filter_channels`` is the actual convolution-bank width, while
    ``output_channels`` describes the postprocessed result.  Keeping both is
    essential for fused real/imaginary Gabor power, where the convolution bank
    is twice as wide as the final orientation result.
    """

    frame_count: int
    spatial_pixels: int
    filter_channels: int
    output_channels: int
    kernel_height: int = 1
    kernel_width: int = 1
    activation_dtype_bytes: int = 4
    output_buffer_count: int = 2

    def working_set_bytes(self, frames: Optional[int] = None) -> int:
        """Conservative device-resident bytes for ``frames`` of this workload."""
        frames = max(1, int(self.frame_count if frames is None else frames))
        pixels = max(1, int(self.spatial_pixels))
        filters = max(1, int(self.filter_channels))
        outputs = max(1, int(self.output_channels))
        output_buffers = max(1, int(self.output_buffer_count))
        dtype_bytes = max(1, int(self.activation_dtype_bytes))
        # Input is float32. The kernel copy/workspace reserve reflects the
        # module and cuDNN bookkeeping; activations include response, fused
        # output, and transfer staging.
        kernel_bytes = filters * max(1, int(self.kernel_height)) * max(1, int(self.kernel_width)) * 4 * 2
        activation_bytes = frames * pixels * (4 + dtype_bytes * (filters + outputs * output_buffers))
        return int(kernel_bytes + activation_bytes)

    def output_bytes(self, frames: Optional[int] = None) -> int:
        """Bytes of the postprocessed host-transfer result for ``frames``."""
        frames = max(1, int(self.frame_count if frames is None else frames))
        return int(
            frames
            * max(1, int(self.spatial_pixels))
            * max(1, int(self.output_channels))
            * max(1, int(self.activation_dtype_bytes))
        )


@dataclass(frozen=True)
class ConvolutionExecutionPlan:
    """Chosen execution mode for a concrete Gabor convolution workload."""

    strategy: Literal["single", "frame_parallel"]
    devices: Tuple[str, ...]
    frames_per_device: int
    estimated_device_bytes: int
    reason: str


class OperationTelemetry:
    """Low-overhead timing and throughput counters for one pipeline operation."""

    def __init__(self, label: str):
        self.label = str(label)
        self.started = time.perf_counter()
        self.seconds: Dict[str, float] = {}
        self.bytes: Dict[str, int] = {}
        self.operations: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_live_report = self.started

    def add(
        self,
        stage: str,
        seconds: float = 0.0,
        byte_count: int = 0,
        operation_count: int = 0,
    ) -> None:
        with self._lock:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + max(0.0, float(seconds))
            self.bytes[stage] = self.bytes.get(stage, 0) + max(0, int(byte_count))
            self.operations[stage] = self.operations.get(stage, 0) + max(0, int(operation_count))

    def _message(self, suffix: str = "") -> str:
        total = max(time.perf_counter() - self.started, 1e-9)
        with self._lock:
            stages = sorted(self.seconds.items())
            byte_counts = dict(self.bytes)
            operation_counts = dict(self.operations)
        if stages:
            dominant_stage, dominant_seconds = max(stages, key=lambda item: item[1])
            dominant_note = f" | largest measured stage: {dominant_stage} ({dominant_seconds / total:.0%})"
        else:
            dominant_note = ""
        pieces = [f"[perf] {self.label}{suffix} | elapsed {total:.1f}s{dominant_note}"]
        for stage, duration in stages:
            rate = byte_counts.get(stage, 0) / max(duration, 1e-9) / 1024**2
            rate_suffix = f", {rate:.1f} MiB/s" if byte_counts.get(stage, 0) else ""
            flop_rate = operation_counts.get(stage, 0) / max(duration, 1e-9) / 1e9
            flop_suffix = f", {flop_rate:.2f} GFLOP/s" if operation_counts.get(stage, 0) else ""
            pieces.append(f"{stage} {duration:.1f}s{rate_suffix}{flop_suffix}")
        if len(pieces) == 1:
            return f"  {pieces[0]}"
        return f"  {pieces[0]}\n     -> " + " | ".join(pieces[1:])

    def maybe_report(self, interval_seconds: float = 10.0) -> None:
        """Print a rate-limited live throughput snapshot for a long operation."""
        now = time.perf_counter()
        with self._lock:
            if now - self._last_live_report < float(interval_seconds):
                return
            self._last_live_report = now
        print(self._message(" (live)"))

    def report(self) -> None:
        print(self._message())


_ALWAYS_ENABLED_FEATURES = frozenset({
    "AUTOTUNE",
    "PREFETCH",
    "ASYNC_WRITER",
    "TIME_MAJOR_CONV",
    "2P_PARALLEL_IO",
    "RF_GPU",
})


def enabled_feature(name: str, default: bool = True) -> bool:
    """Return a runtime feature state.

    The core scheduling paths listed in ``_ALWAYS_ENABLED_FEATURES`` are part
    of Waven's execution contract rather than user preferences.  In
    particular, old saved GUI settings and inherited ``WAVEN_*`` environment
    variables cannot accidentally disable them.  Other flags retain their
    conservative environment-controlled behaviour.
    """
    normalized_name = str(name).upper()
    if normalized_name in _ALWAYS_ENABLED_FEATURES:
        return True
    value = os.environ.get(f"WAVEN_{normalized_name}")
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_zarr_codec_threads(max_threads: int = 8) -> Optional[int]:
    """Configure a bounded Blosc pool for large Zarr cache reads and writes.

    Numcodecs uses a process-wide native worker count.  A single writer thread
    otherwise leaves large cache compression/decompression serial, while an
    unbounded setting can make the GUI and GPU feeder contend with the codec.
    ``WAVEN_ZARR_THREADS`` overrides the automatic setting; the earlier
    coarse-specific name remains accepted for existing user configurations.
    """
    requested = os.environ.get(
        "WAVEN_ZARR_THREADS", os.environ.get("WAVEN_COARSE_ZARR_THREADS", "")
    ).strip()
    automatic = min(max(1, int(max_threads)), max(1, (os.cpu_count() or 2) - 1))
    try:
        threads = int(requested) if requested else automatic
    except ValueError:
        print(f"Invalid WAVEN_ZARR_THREADS={requested!r}; using automatic codec threads.")
        threads = automatic
    threads = max(1, min(32, threads))
    try:
        from numcodecs import blosc as numcodecs_blosc

        numcodecs_blosc.set_nthreads(threads)
        return threads
    except Exception:
        # Numcodecs is optional for non-Zarr workflows, and some legacy
        # releases expose no thread-control API.
        return None


def _gpu_descriptor(device_id: int) -> Dict[str, object]:
    """Return the hardware traits relevant to synchronous DataParallel work."""
    properties = torch.cuda.get_device_properties(int(device_id))
    # ``clock_rate`` is absent on some supported PyTorch/CUDA property objects.
    # Retain a deterministic, conservative score in that case rather than
    # making the opt-in multi-GPU switch fail before a convolution starts.
    multiprocessors = max(1, int(getattr(properties, "multi_processor_count", 1)))
    clock_rate = max(1, int(getattr(properties, "clock_rate", 1)))
    return {
        "id": int(device_id),
        "name": str(properties.name),
        "capability": (int(properties.major), int(properties.minor)),
        "vram_bytes": int(properties.total_memory),
        "free_vram_bytes": int(gpu_available_vram_bytes(int(device_id))),
        # SM count and core clock are a stable, inexpensive proxy for relative
        # convolution throughput.  Exact core counts vary by architecture, so
        # they intentionally are not inferred from marketing names.
        "throughput_score": multiprocessors * clock_rate,
    }


def select_compatible_multi_gpu_ids(
    descriptors: Iterable[Dict[str, object]],
    minimum_relative_score: float = 0.75,
    minimum_relative_memory: float = 0.75,
) -> Tuple[Tuple[int, ...], Tuple[str, ...]]:
    """Keep only GPUs that can usefully share synchronous ``DataParallel`` work.

    ``DataParallel`` splits each frame chunk, then waits for every GPU before
    gathering. A substantially slower/smaller card therefore throttles the
    whole batch. Different CUDA architectures also make mixed-precision and
    kernel behaviour less predictable, so they are deliberately not combined.
    """
    available = sorted((dict(item) for item in descriptors), key=lambda item: int(item["id"]))
    if not available:
        return (), ()
    # Prefer the card with the greatest currently useful capacity.  This is
    # still deterministic when descriptors omit live-free-memory information,
    # which keeps the helper convenient for tests and non-CUDA callers.
    primary = max(
        available,
        key=lambda item: (
            int(item.get("free_vram_bytes", item["vram_bytes"])) * float(item["throughput_score"]),
            int(item["vram_bytes"]),
            -int(item["id"]),
        ),
    )
    accepted = [int(primary["id"])]
    rejected = []
    for candidate in available:
        if int(candidate["id"]) == int(primary["id"]):
            continue
        device_id = int(candidate["id"])
        reasons = []
        if tuple(candidate["capability"]) != tuple(primary["capability"]):
            reasons.append(
                f"compute capability {candidate['capability']} differs from cuda:{primary['id']} {primary['capability']}"
            )
        score_ratio = min(float(candidate["throughput_score"]), float(primary["throughput_score"])) / max(
            float(candidate["throughput_score"]), float(primary["throughput_score"])
        )
        if score_ratio < float(minimum_relative_score):
            reasons.append(f"estimated convolution throughput is {score_ratio:.0%} of the primary GPU")
        memory_ratio = min(float(candidate["vram_bytes"]), float(primary["vram_bytes"])) / max(
            float(candidate["vram_bytes"]), float(primary["vram_bytes"])
        )
        if memory_ratio < float(minimum_relative_memory):
            reasons.append(f"VRAM is {memory_ratio:.0%} of the primary GPU")
        if reasons:
            rejected.append(f"cuda:{device_id} ({candidate['name']}): " + "; ".join(reasons))
        else:
            accepted.append(device_id)
    return tuple(accepted), tuple(rejected)


def plan_convolution_execution(
    workload: ConvolutionWorkload,
    allow_multi_gpu: bool = False,
    descriptors: Optional[Iterable[Dict[str, object]]] = None,
    minimum_frames_per_device: int = 8,
) -> ConvolutionExecutionPlan:
    """Choose single-GPU or gather-free frame sharding for a Gabor workload.

    PyTorch ``DataParallel`` gathers the full convolution response back onto a
    primary GPU.  That is counterproductive for a high-orientation filter bank:
    its largest tensor is recreated on the very card that was already under
    pressure.  The ``frame_parallel`` plan instead runs disjoint frame slices
    on compatible GPUs and transfers each completed slice directly to host.

    The choice combines current free VRAM, hardware compatibility, actual
    batch/filter/output dimensions, and a minimum useful slice size so small
    jobs avoid multi-GPU scatter/gather overhead.
    """
    total_frames = max(1, int(workload.frame_count))
    single_bytes = workload.working_set_bytes(total_frames)
    if descriptors is None:
        if not torch.cuda.is_available():
            return ConvolutionExecutionPlan("single", ("cpu",), total_frames, single_bytes, "CUDA unavailable")
        descriptors = tuple(
            _gpu_descriptor(index)
            for index in range(torch.cuda.device_count())
            if gpu_available_vram_bytes(index) > 0
        )
    else:
        descriptors = tuple(dict(item) for item in descriptors)

    if not descriptors:
        return ConvolutionExecutionPlan("single", ("cpu",), total_frames, single_bytes, "no CUDA device has free VRAM")

    compatible_ids, _rejected = select_compatible_multi_gpu_ids(descriptors)
    descriptor_by_id = {int(item["id"]): item for item in descriptors}
    primary_id = int(compatible_ids[0]) if compatible_ids else int(descriptors[0]["id"])
    primary_name = f"cuda:{primary_id}"
    if not allow_multi_gpu or len(compatible_ids) < 2:
        reason = "multi-GPU disabled" if not allow_multi_gpu else "no compatible peer GPU"
        return ConvolutionExecutionPlan("single", (primary_name,), total_frames, single_bytes, reason)

    frames_per_device = math.ceil(total_frames / len(compatible_ids))
    estimated_device_bytes = workload.working_set_bytes(frames_per_device)
    if frames_per_device < max(1, int(minimum_frames_per_device)):
        return ConvolutionExecutionPlan(
            "single", (primary_name,), total_frames, single_bytes,
            f"only {frames_per_device} frames per GPU would not amortize sharding",
        )

    viable_ids = []
    for device_id in compatible_ids:
        descriptor = descriptor_by_id[int(device_id)]
        free_bytes = int(descriptor.get("free_vram_bytes", descriptor["vram_bytes"]))
        # Retain 35% of currently free VRAM for cuDNN growth, the display
        # driver, and asynchronous transfer buffers.
        if estimated_device_bytes <= int(free_bytes * 0.65):
            viable_ids.append(int(device_id))
    if len(viable_ids) < 2:
        return ConvolutionExecutionPlan(
            "single", (primary_name,), total_frames, single_bytes,
            "a compatible peer lacks free VRAM for its planned frame slice",
        )

    output_bytes = workload.output_bytes(total_frames)
    intense = single_bytes >= 128 * 1024**2 or output_bytes >= 64 * 1024**2
    if not intense:
        return ConvolutionExecutionPlan(
            "single", (primary_name,), total_frames, single_bytes,
            "single-GPU execution is faster for this small convolution workload",
        )
    devices = tuple(f"cuda:{device_id}" for device_id in viable_ids)
    frames_per_device = math.ceil(total_frames / len(devices))
    return ConvolutionExecutionPlan(
        "frame_parallel", devices, frames_per_device, workload.working_set_bytes(frames_per_device),
        "compatible GPUs have VRAM for gather-free frame sharding",
    )


def compute_devices(allow_multi_gpu: bool = False):
    """Return safe compute devices, rejecting imbalanced DataParallel groups."""
    if not torch.cuda.is_available():
        return ["cpu"]
    if allow_multi_gpu and enabled_feature("MULTI_GPU", default=False):
        available_ids = [idx for idx in range(torch.cuda.device_count()) if gpu_available_vram_bytes(idx) > 0]
        if len(available_ids) > 1:
            accepted_ids, rejected = select_compatible_multi_gpu_ids(
                _gpu_descriptor(idx) for idx in available_ids
            )
            if len(accepted_ids) > 1:
                return [f"cuda:{idx}" for idx in accepted_ids]
            decision_key = tuple(available_ids)
            if decision_key not in _reported_multi_gpu_decisions:
                _reported_multi_gpu_decisions.add(decision_key)
                detail = " | ".join(rejected) or "no compatible peer GPU was available"
                print(
                    "Multi-GPU requested, but synchronous DataParallel would be bottlenecked; "
                    f"using cuda:{accepted_ids[0]} only. {detail}"
                )
            return [f"cuda:{accepted_ids[0]}"]
    return ["cuda:0"]


def autotuned_frame_chunk_size(num_frames: int, nx: int, ny: int, n_channels: int = 1) -> int:
    """Choose a bounded convolution batch from current free device memory."""
    num_frames = max(1, int(num_frames))
    pixels = max(1, int(nx) * int(ny))
    if torch.cuda.is_available():
        budget = max(1, int(gpu_available_vram_bytes() * 0.20))
        # input, convolution workspace, output, and transfer headroom
        estimate_per_frame = max(1, pixels * max(1, int(n_channels)) * 4 * 8)
        return max(16, min(num_frames, 1024, budget // estimate_per_frame))
    budget = max(1, int(available_ram_bytes() * 0.10))
    return max(8, min(num_frames, 512, budget // max(1, pixels * 4 * 4)))


def get_gpu_count() -> int:
    """Return the total number of CUDA devices available on the system."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


@lru_cache(maxsize=8)
def gpu_vram_bytes(device_id: int = 0) -> int:
    """Return total VRAM of the specified CUDA device, or zero when no GPU is present."""
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        return 0
    return int(torch.cuda.get_device_properties(device_id).total_memory)


def gpu_available_vram_bytes(device_id: int = 0) -> int:
    """Return currently free CUDA memory, falling back safely when unavailable."""
    if not torch.cuda.is_available() or device_id >= torch.cuda.device_count():
        return 0
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_id)
        return int(free_bytes)
    except Exception:
        return gpu_vram_bytes(device_id)


def available_ram_bytes() -> int:
    """Return bytes of RAM currently free for allocation (not total installed)."""
    return int(psutil.virtual_memory().available)


def has_enough_ram(
    required_bytes: int,
    safety_margin: float = 1.20,
    os_reserve_fraction: float = 0.30,
) -> bool:
    """Return whether an allocation fits while retaining OS/UI headroom.

    Treating all currently available RAM as application memory can make Windows
    page aggressively or stop servicing the GUI.  The reserve is intentionally
    retained for the desktop, file cache, driver allocations, and transient
    copies made by NumPy/PyTorch.
    """
    reserve_fraction = min(max(float(os_reserve_fraction), 0.0), 0.80)
    usable = int(available_ram_bytes() * (1.0 - reserve_fraction))
    return usable > int(required_bytes * safety_margin)


def cpu_worker_count(cap: Optional[int] = None) -> int:
    """Return a conservative CPU worker count that leaves one core for the OS."""
    cores = os.cpu_count() or 4
    limit = cap if cap is not None else cores
    return max(1, min(cores - 1, limit))


def two_photon_io_worker_count(n_planes: int, cap: int = 4) -> int:
    """Return a bounded worker count for independent Suite2p plane reads.

    Suite2p planes live in separate memory-mapped files.  A small thread pool
    overlaps those reads and the per-plane neuropil correction without copying
    the large arrays between processes.  More workers tend to contend for the
    same disk and memory bandwidth, particularly on Windows, so this is capped
    deliberately rather than using every CPU core.
    """
    return max(1, min(max(1, int(n_planes)), max(1, int(cap)), cpu_worker_count(cap=cap)))


def cpu_inner_thread_count(workers: int = 1) -> int:
    """Return a BLAS/OpenMP budget that avoids nested-worker oversubscription."""
    workers = max(1, int(workers))
    return max(1, cpu_worker_count() // workers)


def cpu_threadpool_scope(workers: int = 1):
    """Limit native NumPy/SciPy/BLAS threads while an outer worker pool is active.

    NumPy, SciPy, PyTorch, and OpenCV may all have their own thread pools.  A
    four-process joblib run with each process using every logical core is much
    slower than a bounded allocation, especially on laptop CPUs.  The scope is
    a no-op only for older environments without ``threadpoolctl``.
    """
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=cpu_inner_thread_count(workers))


def torch_compile_enabled() -> bool:
    """Whether the experimental compiled convolution runner is enabled."""
    return enabled_feature("TORCH_COMPILE", default=False) and hasattr(torch, "compile")


def amp_enabled() -> bool:
    """Whether opt-in Tensor Core mixed precision is enabled for convolution."""
    return enabled_feature("AMP", default=False) and torch.cuda.is_available()


def convolution_precision_scope(device: str):
    """Return the safe default or explicit CUDA autocast context.

    Mixed precision is deliberately not applied to Coarse RF correlation:
    that stage preserves its established float32/float64 numerical paths.
    """
    if str(device).startswith("cuda") and amp_enabled():
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def configure_torch_cpu_threads(workers: int = 1) -> int:
    """Set PyTorch's CPU thread count to the available nested-work budget."""
    threads = cpu_inner_thread_count(workers)
    try:
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, min(4, threads)))
    except RuntimeError:
        # PyTorch disallows changing inter-op threads after work has started;
        # its existing setting is still safe, and threadpoolctl limits BLAS.
        pass
    return threads


def resolve_compute_device(prefer_gpu: bool = True) -> ComputeDevice:
    """Pick ``cuda`` when a GPU exists and ``prefer_gpu`` is True, else ``cpu``."""
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def wavelet_filter_chunk_size() -> int:
    """Filters processed per GPU matmul batch in wavelet decomposition."""
    vram_gb = gpu_vram_bytes() / (1024**3)
    if vram_gb >= 16:
        return 2000
    if vram_gb >= 8:
        return 1000
    if vram_gb >= 4:
        return 500
    return 250


def video_downsample_chunk_size(default: int = 1000) -> int:
    """Frames read per chunk when downsampling stimulus movies to disk."""
    ram_gb = available_ram_bytes() / (1024**3)
    if ram_gb >= 32:
        return default
    if ram_gb >= 16:
        return 750
    if ram_gb >= 8:
        return 500
    return 300


def gpu_neuron_chunk_size(
    n_timepoints: int,
    n_features: int,
    dtype_bytes: int = 4,
    safety: float = 0.55,
    default: int = 1000,
) -> int:
    """Neurons per Pearson-correlation batch given stimulus size and VRAM."""
    if not torch.cuda.is_available():
        return min(default, 512)

    stim_bytes = n_timepoints * n_features * dtype_bytes
    budget = int(gpu_vram_bytes() * safety) - stim_bytes
    if budget <= 0:
        return 64

    per_neuron = max(n_timepoints * dtype_bytes * 2, 1)
    return max(64, min(default, budget // per_neuron))


def model_parallel_jobs() -> int:
    """Joblib worker count for per-neuron model fitting.

    When CUDA is active each worker may allocate GPU memory; cap workers to
    avoid VRAM exhaustion while still exploiting multi-core hosts.
    """
    if torch.cuda.is_available():
        vram_gb = gpu_vram_bytes() / (1024**3)
        if vram_gb >= 24:
            return min(4, cpu_worker_count(cap=4))
        if vram_gb >= 12:
            return min(3, cpu_worker_count(cap=3))
        return min(2, cpu_worker_count(cap=2))
    # Per-neuron workers each build sizeable feature/correlation arrays.  Using
    # every logical core can turn model fitting into a memory-pressure event and
    # leave Windows unable to service the GUI.  Four workers still provides
    # useful parallelism while reserving CPU and RAM for the operating system.
    return min(4, cpu_worker_count(cap=4))


def coarse_wavelet_chunk_size(default: int = 1000) -> int:
    """Time frames processed per chunk when building coarse wavelet caches."""
    ram_gb = available_ram_bytes() / (1024**3)

    # If system is especially powerful
    if ram_gb >= 32:
        # we can safely process roughly 40 frames per GB of available RAM.
        dynamic_chunk = int(ram_gb * 40)
        
        # Cap at 5000 to prevent CPU cache thrashing during skimage.resize
        return min(dynamic_chunk, 5000)
    
    if ram_gb >= 24:
        return default
    if ram_gb >= 12:
        return 750
    return 500


def coarse_wavelet_chunk_size_gpu_or_cpu(
    nx0: int = 138,
    ny0: int = 112,
    no: int = 18,
    ns: int = 4,
    nf: int = 1,
    default: int = 1000,
    vram_safety_margin: float = 0.60
) -> int:
    """Time frames processed per chunk, dynamically bounded by GPU VRAM and CPU RAM."""
    ram_gb = available_ram_bytes() / (1024**3)
    
    # --- 1. GPU VRAM BOUNDARY (If applicable) ---
    if torch.cuda.is_available():
        # Calculate base elements per frame
        elements_per_frame = nx0 * ny0 * no * ns * nf
        
        # 4 bytes per float32. We have 3 primary tensors (w_r, w_i, w_c)
        bytes_per_frame_base = elements_per_frame * 4 * 3
        
        # PyTorch interpolation requires overhead for intermediate gradients/views.
        # We multiply by ~4 to safely estimate the peak VRAM spike during F.interpolate
        peak_bytes_per_frame = bytes_per_frame_base * 4 
        
        vram_budget = gpu_vram_bytes() * vram_safety_margin
        
        if peak_bytes_per_frame > 0:
            gpu_max_chunk = int(vram_budget // peak_bytes_per_frame)
            
            # Bound the GPU chunk by system RAM limits (cap at 5000 to prevent CPU stalling)
            dynamic_chunk = min(gpu_max_chunk, int(ram_gb * 40))
            
            # Ensure we process at least *some* frames, but no more than 5000
            return max(64, min(dynamic_chunk, 5000))

    # --- 2. CPU RAM BOUNDARY (Fallback if no GPU) ---
    if ram_gb >= 32:
        dynamic_chunk = int(ram_gb * 40)
        return min(dynamic_chunk, 5000)
    
    if ram_gb >= 24:
        return default
    if ram_gb >= 12:
        return 750
        
    return 500
