"""Hardware-aware defaults for chunked CPU and GPU workloads.

Chunk sizes and worker counts are derived from currently available system
memory and GPU VRAM so that waven scales across laptops, workstations, and
servers without manual tuning.  All helpers are side-effect free and safe to
call repeatedly (results are cached where profiling would be expensive).
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from functools import lru_cache
from typing import Dict, Literal, Optional

import psutil
import torch

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional protection for legacy installs
    threadpool_limits = None

ComputeDevice = Literal["cuda", "cpu"]


class OperationTelemetry:
    """Low-overhead timing and throughput counters for one pipeline operation."""

    def __init__(self, label: str):
        self.label = str(label)
        self.started = time.perf_counter()
        self.seconds: Dict[str, float] = {}
        self.bytes: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._last_live_report = self.started

    def add(self, stage: str, seconds: float = 0.0, byte_count: int = 0) -> None:
        with self._lock:
            self.seconds[stage] = self.seconds.get(stage, 0.0) + max(0.0, float(seconds))
            self.bytes[stage] = self.bytes.get(stage, 0) + max(0, int(byte_count))

    def _message(self, suffix: str = "") -> str:
        total = max(time.perf_counter() - self.started, 1e-9)
        with self._lock:
            stages = sorted(self.seconds.items())
            byte_counts = dict(self.bytes)
        pieces = [f"{self.label}{suffix}: {total:.1f}s total"]
        for stage, duration in stages:
            rate = byte_counts.get(stage, 0) / max(duration, 1e-9) / 1024**2
            rate_suffix = f", {rate:.1f} MiB/s" if byte_counts.get(stage, 0) else ""
            pieces.append(f"{stage} {duration:.1f}s{rate_suffix}")
        return " | ".join(pieces)

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


def enabled_feature(name: str, default: bool = True) -> bool:
    """Read a conservative runtime feature flag from ``WAVEN_<NAME>``."""
    value = os.environ.get(f"WAVEN_{str(name).upper()}")
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compute_devices(allow_multi_gpu: bool = False):
    """Return safe compute-device names; multi-GPU remains explicit opt-in."""
    if not torch.cuda.is_available():
        return ["cpu"]
    if allow_multi_gpu and enabled_feature("MULTI_GPU", default=False):
        devices = [f"cuda:{idx}" for idx in range(torch.cuda.device_count()) if gpu_available_vram_bytes(idx) > 0]
        if len(devices) > 1:
            return devices
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


def sta_shuffle_batch_size(pixels: int, n_neurons: int, n_shuffles: int, device: str) -> int:
    """Bound concurrent STA shuffle columns by live RAM/VRAM.

    One batch contains the response index matrix, the matrix-product result,
    and a float64 accumulator.  Batching makes each movie pass serve several
    circular-shuffle draws without retaining the complete shuffle cube.
    """
    per_shuffle = max(1, int(pixels) * int(n_neurons) * 4 * 4)
    if str(device).startswith("cuda"):
        budget = int(gpu_available_vram_bytes() * 0.12)
    else:
        budget = int(available_ram_bytes() * 0.05)
    return max(1, min(int(n_shuffles), 16, budget // per_shuffle))


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

    Mixed precision is deliberately not applied to correlation/STA statistics:
    those stages preserve their established float32/float64 numerical paths.
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
