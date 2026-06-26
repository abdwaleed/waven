"""Hardware-aware defaults for chunked CPU and GPU workloads.

Chunk sizes and worker counts are derived from currently available system
memory and GPU VRAM so that waven scales across laptops, workstations, and
servers without manual tuning.  All helpers are side-effect free and safe to
call repeatedly (results are cached where profiling would be expensive).
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal, Optional

import psutil
import torch

ComputeDevice = Literal["cuda", "cpu"]


@lru_cache(maxsize=1)
def gpu_vram_bytes() -> int:
    """Return total VRAM of CUDA device 0, or zero when no GPU is present."""
    if not torch.cuda.is_available():
        return 0
    return int(torch.cuda.get_device_properties(0).total_memory)


def available_ram_bytes() -> int:
    """Return bytes of RAM currently free for allocation (not total installed)."""
    return int(psutil.virtual_memory().available)


def has_enough_ram(required_bytes: int, safety_margin: float = 1.20) -> bool:
    """Return True when ``required_bytes * safety_margin`` fits in free RAM."""
    return available_ram_bytes() > int(required_bytes * safety_margin)


def cpu_worker_count(cap: Optional[int] = None) -> int:
    """Return a conservative CPU worker count that leaves one core for the OS."""
    cores = os.cpu_count() or 4
    limit = cap if cap is not None else cores
    return max(1, min(cores - 1, limit))


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
    return cpu_worker_count()


def coarse_wavelet_chunk_size(default: int = 1000) -> int:
    """Time frames processed per chunk when building coarse wavelet caches."""
    ram_gb = available_ram_bytes() / (1024**3)
    if ram_gb >= 24:
        return default
    if ram_gb >= 12:
        return 750
    return 500
