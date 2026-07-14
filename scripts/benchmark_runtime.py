"""Small, repeatable local benchmark for Waven execution backends.

This is intentionally synthetic: it checks relative backend throughput on the
current computer without reading or writing an experiment cache. Run from the
repository root after installing the project:

    python scripts/benchmark_runtime.py --frames 1024 --neurons 32
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import psutil
import torch

from waven.analysis.rf_correlation import streaming_cross_correlation
from waven.analysis.sta import compute_sta


def measure(label, callback):
    started = time.perf_counter()
    callback()
    elapsed = time.perf_counter() - started
    print(f"{label}: {elapsed:.3f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--height", type=int, default=18)
    parser.add_argument("--neurons", type=int, default=16)
    args = parser.parse_args()
    rng = np.random.default_rng(20260714)
    print(
        f"CPU logical={psutil.cpu_count()} | RAM available={psutil.virtual_memory().available / 2**30:.1f} GiB | "
        f"CUDA={torch.cuda.is_available()}"
    )
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        print(f"GPU={torch.cuda.get_device_name(0)} | VRAM free/total={free / 2**30:.1f}/{total / 2**30:.1f} GiB")

    movie = rng.normal(size=(args.frames, args.height, args.width)).astype(np.float32)
    response = rng.normal(size=(args.frames, args.neurons)).astype(np.float32)
    stimulus = rng.random(size=(args.frames, 4, 3, 4, 2), dtype=np.float32)
    counts = rng.poisson(0.2, size=(2, args.frames, args.neurons)).astype(np.int32)

    old_rf = os.environ.get("WAVEN_RF_GPU")
    for use_gpu in (False, True):
        if use_gpu and not torch.cuda.is_available():
            continue
        os.environ["WAVEN_RF_GPU"] = "1" if use_gpu else "0"
        measure(f"Coarse RF ({'GPU' if use_gpu else 'CPU'})", lambda: streaming_cross_correlation(stimulus, response))
    if old_rf is None:
        os.environ.pop("WAVEN_RF_GPU", None)
    else:
        os.environ["WAVEN_RF_GPU"] = old_rf

    old_fft = os.environ.get("WAVEN_STA_FFT")
    os.environ["WAVEN_STA_FFT"] = "0"
    measure("STA batched GEMM", lambda: compute_sta(movie, counts, 1000.0, max_lag_ms=8, n_shuffles=8))
    if torch.cuda.is_available():
        os.environ["WAVEN_STA_FFT"] = "1"
        measure("STA FFT actual maps + batched null", lambda: compute_sta(movie, counts, 1000.0, max_lag_ms=8, n_shuffles=8))
    if old_fft is None:
        os.environ.pop("WAVEN_STA_FFT", None)
    else:
        os.environ["WAVEN_STA_FFT"] = old_fft


if __name__ == "__main__":
    main()
