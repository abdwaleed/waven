"""Memory-bounded correlation primitives for receptive-field analysis.

This module deliberately contains no plotting or GUI code.  Keeping the
disk-backed correlation implementation here makes its memory contract clear
and lets the legacy :mod:`receptive_fields` API remain a compatibility layer.
"""
from __future__ import annotations

import math
import time
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch

from ..runtime.performance import (
    OperationTelemetry,
    enabled_feature,
    gpu_available_vram_bytes,
    has_enough_ram,
)


def _chunk_aligned_structured_correlation(stimulus, response, n_time, n_features):
    """Correlate a 5/6-D Zarr tensor by reading each storage chunk once.

    Pearson correlation can be reconstructed from sums, sums of squares, and
    cross-products.  Accumulating those sufficient statistics over time avoids
    repeatedly decompressing the same Zarr chunks for each small feature tile.
    """
    shape = tuple(int(dim) for dim in stimulus.shape)
    nx, ny = shape[1], shape[2]
    features_per_pixel = int(np.prod(shape[3:], dtype=np.int64))
    chunks = getattr(stimulus, "chunks", None) or (128, min(nx, 16), min(ny, 16))
    time_chunk = max(1, min(n_time, int(chunks[0])))
    x_chunk = max(1, min(nx, int(chunks[1])))
    y_chunk = max(1, min(ny, int(chunks[2])))
    n_neurons = int(response.shape[1])
    result = np.empty((n_neurons, n_features), dtype=np.float32)
    response = np.asarray(response, dtype=np.float32)
    response_sum = response.sum(axis=0, dtype=np.float64)
    response_ss = np.einsum("ij,ij->j", response, response, dtype=np.float64)
    response_var = np.maximum(response_ss - (response_sum * response_sum / n_time), 0.0)
    total_tiles = math.ceil(nx / x_chunk) * math.ceil(ny / y_chunk)
    tile_number = 0
    telemetry = OperationTelemetry("Coarse RF correlation")
    use_gpu = enabled_feature("RF_GPU", default=True) and torch.cuda.is_available()
    response_tensor = None
    if use_gpu:
        try:
            response_tensor = torch.as_tensor(response, dtype=torch.float32, device="cuda:0")
        except RuntimeError as exc:
            print(f"RF GPU setup failed; using CPU accumulation: {exc}")
            use_gpu = False
    print(
        "RF correlation using chunk-aligned sufficient statistics: "
        f"time={time_chunk}, x={x_chunk}, y={y_chunk}; {total_tiles} spatial tiles."
    )
    for x_start in range(0, nx, x_chunk):
        x_end = min(nx, x_start + x_chunk)
        for y_start in range(0, ny, y_chunk):
            y_end = min(ny, y_start + y_chunk)
            feature_start = (x_start * ny + y_start) * features_per_pixel
            feature_end = ((x_end - 1) * ny + y_end) * features_per_pixel
            tile_features = (x_end - x_start) * (y_end - y_start) * features_per_pixel
            feature_sum = np.zeros(tile_features, dtype=np.float64)
            feature_ss = np.zeros(tile_features, dtype=np.float64)
            cross = np.zeros((tile_features, n_neurons), dtype=np.float32)
            cross_tensor = None
            if use_gpu:
                required = tile_features * n_neurons * np.dtype(np.float32).itemsize * 3
                if required < gpu_available_vram_bytes() * 0.35:
                    try:
                        cross_tensor = torch.zeros((tile_features, n_neurons), dtype=torch.float32, device="cuda:0")
                    except RuntimeError:
                        cross_tensor = None
            for time_start in range(0, n_time, time_chunk):
                time_end = min(n_time, time_start + time_chunk)
                read_start = time.perf_counter()
                block = np.asarray(
                    stimulus[time_start:time_end, x_start:x_end, y_start:y_end, ...],
                    dtype=np.float32,
                ).reshape(time_end - time_start, tile_features)
                telemetry.add("input", time.perf_counter() - read_start, block.nbytes)
                feature_sum += block.sum(axis=0, dtype=np.float64)
                feature_ss += np.einsum("ij,ij->j", block, block, dtype=np.float64)
                if cross_tensor is not None:
                    try:
                        compute_start = time.perf_counter()
                        block_tensor = torch.as_tensor(block, dtype=torch.float32, device="cuda:0")
                        cross_tensor += block_tensor.T @ response_tensor[time_start:time_end]
                        telemetry.add("gpu_cross_product", time.perf_counter() - compute_start, block.nbytes)
                        del block_tensor
                    except RuntimeError as exc:
                        print(f"RF GPU tile fell back to CPU: {exc}")
                        # Preserve cross-products already accumulated on the GPU
                        # before processing the failed chunk on the CPU.
                        cross += cross_tensor.cpu().numpy()
                        del cross_tensor
                        torch.cuda.empty_cache()
                        cross += block.T @ response[time_start:time_end]
                        cross_tensor = None
                else:
                    compute_start = time.perf_counter()
                    cross += block.T @ response[time_start:time_end]
                    telemetry.add("cpu_cross_product", time.perf_counter() - compute_start, block.nbytes)
            if cross_tensor is not None:
                cross = cross_tensor.cpu().numpy()
                del cross_tensor
            feature_var = np.maximum(feature_ss - (feature_sum * feature_sum / n_time), 0.0)
            denominator = np.sqrt(feature_var[:, None] * response_var[None, :])
            tile_corr = np.divide(
                cross - (feature_sum[:, None] * response_sum[None, :] / n_time),
                denominator,
                out=np.zeros_like(cross),
                where=denominator > 1e-12,
            )
            # x/y tiles are contiguous only when x has one value.  Assign one
            # x row at a time to preserve C-order flattening exactly.
            row_features = (y_end - y_start) * features_per_pixel
            for local_x, global_x in enumerate(range(x_start, x_end)):
                row_start = (global_x * ny + y_start) * features_per_pixel
                row_end = row_start + row_features
                local_start = local_x * row_features
                result[:, row_start:row_end] = tile_corr[local_start:local_start + row_features].T
            tile_number += 1
            if tile_number % 8 == 0 or tile_number == total_tiles:
                print(f"RF correlation spatial tiles: {tile_number}/{total_tiles}")
            telemetry.maybe_report()
    if response_tensor is not None:
        del response_tensor
        torch.cuda.empty_cache()
    telemetry.report()
    return result


def streaming_cross_correlation(
    stimulus: Any,
    response: np.ndarray,
    chunk_size: Optional[int] = None,
    n_time: Optional[int] = None,
) -> np.ndarray:
    """Return Pearson correlations while reading a disk-backed stimulus in blocks.

    ``stimulus`` may be a dense NumPy matrix, an NPY memmap, or a 5/6-D Zarr
    wavelet tensor. Structured tensors are read in spatial tiles so no call
    reshapes the complete tensor into RAM.
    """
    stimulus_shape = tuple(int(dim) for dim in stimulus.shape)
    if len(stimulus_shape) < 2:
        raise ValueError(
            "RF correlation expects a time axis plus at least one feature axis; "
            f"got shape {stimulus_shape}."
        )
    available_time = stimulus_shape[0]
    n_time = available_time if n_time is None else int(n_time)
    if n_time <= 0 or n_time > available_time:
        raise ValueError(
            f"Requested {n_time} RF frames, but stimulus provides {available_time}."
        )

    n_features = int(np.prod(stimulus_shape[1:], dtype=np.int64))
    response = np.asarray(response, dtype=np.float32)
    if response.ndim != 2 or response.shape[0] != n_time:
        raise ValueError(
            "Stimulus/response time axes must match for RF correlation: "
            f"stimulus {stimulus.shape}, response {response.shape}."
        )
    if n_time < 2:
        raise ValueError("RF correlation requires at least two shared stimulus frames.")

    n_neurons = int(response.shape[1])
    if chunk_size is None:
        # Raw + normalized + GEMM work buffers, capped at 128 MiB.
        bytes_per_feature = max(1, n_time * np.dtype(np.float32).itemsize * 4)
        chunk_size = max(1, min(1024, (128 * 1024**2) // bytes_per_feature))
    chunk_size = max(1, min(int(chunk_size), n_features))
    print(
        f"RF correlation streaming {n_features:,} features in {chunk_size:,}-feature blocks "
        f"({n_time:,} frames × {n_neurons:,} neurons)."
    )

    response -= response.mean(axis=0, keepdims=True)
    response /= np.maximum(response.std(axis=0, keepdims=True, ddof=1), 1e-9)
    result_bytes = n_neurons * n_features * np.dtype(np.float32).itemsize
    if not has_enough_ram(result_bytes, safety_margin=1.50):
        raise MemoryError(
            "RF result tensor cannot be held safely in RAM: "
            f"requires {result_bytes / 1024**3:.2f} GiB. Reduce downsampling percentage, "
            "orientation/sigma bins, or number of units before running RF analysis."
        )
    if len(stimulus_shape) >= 4:
        # A structured Zarr tensor benefits from chunk-aligned temporal
        # accumulation; it is both faster and lower-I/O than reading all time
        # frames separately for each small spatial feature block.
        return _chunk_aligned_structured_correlation(stimulus, response, n_time, n_features)

    correlations = np.empty((n_neurons, n_features), dtype=np.float32)

    def feature_blocks() -> Iterator[Tuple[int, int, Any]]:
        if len(stimulus_shape) == 2:
            for start in range(0, n_features, chunk_size):
                end = min(start + chunk_size, n_features)
                yield start, end, stimulus[:n_time, start:end]
            return

        if len(stimulus_shape) < 4:
            raise ValueError(
                "Structured RF stimulus must have shape (time, x, y, features...); "
                f"got {stimulus_shape}."
            )
        nx, ny = stimulus_shape[1], stimulus_shape[2]
        features_per_pixel = int(np.prod(stimulus_shape[3:], dtype=np.int64))
        pixels_per_block = max(1, chunk_size // max(1, features_per_pixel))
        for x in range(nx):
            for y_start in range(0, ny, pixels_per_block):
                y_end = min(y_start + pixels_per_block, ny)
                start = (x * ny + y_start) * features_per_pixel
                end = (x * ny + y_end) * features_per_pixel
                yield start, end, stimulus[:n_time, x:x + 1, y_start:y_end, ...]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    response_tensor = None
    if device == "cuda":
        try:
            response_tensor = torch.as_tensor(response.T, dtype=torch.float32, device=device)
        except RuntimeError as exc:
            print(f"CUDA RF correlation setup failed; using CPU: {exc}")
            torch.cuda.empty_cache()
            device = "cpu"

    for block_index, (start, end, source_block) in enumerate(feature_blocks()):
        features = np.array(source_block, dtype=np.float32, copy=True).reshape(n_time, -1)
        features -= features.mean(axis=0, keepdims=True)
        features /= np.maximum(features.std(axis=0, keepdims=True, ddof=1), 1e-9)
        if device == "cuda":
            try:
                feature_tensor = torch.as_tensor(features.T, dtype=torch.float32, device=device)
                correlations[:, start:end] = (
                    response_tensor @ feature_tensor.T / (n_time - 1)
                ).cpu().numpy()
                del feature_tensor
            except RuntimeError as exc:
                print(f"CUDA RF block {start}:{end} failed; switching remaining blocks to CPU: {exc}")
                del response_tensor
                response_tensor = None
                torch.cuda.empty_cache()
                device = "cpu"
                correlations[:, start:end] = response.T @ features / (n_time - 1)
        else:
            correlations[:, start:end] = response.T @ features / (n_time - 1)
        if block_index % 20 == 0 or end == n_features:
            print(f"RF correlation features: {end:,}/{n_features:,}")

    if response_tensor is not None:
        del response_tensor
        torch.cuda.empty_cache()
    return correlations


__all__ = ["streaming_cross_correlation"]
