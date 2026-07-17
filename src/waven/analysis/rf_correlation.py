"""Memory-bounded correlation primitives for receptive-field analysis.

This module deliberately contains no plotting or GUI code.  Keeping the
disk-backed correlation implementation here makes its memory contract clear
and lets the legacy :mod:`receptive_fields` API remain a compatibility layer.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch

from ..runtime.performance import (
    OperationTelemetry,
    configure_zarr_codec_threads,
    enabled_feature,
    gpu_available_vram_bytes,
    has_enough_ram,
)


def _chunk_aligned_structured_correlation(stimulus, response, n_time, n_features):
    """Correlate a 5/6-D tensor with stable, chunk-aligned covariance updates.

    The calculation uses the parallel/Welford covariance update.  In contrast
    to reconstructing covariance from ``sum(x*y) - sum(x)*sum(y)/n``, this
    does not lose the small neural signal when non-negative wavelet power has a
    large baseline.  That cancellation previously produced impossible Pearson
    values (for example, values far outside ``[-1, 1]``) and corrupted the RF
    peak used by every downstream tuning plot.

    Each spatial tile is read once per time chunk.  Tile working memory remains
    bounded by the Zarr chunk and one feature-by-neuron covariance matrix.
    """
    shape = tuple(int(dim) for dim in stimulus.shape)
    nx, ny = shape[1], shape[2]
    features_per_pixel = int(np.prod(shape[3:], dtype=np.int64))
    chunks = getattr(stimulus, "chunks", None) or (128, min(nx, 16), min(ny, 16))
    time_chunk = max(1, min(n_time, int(chunks[0])))
    x_chunk = max(1, min(nx, int(chunks[1])))
    y_chunk = max(1, min(ny, int(chunks[2])))
    n_neurons = int(response.shape[1])
    # Float64 running statistics are deliberate.  Wavelet power can have a
    # sizeable positive baseline while the correlation signal is small.
    response = np.asarray(response, dtype=np.float64)
    result = np.empty((n_neurons, n_features), dtype=np.float32)
    # Response summaries are identical for every spatial tile.  Computing them
    # once avoids repeatedly centering the same 18,000 x N-neuron matrix while
    # retaining the numerically stable parallel/Welford covariance update.
    response_schedule = []
    response_mean = np.zeros(n_neurons, dtype=np.float64)
    response_m2 = np.zeros(n_neurons, dtype=np.float64)
    response_seen = 0
    for time_start in range(0, n_time, time_chunk):
        time_end = min(n_time, time_start + time_chunk)
        response_block = response[time_start:time_end]
        batch_size = time_end - time_start
        response_block_mean = response_block.mean(axis=0, dtype=np.float64)
        centered_response = response_block - response_block_mean
        total = response_seen + batch_size
        response_mean_before = response_mean.copy()
        correction_scale = response_seen * batch_size / total if response_seen else 0.0
        if response_seen:
            response_delta = response_block_mean - response_mean
            response_m2 += response_delta * response_delta * correction_scale
        response_m2 += np.einsum(
            "ij,ij->j", centered_response, centered_response, dtype=np.float64
        )
        response_mean += (response_block_mean - response_mean) * (batch_size / total)
        response_schedule.append(
            (
                time_start,
                time_end,
                batch_size,
                response_block_mean,
                centered_response,
                response_mean_before,
                correction_scale,
            )
        )
        response_seen = total
    total_tiles = math.ceil(nx / x_chunk) * math.ceil(ny / y_chunk)
    tile_number = 0
    telemetry = OperationTelemetry("Coarse RF correlation")
    # This also applies when an existing coarse-power cache is reused in a
    # later session, where the cache-writing stage did not configure Blosc.
    codec_threads = configure_zarr_codec_threads()
    use_gpu = enabled_feature("RF_GPU", default=True) and torch.cuda.is_available()
    response_gpu = None
    if use_gpu:
        # Response is reused for every spatial tile.  Keeping it resident
        # removes a host-to-device transfer from every tile/time chunk when
        # the current VRAM budget permits it.
        response_bytes = response.nbytes
        if response_bytes <= gpu_available_vram_bytes() * 0.15:
            try:
                response_gpu = torch.as_tensor(response, dtype=torch.float64, device="cuda:0")
            except RuntimeError:
                response_gpu = None
    print(
        "RF correlation using stable chunk-aligned covariance: "
        f"time={time_chunk}, x={x_chunk}, y={y_chunk}; {total_tiles} spatial tiles."
        + (f" Blosc threads={codec_threads}." if codec_threads is not None else "")
    )
    # One reader overlaps decompression/storage latency for the next time chunk
    # with the covariance and GPU work for the current one.  It touches a
    # different immutable Zarr chunk, and is intentionally limited to one
    # thread so it does not turn a storage-bound analysis into disk contention.
    prefetch_executor = (
        ThreadPoolExecutor(max_workers=1)
        if enabled_feature("RF_PREFETCH", default=True)
        else None
    )
    try:
        for x_start in range(0, nx, x_chunk):
            x_end = min(nx, x_start + x_chunk)
            for y_start in range(0, ny, y_chunk):
                y_end = min(ny, y_start + y_chunk)
                tile_features = (x_end - x_start) * (y_end - y_start) * features_per_pixel
                feature_mean = np.zeros(tile_features, dtype=np.float64)
                feature_m2 = np.zeros(tile_features, dtype=np.float64)
                cross = np.zeros((tile_features, n_neurons), dtype=np.float64)
                cross_tensor = None
                if use_gpu:
                    required = tile_features * n_neurons * np.dtype(np.float64).itemsize * 3
                    if required < gpu_available_vram_bytes() * 0.35:
                        try:
                            cross_tensor = torch.zeros(
                                (tile_features, n_neurons), dtype=torch.float64, device="cuda:0"
                            )
                        except RuntimeError:
                            cross_tensor = None

                def read_block(schedule_item):
                    time_start, time_end, batch_size, *_unused = schedule_item
                    read_start = time.perf_counter()
                    block = np.asarray(
                        stimulus[time_start:time_end, x_start:x_end, y_start:y_end, ...],
                        dtype=np.float64,
                    ).reshape(batch_size, tile_features)
                    return block, time.perf_counter() - read_start

                future = (
                    prefetch_executor.submit(read_block, response_schedule[0])
                    if prefetch_executor is not None
                    else None
                )
                seen = 0
                for schedule_index, schedule_item in enumerate(response_schedule):
                    (
                        time_start,
                        time_end,
                        batch_size,
                        response_block_mean,
                        centered_response,
                        response_mean_before,
                        correction_scale,
                    ) = schedule_item
                    if future is None:
                        block, read_seconds = read_block(schedule_item)
                    else:
                        wait_start = time.perf_counter()
                        block, read_seconds = future.result()
                        telemetry.add("input_wait", time.perf_counter() - wait_start)
                    next_index = schedule_index + 1
                    future = (
                        prefetch_executor.submit(read_block, response_schedule[next_index])
                        if prefetch_executor is not None and next_index < len(response_schedule)
                        else None
                    )
                    telemetry.add("input", read_seconds, block.nbytes)
                    block_mean = block.mean(axis=0, dtype=np.float64)
                    centered_block = block - block_mean
                    total = seen + batch_size
                    if seen:
                        feature_delta = block_mean - feature_mean
                        response_delta = response_block_mean - response_mean_before
                        cross += feature_delta[:, None] * response_delta[None, :] * correction_scale
                        feature_m2 += feature_delta * feature_delta * correction_scale
                    if cross_tensor is not None:
                        try:
                            compute_start = time.perf_counter()
                            block_tensor = torch.as_tensor(centered_block, dtype=torch.float64, device="cuda:0")
                            response_tensor = (
                                response_gpu[time_start:time_end]
                                - torch.as_tensor(response_block_mean, dtype=torch.float64, device="cuda:0")
                                if response_gpu is not None
                                else torch.as_tensor(centered_response, dtype=torch.float64, device="cuda:0")
                            )
                            cross_tensor += block_tensor.T @ response_tensor
                            telemetry.add("gpu_cross_product", time.perf_counter() - compute_start, block.nbytes)
                            del block_tensor, response_tensor
                        except RuntimeError as exc:
                            print(f"RF GPU tile fell back to CPU: {exc}")
                            # Preserve completed GPU chunks before finishing this
                            # and remaining chunks with the numerically identical
                            # CPU calculation.
                            cross += cross_tensor.cpu().numpy()
                            del cross_tensor
                            torch.cuda.empty_cache()
                            cross += centered_block.T @ centered_response
                            cross_tensor = None
                    else:
                        compute_start = time.perf_counter()
                        cross += centered_block.T @ centered_response
                        telemetry.add("cpu_cross_product", time.perf_counter() - compute_start, block.nbytes)
                    feature_m2 += np.einsum("ij,ij->j", centered_block, centered_block, dtype=np.float64)
                    feature_mean += (block_mean - feature_mean) * (batch_size / total)
                    seen = total
                if cross_tensor is not None:
                    cross += cross_tensor.cpu().numpy()
                    del cross_tensor
                denominator = np.sqrt(feature_m2[:, None] * response_m2[None, :])
                tile_corr = np.divide(
                    cross,
                    denominator,
                    out=np.zeros_like(cross, dtype=np.float64),
                    where=denominator > 1e-12,
                )
                # A valid Pearson coefficient cannot exceed one.  The clip only
                # removes last-bit roundoff and prevents a malformed cache from
                # silently becoming a false preferred feature.
                np.clip(tile_corr, -1.0, 1.0, out=tile_corr)
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
    finally:
        if prefetch_executor is not None:
            prefetch_executor.shutdown(wait=True)
        if response_gpu is not None:
            del response_gpu
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
    response = np.asarray(response, dtype=np.float64)
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

    response -= response.mean(axis=0, keepdims=True)
    response /= np.maximum(response.std(axis=0, keepdims=True, ddof=1), 1e-9)

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
