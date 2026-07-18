"""Memory-bounded correlation primitives for receptive-field analysis.

This module deliberately contains no plotting or GUI code.  Keeping the
disk-backed correlation implementation here makes its memory contract clear
and lets the legacy :mod:`receptive_fields` API remain a compatibility layer.
"""
from __future__ import annotations

import math
import os
import shutil
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


def _is_zarr_output_path(output_path) -> bool:
    """Return whether an RF result should be materialized as a Zarr store."""
    return bool(output_path) and os.fspath(output_path).lower().endswith(".zarr")


def _allocate_correlation_result(shape, output_path=None, chunks=None):
    """Allocate RF correlations in RAM, NPY, or Zarr without changing precision."""
    required_bytes = int(np.prod(shape, dtype=np.int64) * np.dtype(np.float32).itemsize)
    if output_path is None:
        if not has_enough_ram(required_bytes, safety_margin=1.50):
            raise MemoryError(
                "RF result tensor cannot be held safely in RAM: "
                f"requires {required_bytes / 1024**3:.2f} GiB. Provide an RF output path "
                "or reduce the feature-grid parameters."
            )
        return np.empty(shape, dtype=np.float32)
    output_path = os.fspath(output_path)
    output_folder = os.path.dirname(output_path) or "."
    os.makedirs(output_folder, exist_ok=True)
    free_bytes = shutil.disk_usage(output_folder).free
    # ``w+`` replaces a previous NPY result at this exact path. Its existing
    # allocation is reclaimable, so include it in the preflight rather than
    # falsely rejecting a recomputation on an otherwise full volume.  A Zarr
    # directory is deliberately not counted here: its compressed on-disk size
    # is not a reliable reserve for a fresh uncompressed worst-case write.
    reclaimable_bytes = os.path.getsize(output_path) if os.path.isfile(output_path) else 0
    if free_bytes + reclaimable_bytes < int(required_bytes * 1.10):
        raise OSError(
            "Insufficient disk space for disk-backed RF correlations: "
            f"need {required_bytes * 1.10 / 1024**3:.2f} GiB, "
            f"available {(free_bytes + reclaimable_bytes) / 1024**3:.2f} GiB."
        )
    storage_name = "Zarr" if _is_zarr_output_path(output_path) else "NPY"
    print(
        f"Writing RF correlations to a disk-backed {storage_name} array to preserve RAM: "
        f"{output_path} ({required_bytes / 1024**3:.2f} GiB)."
    )
    if _is_zarr_output_path(output_path):
        try:
            import zarr
            from numcodecs import Blosc
        except ImportError as exc:
            raise ImportError(
                "Zarr RF correlation output was selected, but zarr and numcodecs are required."
            ) from exc
        if os.path.isdir(output_path):
            # The RF result is a generated cache, analogous to replacing an
            # NPY memmap with ``mode='w+'``.  Removing it first prevents stale
            # chunks from an earlier shape from being retained in the store.
            shutil.rmtree(output_path)
        compressor = Blosc(cname="lz4", clevel=3, shuffle=Blosc.BITSHUFFLE)
        zarr_kwargs = {
            "mode": "w",
            "shape": tuple(int(dim) for dim in shape),
            "chunks": tuple(int(dim) for dim in (chunks or shape)),
            "dtype": np.float32,
            "compressor": compressor,
        }
        try:
            return zarr.open(output_path, **zarr_kwargs)
        except TypeError:
            # Zarr v3 takes a compressor list while current v2 releases use a
            # single ``compressor`` argument.
            zarr_kwargs["compressors"] = [zarr_kwargs.pop("compressor")]
            return zarr.open(output_path, **zarr_kwargs)
    return np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)


def _structured_zarr_chunks(n_neurons, nx, ny, feature_shape, x_chunk, y_chunk):
    """Choose output chunks aligned with RF spatial-tile writes.

    A complete orientation/sigma/frequency vector is consumed by the GUI for a
    selected neuron, while the correlation writer produces an x/y tile for all
    neurons.  This layout keeps both operations bounded without repeatedly
    recompressing partial chunks.
    """
    feature_values = max(1, int(np.prod(feature_shape, dtype=np.int64)))
    bytes_per_neuron_tile = max(1, int(x_chunk) * int(y_chunk) * feature_values * 4)
    neuron_chunk = max(1, min(int(n_neurons), (16 * 1024**2) // bytes_per_neuron_tile))
    return (
        neuron_chunk,
        int(x_chunk),
        int(y_chunk),
        *(int(dim) for dim in feature_shape),
    )


def _gpu_cross_feature_batch_size(
    tile_features: int,
    n_neurons: int,
    time_chunk: int,
    free_vram_bytes: int,
) -> int:
    """Return a conservative GPU feature-subtile width for RF correlation.

    The persistent cross-product, temporary GEMM workspace, and current
    float64 input block all coexist.  A 28% free-VRAM budget keeps enough room
    for PyTorch/cuBLAS allocations and other active application work.  The
    resulting subtiles let an oversized spatial Zarr tile retain GPU math
    without rereading and decompressing that tile from disk.
    """
    tile_features = max(1, int(tile_features))
    n_neurons = max(1, int(n_neurons))
    time_chunk = max(1, int(time_chunk))
    free_vram_bytes = max(0, int(free_vram_bytes))
    if free_vram_bytes <= 0:
        return 0
    # Three feature-by-neuron buffers cover GEMM output/workspace/staging;
    # the input block is an additional time-by-feature float64 tensor.
    bytes_per_feature = np.dtype(np.float64).itemsize * (n_neurons * 3 + time_chunk)
    budget = int(free_vram_bytes * 0.28)
    return max(1, min(tile_features, budget // max(1, bytes_per_feature)))


def _add_gpu_cross_subtiles(
    cross: np.ndarray,
    centered_block: np.ndarray,
    centered_response: np.ndarray,
    *,
    response_gpu,
    response_block_mean: np.ndarray,
    time_start: int,
    time_end: int,
    feature_batch: int,
) -> tuple[bool, Optional[RuntimeError]]:
    """Accumulate one RF time block through VRAM-safe GPU feature subtiles.

    Completed subtiles are retained in ``cross``.  If a transient CUDA failure
    occurs, only the unfinished suffix uses the numerically identical CPU
    product; later time blocks retry with a smaller GPU subtile.
    """
    response_tensor = None
    block_tensor = None
    completed = 0
    try:
        response_tensor = (
            response_gpu[time_start:time_end]
            - torch.as_tensor(response_block_mean, dtype=torch.float64, device="cuda:0")
            if response_gpu is not None
            else torch.as_tensor(centered_response, dtype=torch.float64, device="cuda:0")
        )
        for start in range(0, centered_block.shape[1], feature_batch):
            end = min(centered_block.shape[1], start + feature_batch)
            block_tensor = torch.as_tensor(
                centered_block[:, start:end], dtype=torch.float64, device="cuda:0"
            )
            cross[start:end] += (block_tensor.T @ response_tensor).cpu().numpy()
            del block_tensor
            block_tensor = None
            completed = end
        return True, None
    except RuntimeError as exc:
        if completed < centered_block.shape[1]:
            cross[completed:] += centered_block[:, completed:].T @ centered_response
        return False, exc
    finally:
        if block_tensor is not None:
            del block_tensor
        if response_tensor is not None:
            del response_tensor


def _chunk_aligned_structured_correlation(
    stimulus,
    response,
    n_time,
    n_features,
    output_path=None,
    structured_output=False,
    output_feature_shape=None,
):
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
    if output_feature_shape is None:
        output_feature_shape = shape[1:]
    else:
        output_feature_shape = tuple(int(dim) for dim in output_feature_shape)
        if (
            len(output_feature_shape) < 2
            or tuple(output_feature_shape[:2]) != (nx, ny)
            or int(np.prod(output_feature_shape, dtype=np.int64)) != n_features
        ):
            raise ValueError(
                "Structured RF output shape must preserve the input x/y grid and "
                f"feature count; got {output_feature_shape} for {(nx, ny)} and "
                f"{n_features} features."
            )
    write_structured_zarr = bool(structured_output and _is_zarr_output_path(output_path))
    result_shape = (
        (n_neurons, *output_feature_shape)
        if write_structured_zarr
        else (n_neurons, n_features)
    )
    result_chunks = (
        _structured_zarr_chunks(
            n_neurons,
            output_feature_shape[0],
            output_feature_shape[1],
            output_feature_shape[2:],
            x_chunk,
            y_chunk,
        )
        if write_structured_zarr
        else None
    )
    result = _allocate_correlation_result(
        result_shape,
        output_path=output_path,
        chunks=result_chunks,
    )
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
                gpu_feature_batch = 0
                if use_gpu:
                    free_vram_bytes = gpu_available_vram_bytes()
                    required = tile_features * n_neurons * np.dtype(np.float64).itemsize * 3
                    if required < free_vram_bytes * 0.35:
                        try:
                            cross_tensor = torch.zeros(
                                (tile_features, n_neurons), dtype=torch.float64, device="cuda:0"
                            )
                        except RuntimeError:
                            cross_tensor = None
                    if cross_tensor is None:
                        gpu_feature_batch = _gpu_cross_feature_batch_size(
                            tile_features,
                            n_neurons,
                            time_chunk,
                            free_vram_bytes,
                        )
                        # A failed full allocation is often fragmentation. Do
                        # not retry the same allocation path as a single batch.
                        if gpu_feature_batch >= tile_features:
                            gpu_feature_batch = max(1, tile_features // 2)
                        if gpu_feature_batch:
                            print(
                                "RF GPU tile downshift: retaining the Zarr read tile but "
                                f"processing {tile_features:,} features as "
                                f"{math.ceil(tile_features / gpu_feature_batch):,} GPU subtiles "
                                f"of up to {gpu_feature_batch:,} features."
                            )

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
                    gpu_cross_completed = False
                    if cross_tensor is not None:
                        block_tensor = None
                        response_tensor = None
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
                            block_tensor = response_tensor = None
                            gpu_cross_completed = True
                        except RuntimeError as exc:
                            print(f"RF GPU tile downshift after full-tile allocation failed: {exc}")
                            # Preserve completed full-tile GPU chunks, then
                            # retry this and later time blocks as subtiles.
                            cross += cross_tensor.cpu().numpy()
                            del cross_tensor
                            if block_tensor is not None:
                                del block_tensor
                            if response_tensor is not None:
                                del response_tensor
                            torch.cuda.empty_cache()
                            cross_tensor = None
                            gpu_feature_batch = _gpu_cross_feature_batch_size(
                                tile_features,
                                n_neurons,
                                batch_size,
                                gpu_available_vram_bytes(),
                            )
                            if gpu_feature_batch >= tile_features:
                                gpu_feature_batch = max(1, tile_features // 2)
                    if not gpu_cross_completed and gpu_feature_batch:
                        compute_start = time.perf_counter()
                        gpu_success, gpu_error = _add_gpu_cross_subtiles(
                            cross,
                            centered_block,
                            centered_response,
                            response_gpu=response_gpu,
                            response_block_mean=response_block_mean,
                            time_start=time_start,
                            time_end=time_end,
                            feature_batch=gpu_feature_batch,
                        )
                        telemetry.add("gpu_cross_product", time.perf_counter() - compute_start, block.nbytes)
                        if not gpu_success:
                            print(f"RF GPU subtile fell back to CPU for an unfinished suffix: {gpu_error}")
                            torch.cuda.empty_cache()
                            gpu_feature_batch //= 2
                    elif not gpu_cross_completed:
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
                if write_structured_zarr:
                    result[:, x_start:x_end, y_start:y_end, ...] = tile_corr.T.reshape(
                        n_neurons,
                        x_end - x_start,
                        y_end - y_start,
                        *output_feature_shape[2:],
                    )
                else:
                    # x/y tiles are contiguous only when x has one value. Assign
                    # one x row at a time to preserve the flattened C-order NPY
                    # layout used by the established public streaming API.
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
        if hasattr(result, "flush"):
            result.flush()
    telemetry.report()
    return result


def streaming_cross_correlation(
    stimulus: Any,
    response: np.ndarray,
    chunk_size: Optional[int] = None,
    n_time: Optional[int] = None,
    output_path: Optional[str] = None,
    structured_output: bool = False,
    output_feature_shape: Optional[Tuple[int, ...]] = None,
) -> Any:
    """Return Pearson correlations while reading a disk-backed stimulus in blocks.

    ``stimulus`` may be a dense NumPy matrix, an NPY memmap, or a 5/6-D Zarr
    wavelet tensor. Structured tensors are read in spatial tiles so no call
    reshapes the complete tensor into RAM.  ``structured_output`` is reserved
    for callers that need a Zarr result in the stimulus feature layout rather
    than the established flattened ``(neurons, features)`` public shape.
    ``output_feature_shape`` optionally adds a logically singleton feature axis
    (for example, the coarse-model frequency axis) without changing values.
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

    if len(stimulus_shape) >= 4:
        # A structured Zarr tensor benefits from chunk-aligned temporal
        # accumulation; it is both faster and lower-I/O than reading all time
        # frames separately for each small spatial feature block.
        return _chunk_aligned_structured_correlation(
            stimulus,
            response,
            n_time,
            n_features,
            output_path=output_path,
            structured_output=structured_output,
            output_feature_shape=output_feature_shape,
        )

    response -= response.mean(axis=0, keepdims=True)
    response /= np.maximum(response.std(axis=0, keepdims=True, ddof=1), 1e-9)

    correlations = _allocate_correlation_result(
        (n_neurons, n_features), output_path=output_path
    )

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
    if hasattr(correlations, "flush"):
        correlations.flush()
    return correlations


__all__ = ["streaming_cross_correlation"]
