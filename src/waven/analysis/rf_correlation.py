"""Memory-bounded correlation primitives for receptive-field analysis.

This module deliberately contains no plotting or GUI code.  Keeping the
disk-backed correlation implementation here makes its memory contract clear
and lets the legacy :mod:`receptive_fields` API remain a compatibility layer.
"""
from __future__ import annotations

from typing import Any, Iterator, Optional, Tuple

import numpy as np
import torch

from ..runtime.performance import has_enough_ram


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
