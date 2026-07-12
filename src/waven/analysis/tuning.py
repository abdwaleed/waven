"""Validated extraction of one neuron's receptive-field tuning curves.

The receptive-field correlation tensor has a fixed feature order:
``(neuron, x, y, orientation, size, frequency)``.  Keeping the indexing in one
small module prevents GUI plots, exports, and legacy helper functions from
quietly selecting different axes for the same neuron.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def extract_rf_tuning_curves(
    rf_tensor: np.ndarray,
    best_indices: np.ndarray,
    neuron_index: int,
) -> Dict[str, np.ndarray]:
    """Extract direct correlation tunings for one neuron.

    Args:
        rf_tensor: Receptive-field correlations with shape ``(n_neurons, nx,
            ny, n_orientations, n_sizes, n_frequencies)``.  The first axis
            indexes neural units; all remaining axes are wavelet features.
        best_indices: Preferred feature indices with shape ``(at least 4,
            n_neurons)`` in ``x, y, orientation, size[, frequency]`` order.
            A missing frequency row is interpreted as index zero.
        neuron_index: Zero-based neuron index to extract.

    Returns:
        A mapping containing the 2-D spatial RF (in ``x, y`` order), direct
        one-dimensional ``azimuth``, ``elevation``, ``orientation``, ``size``,
        and ``frequency`` correlation curves, and the validated integer
        ``indices`` used for extraction.

    Raises:
        ValueError: If the tensor/index dimensions are incompatible or a
            preferred feature lies outside the current RF tensor.  Failing here
            is safer than displaying a curve from a different cache.
    """
    values = np.asarray(rf_tensor)
    preferred = np.asarray(best_indices)
    if values.ndim != 6:
        raise ValueError(
            "RF tuning extraction requires a six-dimensional correlation tensor "
            "(neuron, x, y, orientation, size, frequency); "
            f"received shape {values.shape}."
        )
    if preferred.ndim != 2 or preferred.shape[0] < 4:
        raise ValueError(
            "Preferred RF indices must have shape (at least 4, n_neurons) in "
            f"x/y/orientation/size[/frequency] order; received {preferred.shape}."
        )
    neuron_index = int(neuron_index)
    if not 0 <= neuron_index < values.shape[0] or neuron_index >= preferred.shape[1]:
        raise ValueError(
            f"Neuron index {neuron_index} is outside RF data with "
            f"{values.shape[0]} tensor units and {preferred.shape[1]} index units."
        )

    raw = preferred[:, neuron_index]
    if not np.all(np.isfinite(raw[:4])):
        raise ValueError(f"Neuron {neuron_index} has non-finite preferred RF indices: {raw!r}.")
    x, y, orientation, size = (int(np.rint(value)) for value in raw[:4])
    frequency = int(np.rint(raw[4])) if raw.size > 4 and np.isfinite(raw[4]) else 0
    indices = (x, y, orientation, size, frequency)
    limits = values.shape[1:]
    if any(index < 0 or index >= limit for index, limit in zip(indices, limits)):
        raise ValueError(
            f"Neuron {neuron_index} preferred indices {indices} are outside RF "
            f"feature dimensions {limits}."
        )

    return {
        "rf_xy": np.asarray(values[neuron_index, :, :, orientation, size, frequency]),
        "azimuth": np.asarray(values[neuron_index, :, y, orientation, size, frequency]),
        "elevation": np.asarray(values[neuron_index, x, :, orientation, size, frequency]),
        "orientation": np.asarray(values[neuron_index, x, y, :, size, frequency]),
        "size": np.asarray(values[neuron_index, x, y, orientation, :, frequency]),
        "frequency": np.asarray(values[neuron_index, x, y, orientation, size, :]),
        "indices": np.asarray(indices, dtype=int),
    }
