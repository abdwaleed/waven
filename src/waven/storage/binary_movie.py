"""Disk-backed views of binary stimulus movies."""
from __future__ import annotations

from typing import Any

import numpy as np


class SignedBinaryMovie:
    """Expose a boolean/0-1 movie as signed values without a full copy.

    The convolution backend requests frame slices.  Each slice is converted
    from ``{0, 1}`` to ``{-1, 1}`` only when requested, so Zarr arrays and NPY
    memmaps keep their disk-backed behavior throughout decomposition.
    """

    def __init__(self, source: Any) -> None:
        self._source = source
        self.shape = tuple(int(dim) for dim in source.shape)

    def __getitem__(self, item):
        return np.asarray(self._source[item], dtype=np.int8) * 2 - 1


__all__ = ["SignedBinaryMovie"]
