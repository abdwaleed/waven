"""Wavelet helpers with lazy heavy numerical imports.

Filter-bank planning is useful in the GUI before PyTorch, Matplotlib, or the
decomposition stack needs to be initialized.  Existing public wavelet exports
remain available through ``__getattr__``.
"""
from __future__ import annotations

from importlib import import_module

from .recommendations import FilterBankRecommendation, recommend_filter_bank


_FILTER_EXPORTS = {
    "has_enough_ram",
    "makeFilterLibrary",
    "makeFilterLibrary2",
    "makeFilterLibrary3D",
    "makeGaborFilter",
    "makeGaborFilter3D",
}
_DECOMPOSITION_EXPORTS = {
    "build_convolution_kernel_cache",
    "convolution_kernel_cache_path",
    "downsample_video_binary",
    "downsample_video_uint",
    "getTrueRF",
    "getWTfromNPY",
    "waveletDecomposition",
    "waveletDecompositionConv",
    "waveletPowerDecompositionConv",
    "waveletDecompositionFull",
    "waveletDecompositionFullConv",
    "waveletTransform",
    "waveletTransform3D",
}


def __getattr__(name):
    """Resolve legacy public helpers without eager numerical imports."""
    if name in _FILTER_EXPORTS:
        value = getattr(import_module(".filters", __name__), name)
    elif name in _DECOMPOSITION_EXPORTS:
        value = getattr(import_module(".decomposition", __name__), name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


__all__ = [
    "has_enough_ram",
    "build_convolution_kernel_cache",
    "convolution_kernel_cache_path",
    "makeGaborFilter",
    "makeGaborFilter3D",
    "makeFilterLibrary",
    "makeFilterLibrary2",
    "makeFilterLibrary3D",
    "downsample_video_binary",
    "downsample_video_uint",
    "getWTfromNPY",
    "waveletTransform",
    "waveletTransform3D",
    "waveletDecomposition",
    "waveletDecompositionConv",
    "waveletPowerDecompositionConv",
    "waveletDecompositionFull",
    "waveletDecompositionFullConv",
    "getTrueRF",
    "FilterBankRecommendation",
    "recommend_filter_bank",
]
