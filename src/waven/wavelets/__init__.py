"""Wavelet-domain helpers grouped by responsibility.

``filters`` builds Gabor filters and libraries. ``decomposition`` applies those
libraries to stimulus videos and stores wavelet coefficient arrays.
"""
from .filters import (
    has_enough_ram,
    makeFilterLibrary,
    makeFilterLibrary2,
    makeFilterLibrary3D,
    makeGaborFilter,
    makeGaborFilter3D,
)
from .decomposition import (
    downsample_video_binary,
    downsample_video_uint,
    getTrueRF,
    getWTfromNPY,
    waveletDecomposition,
    waveletDecompositionFull,
    waveletTransform,
    waveletTransform3D,
)

__all__ = [
    "has_enough_ram",
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
    "waveletDecompositionFull",
    "getTrueRF",
]
