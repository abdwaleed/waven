"""Compatibility shim for the legacy WaveletGenerator module.

New code should import Gabor builders from :mod:`waven.wavelets.filters` and
video/wavelet decomposition helpers from :mod:`waven.wavelets.decomposition`.
This file keeps existing notebooks and scripts working while the package layout
uses clearer module names.
"""
from .wavelets import *
