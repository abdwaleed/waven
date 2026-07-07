"""Compatibility shim for the legacy LoadPinkNoise module.

New code should import stimulus wavelet helpers from :mod:`waven.stimulus` and
neural loading/alignment helpers from :mod:`waven.data.neural`. This shim keeps
older scripts working while the source tree uses clearer module boundaries.
"""
from .stimulus import *
from .data.neural import *
