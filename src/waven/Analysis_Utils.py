"""Compatibility shim for the legacy Analysis_Utils module.

New code should import from the responsibility-focused modules under
:mod:`waven.analysis`: ``receptive_fields``, ``nonlinear``, ``trial_stats``, and
``model_runs``. This module re-exports those names for older notebooks and GUI
code that still imports :mod:`waven.Analysis_Utils`.
"""
from .analysis import *
