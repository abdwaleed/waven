"""Lazy dependency bundles for optional GUI actions.

Importing :mod:`waven.app.gui` must remain cheap and must not initialize a
Matplotlib Tk backend, PyTorch, or a large analysis stack. Each loader below
owns one coherent optional dependency boundary and returns an immutable bundle
instead of mutating module-level globals.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable


@dataclass(frozen=True)
class PlotBackend:
    """Tk-compatible Matplotlib objects required for embedded figures."""

    pyplot: Any
    figure_canvas: type
    navigation_toolbar: type


@dataclass(frozen=True)
class WaveletOperations:
    """Disk-backed stimulus and wavelet cache operations."""

    build_convolution_kernel_cache: Callable[..., Any]
    convolution_kernel_cache_path: Callable[..., Any]
    downsample_video_binary: Callable[..., Any]
    wavelet_power_decomposition_conv: Callable[..., Any]
    wavelet_decomposition_full_conv: Callable[..., Any]
    coarse_rf_zarr_layout: Callable[..., Any]
    video_downsample_chunk_size: Callable[..., Any]


@dataclass(frozen=True)
class RFOperations:
    """Numerical operations needed by the Coarse RF GUI action."""

    compute_skewness_neurons: Callable[..., Any]
    pearson_correlation_pink_noise: Callable[..., Any]
    repeatability_trial: Callable[..., Any]
    close_orientation_curve: Callable[..., Any]
    correlation_orientation_tuning: Callable[..., Any]
    firing_rate_orientation_tuning: Callable[..., Any]
    orientation_tuning_bundle: Callable[..., Any]
    orientation_selectivity_from_tuning: Callable[..., Any]


@dataclass(frozen=True)
class ModelOperations:
    """Numerical operations needed by the model GUI actions."""

    run_model: Callable[..., Any]
    run_full_model: Callable[..., Any]
    smooth_best_positions: Callable[..., Any]


@lru_cache(maxsize=1)
def load_plot_backend() -> PlotBackend:
    """Load the Tk Matplotlib backend only when a figure must be displayed."""
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt

    return PlotBackend(plt, FigureCanvasTkAgg, NavigationToolbar2Tk)


@lru_cache(maxsize=1)
def load_wavelet_operations() -> WaveletOperations:
    """Load wavelet generation and storage operations on first use."""
    from ..runtime.performance import video_downsample_chunk_size
    from ..wavelets.decomposition import (
        build_convolution_kernel_cache,
        coarse_rf_zarr_layout,
        convolution_kernel_cache_path,
        downsample_video_binary,
        waveletDecompositionFullConv,
        waveletPowerDecompositionConv,
    )

    return WaveletOperations(
        build_convolution_kernel_cache,
        convolution_kernel_cache_path,
        downsample_video_binary,
        waveletPowerDecompositionConv,
        waveletDecompositionFullConv,
        coarse_rf_zarr_layout,
        video_downsample_chunk_size,
    )


@lru_cache(maxsize=1)
def load_rf_operations() -> RFOperations:
    """Load Coarse RF and orientation-tuning analysis functions on first use."""
    from ..analysis.orientation_selectivity import (
        close_orientation_curve,
        correlation_orientation_tuning,
        firing_rate_orientation_tuning,
        orientation_selectivity_from_tuning,
        orientation_tuning_bundle,
    )
    from ..analysis.receptive_fields import (
        compute_skewness_neurons,
        PearsonCorrelationPinkNoise,
        repetability_trial3,
    )

    return RFOperations(
        compute_skewness_neurons,
        PearsonCorrelationPinkNoise,
        repetability_trial3,
        close_orientation_curve,
        correlation_orientation_tuning,
        firing_rate_orientation_tuning,
        orientation_tuning_bundle,
        orientation_selectivity_from_tuning,
    )


@lru_cache(maxsize=1)
def load_model_operations() -> ModelOperations:
    """Load model execution functions on first use."""
    from ..analysis.model_runs import run_Full_Model, run_Model
    from ..pipeline import smooth_best_positions

    return ModelOperations(run_Model, run_Full_Model, smooth_best_positions)
