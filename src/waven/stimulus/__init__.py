"""Stimulus helpers with lightweight imports for geometry-only workflows."""
from __future__ import annotations

from importlib import import_module

from .metadata import coverage_crop_bounds, coverage_ratios, downsampled_grid_dimensions, read_movie_metadata
from .sampling import (
    SamplingPlan,
    sampling_plan_from_degrees_per_pixel,
    sampling_plan_from_max_cpd,
    sampling_plan_from_percent,
)


_LAZY_EXPORTS = {
    "load_stimulus": (".full_model", "load_stimulus"),
    "load_wavelets": (".full_model", "load_wavelets"),
    "coarseWavelet": (".wavelet_cache", "coarseWavelet"),
    "load_stimulus_simple_cell": (".wavelet_cache", "load_stimulus_simple_cell"),
    "load_stimulus_simple_cell2": (".wavelet_cache", "load_stimulus_simple_cell2"),
    "load_stimulus_simple_cell2_i": (".wavelet_cache", "load_stimulus_simple_cell2_i"),
    "load_stimulus_simple_cell2_r": (".wavelet_cache", "load_stimulus_simple_cell2_r"),
}


def __getattr__(name):
    """Load array/plotting helpers only when a caller actually needs them."""
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


__all__ = [
    "load_wavelets",
    "load_stimulus",
    "load_stimulus_simple_cell",
    "load_stimulus_simple_cell2_i",
    "load_stimulus_simple_cell2_r",
    "load_stimulus_simple_cell2",
    "coarseWavelet",
    "coverage_crop_bounds",
    "coverage_ratios",
    "downsampled_grid_dimensions",
    "read_movie_metadata",
    "SamplingPlan",
    "sampling_plan_from_percent",
    "sampling_plan_from_degrees_per_pixel",
    "sampling_plan_from_max_cpd",
]
