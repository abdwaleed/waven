"""Scientifically calibrated analysis-grid planning.

The movie is resampled onto pixels that are square in visual degrees.  These
helpers turn user-facing sampling goals into the compatibility percentage that
the cache naming and preparation code still uses internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

from .metadata import downsampled_grid_dimensions


@dataclass(frozen=True)
class SamplingPlan:
    """A reproducible visual-angle sampling plan for one stimulus movie."""

    horizontal_percent: float
    grid_width: int
    grid_height: int
    degrees_per_pixel_x: float
    degrees_per_pixel_y: float
    nyquist_cpd: float
    was_clamped: bool = False

    @property
    def degrees_per_pixel(self) -> float:
        """Return the mean calibrated sampling, accounting for pixel rounding."""
        return (self.degrees_per_pixel_x + self.degrees_per_pixel_y) / 2.0


def _coverage_spans(analysis_coverage: Sequence[float]) -> tuple[float, float]:
    """Validate coverage and return horizontal/vertical angular spans."""
    if len(analysis_coverage) != 4:
        raise ValueError("Analysis Coverage must contain x_max, x_min, y_max, y_min.")
    x_max, x_min, y_max, y_min = (float(value) for value in analysis_coverage)
    x_span = abs(x_max - x_min)
    y_span = abs(y_max - y_min)
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Analysis Coverage must span non-zero horizontal and vertical fields.")
    return x_span, y_span


def sampling_plan_from_percent(
    metadata: Dict[str, float], percent: float, analysis_coverage: Sequence[float]
) -> SamplingPlan:
    """Describe the calibrated grid obtained from a legacy percentage setting."""
    requested_percent = float(percent)
    bounded_percent = max(1.0, min(100.0, requested_percent))
    grid_width, grid_height = downsampled_grid_dimensions(
        metadata, bounded_percent, analysis_coverage
    )
    x_span, y_span = _coverage_spans(analysis_coverage)
    degrees_x = x_span / grid_width
    degrees_y = y_span / grid_height
    degrees_per_pixel = (degrees_x + degrees_y) / 2.0
    return SamplingPlan(
        horizontal_percent=bounded_percent,
        grid_width=grid_width,
        grid_height=grid_height,
        degrees_per_pixel_x=degrees_x,
        degrees_per_pixel_y=degrees_y,
        nyquist_cpd=1.0 / (2.0 * degrees_per_pixel),
        was_clamped=bounded_percent != requested_percent,
    )


def sampling_plan_from_degrees_per_pixel(
    metadata: Dict[str, float], target_degrees_per_pixel: float, analysis_coverage: Sequence[float]
) -> SamplingPlan:
    """Plan the closest supported grid for a target visual-angle sampling."""
    target = float(target_degrees_per_pixel)
    if target <= 0:
        raise ValueError("Analysis sampling must be greater than zero degrees/pixel.")
    x_span, _ = _coverage_spans(analysis_coverage)
    width = float(metadata.get("width", 0))
    if width <= 0:
        raise ValueError("Movie metadata must contain a positive source width.")
    requested_percent = 100.0 * x_span / (width * target)
    plan = sampling_plan_from_percent(metadata, requested_percent, analysis_coverage)
    return SamplingPlan(**{**plan.__dict__, "was_clamped": plan.was_clamped or requested_percent < 1 or requested_percent > 100})


def sampling_plan_from_max_cpd(
    metadata: Dict[str, float],
    maximum_cpd: float,
    analysis_coverage: Sequence[float],
    oversampling_factor: float = 1.5,
) -> SamplingPlan:
    """Plan sampling that retains ``maximum_cpd`` with a safety margin.

    ``oversampling_factor=1.5`` places the selected limit at two thirds of
    Nyquist: useful resolution remains above the requested frequency rather
    than relying on the unstable edge of the sampling limit.
    """
    maximum_cpd = float(maximum_cpd)
    oversampling_factor = float(oversampling_factor)
    if maximum_cpd <= 0:
        raise ValueError("Maximum retained spatial frequency must be greater than zero cpd.")
    if oversampling_factor < 1:
        raise ValueError("Oversampling factor must be at least one.")
    target_degrees_per_pixel = 1.0 / (2.0 * maximum_cpd * oversampling_factor)
    return sampling_plan_from_degrees_per_pixel(
        metadata, target_degrees_per_pixel, analysis_coverage
    )
