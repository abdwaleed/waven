"""Filter-bank recommendations expressed in visual, rather than pixel, units."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple


_DENSITY_COUNTS = {"Fast": 3, "Standard": 5, "Dense": 7}


@dataclass(frozen=True)
class FilterBankRecommendation:
    """A calibrated Gabor bank plus the internal values needed by WAVEN."""

    frequencies_cpd: Tuple[float, ...]
    sigmas_degrees: Tuple[float, ...]
    frequencies_per_pixel: Tuple[float, ...]
    sigmas_pixels: Tuple[float, ...]
    cycles_per_sigma: float
    warnings: Tuple[str, ...]

    @property
    def coupled_resource_multiplier(self) -> int:
        """Number of filter-size/frequency pairs in the coupled bank."""
        return len(self.frequencies_cpd)

    @property
    def independent_resource_multiplier(self) -> int:
        """Number of size/frequency combinations in the independent bank."""
        return len(self.frequencies_cpd) * len(self.sigmas_degrees)


def recommend_filter_bank(
    minimum_cpd: float,
    maximum_cpd: float,
    degrees_per_pixel: float,
    density: str = "Standard",
    cycles_per_sigma: float = 1.0,
    nyquist_cpd: float | None = None,
) -> FilterBankRecommendation:
    """Recommend log-spaced Gabor frequencies and matched Gaussian envelopes.

    A matched Gabor pair has ``sigma_deg = cycles_per_sigma / frequency_cpd``.
    This keeps a consistent number of carrier cycles within one Gaussian sigma,
    unlike arbitrary pixel-size and cycles/pixel lists.
    """
    minimum_cpd = float(minimum_cpd)
    maximum_cpd = float(maximum_cpd)
    degrees_per_pixel = float(degrees_per_pixel)
    cycles_per_sigma = float(cycles_per_sigma)
    if minimum_cpd <= 0 or maximum_cpd <= 0 or minimum_cpd >= maximum_cpd:
        raise ValueError("Use positive spatial-frequency limits with minimum below maximum.")
    if degrees_per_pixel <= 0:
        raise ValueError("Analysis sampling must be greater than zero degrees/pixel.")
    if cycles_per_sigma <= 0:
        raise ValueError("Cycles per Gaussian sigma must be greater than zero.")
    count = _DENSITY_COUNTS.get(str(density), _DENSITY_COUNTS["Standard"])
    log_min, log_max = math.log(minimum_cpd), math.log(maximum_cpd)
    frequencies = tuple(
        math.exp(log_min + index * (log_max - log_min) / (count - 1))
        for index in range(count)
    )
    sigmas_degrees = tuple(cycles_per_sigma / frequency for frequency in frequencies)
    warnings = []
    if nyquist_cpd is not None and maximum_cpd > float(nyquist_cpd):
        warnings.append(
            "Requested maximum is above the calibrated Nyquist limit; reduce it or choose finer analysis sampling."
        )
    elif nyquist_cpd is not None and maximum_cpd > float(nyquist_cpd) / 1.5:
        warnings.append(
            "Requested maximum is close to Nyquist; finer sampling will give it more reliable headroom."
        )
    return FilterBankRecommendation(
        frequencies_cpd=frequencies,
        sigmas_degrees=sigmas_degrees,
        frequencies_per_pixel=tuple(value * degrees_per_pixel for value in frequencies),
        sigmas_pixels=tuple(value / degrees_per_pixel for value in sigmas_degrees),
        cycles_per_sigma=cycles_per_sigma,
        warnings=tuple(warnings),
    )
