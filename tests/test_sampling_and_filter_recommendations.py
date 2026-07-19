"""Regression coverage for scientific sampling and filter-bank planning."""
from __future__ import annotations

import pytest

from waven.stimulus.sampling import (
    sampling_plan_from_degrees_per_pixel,
    sampling_plan_from_max_cpd,
)
from waven.wavelets.recommendations import recommend_filter_bank


METADATA = {"width": 1280, "height": 720}
COVERAGE = (69.0, -69.0, 56.0, -56.0)


def test_target_sampling_preserves_square_visual_angle_pixels():
    plan = sampling_plan_from_degrees_per_pixel(METADATA, 0.75, COVERAGE)

    assert plan.grid_width > plan.grid_height
    assert plan.degrees_per_pixel_x == pytest.approx(plan.degrees_per_pixel_y, rel=0.01)
    assert plan.degrees_per_pixel == pytest.approx(0.75, rel=0.02)


def test_maximum_frequency_plans_headroom_above_requested_limit():
    plan = sampling_plan_from_max_cpd(METADATA, 0.4, COVERAGE)

    assert plan.nyquist_cpd >= 0.4 * 1.45


def test_filter_recommendation_is_log_spaced_and_calibrated():
    recommendation = recommend_filter_bank(0.025, 0.4, 0.5, "Standard", 1.0, 1.0)

    assert recommendation.frequencies_cpd == pytest.approx((0.025, 0.05, 0.1, 0.2, 0.4))
    assert recommendation.sigmas_degrees[0] > recommendation.sigmas_degrees[-1]
    assert recommendation.frequencies_per_pixel[-1] == pytest.approx(0.2)
    assert recommendation.coupled_resource_multiplier == 5
    assert recommendation.independent_resource_multiplier == 25
