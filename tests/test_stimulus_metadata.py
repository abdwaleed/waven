"""Regression tests for visual-angle to movie-pixel geometry."""

import pytest

from waven.stimulus.metadata import coverage_crop_bounds


def test_coverage_crop_bounds_selects_the_requested_central_field():
    """Angular coverage differences are not themselves pixel offsets."""
    bounds = coverage_crop_bounds(
        (720, 1280),
        visual_coverage=(-69, 69, 56, -56),
        analysis_coverage=(-35, 35, 35, -35),
    )
    # 70 deg of a 138-deg visual field, centred horizontally; 70 deg of a
    # 112-deg visual field, centred vertically.
    assert bounds == (135, 585, 315, 965)


def test_coverage_crop_bounds_rejects_analysis_outside_the_visual_field():
    with pytest.raises(ValueError, match="within Visual Coverage"):
        coverage_crop_bounds(
            (100, 100),
            visual_coverage=(-10, 10, 10, -10),
            analysis_coverage=(-12, 10, 10, -10),
        )
