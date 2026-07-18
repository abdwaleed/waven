"""Regression tests for visual-angle to movie-pixel geometry."""

import pytest

from waven.stimulus.metadata import coverage_crop_bounds, downsampled_grid_dimensions


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


def test_zero_downsample_percent_is_normalized_to_the_smallest_usable_grid():
    """Legacy saved GUI values of zero must not create a one-pixel cache."""
    assert downsampled_grid_dimensions({"width": 1920, "height": 1080}, 0) == (19, 11)


def test_coverage_calibrated_grid_has_square_visual_angle_pixels():
    """A 16:9 movie is resampled to the requested visual-field aspect ratio."""
    width, height = downsampled_grid_dimensions(
        {"width": 1280, "height": 720},
        32,
        analysis_coverage=(-69, 69, 56, -56),
    )
    assert (width, height) == (410, 333)
    assert abs((138 / width) - (112 / height)) < 0.001
