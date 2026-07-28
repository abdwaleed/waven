"""Authoritative stimulus-movie metadata and geometry helpers.

These functions are intentionally UI-independent: the GUI, CLI pipeline, and
alignment code can all derive dimensions and duration from the same source.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import numpy as np


MoviePath = Union[str, Path]


def centered_screen_visual_coverage(
    screen_width_cm: float,
    screen_height_cm: float,
    viewing_distance_cm: float,
) -> Tuple[float, float, float, float]:
    """Return visual coverage for a screen centred on a single eye.

    The returned order matches the GUI coverage fields: ``(left, right, top,
    bottom)`` in degrees.  It assumes the eye is level with, and centred on,
    the active display area.
    """
    try:
        width, height, distance = (
            float(screen_width_cm),
            float(screen_height_cm),
            float(viewing_distance_cm),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Screen width, screen height, and viewing distance must each be a positive finite value in cm."
        ) from exc
    if not all(math.isfinite(value) and value > 0 for value in (width, height, distance)):
        raise ValueError(
            "Screen width, screen height, and viewing distance must each be a positive finite value in cm."
        )
    horizontal_angle = math.degrees(math.atan((width / 2.0) / distance))
    vertical_angle = math.degrees(math.atan((height / 2.0) / distance))
    return -horizontal_angle, horizontal_angle, vertical_angle, -vertical_angle


def validate_coverage_pair(
    visual_coverage: Sequence[float], analysis_coverage: Sequence[float]
) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
    """Validate that a non-empty analysis field is entirely within the visual field."""
    if len(visual_coverage) != 4 or len(analysis_coverage) != 4:
        raise ValueError("Coverage values must each contain left, right, top, and bottom values.")

    visual = tuple(float(value) for value in visual_coverage)
    analysis = tuple(float(value) for value in analysis_coverage)
    if not all(math.isfinite(value) for value in (*visual, *analysis)):
        raise ValueError("Coverage values must be finite numbers.")

    visual_x_min, visual_x_max = sorted(visual[:2])
    visual_y_min, visual_y_max = sorted(visual[2:])
    analysis_x_min, analysis_x_max = sorted(analysis[:2])
    analysis_y_min, analysis_y_max = sorted(analysis[2:])
    if visual_x_max <= visual_x_min or visual_y_max <= visual_y_min:
        raise ValueError("Visual Coverage must span a non-zero horizontal and vertical field.")
    if analysis_x_max <= analysis_x_min or analysis_y_max <= analysis_y_min:
        raise ValueError("Analysis Coverage must span a non-zero horizontal and vertical field.")

    tolerance = 1e-9
    if (
        analysis_x_min < visual_x_min - tolerance
        or analysis_x_max > visual_x_max + tolerance
        or analysis_y_min < visual_y_min - tolerance
        or analysis_y_max > visual_y_max + tolerance
    ):
        raise ValueError("Analysis Coverage must lie within Visual Coverage.")
    return visual, analysis


def read_movie_metadata(path: MoviePath) -> Dict[str, float]:
    """Read validated width, height, frame count, FPS, and duration from a movie."""
    import cv2

    movie_path = Path(path)
    capture = cv2.VideoCapture(str(movie_path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open stimulus movie: {movie_path}")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(width, height, frames) <= 0 or fps <= 0:
        raise ValueError(f"Stimulus movie metadata is incomplete: {movie_path}")
    return {
        "width": width,
        "height": height,
        "frames": frames,
        "fps": fps,
        "duration": frames / fps,
    }


def downsampled_grid_dimensions(
    metadata: Dict[str, float],
    percent: float,
    analysis_coverage: Sequence[float] | None = None,
) -> Tuple[int, int]:
    """Return a usable analysis grid, optionally with square visual-angle pixels.

    A zero-sized stimulus grid has no scientific interpretation and previously
    collapsed to a misleading one-pixel cache.  Clamp legacy zero-percent
    configurations to the smallest supported setting (1%) instead.

    When ``analysis_coverage`` is supplied, the percentage determines the
    horizontal sampling density and the height is derived from the angular
    field aspect ratio.  This deliberately resamples a non-square stimulus
    video onto pixels that are square in visual degrees, so an isotropic Gabor
    in pixel coordinates is also isotropic in the animal's visual field.
    Callers omitting coverage retain the legacy video-aspect grid.
    """
    percent = max(1.0, min(100.0, float(percent)))
    width = max(1, int(round(float(metadata["width"]) * percent / 100.0)))
    if analysis_coverage is None:
        height = max(1, int(round(float(metadata["height"]) * percent / 100.0)))
        return width, height
    if len(analysis_coverage) != 4:
        raise ValueError("Analysis Coverage must contain x_max, x_min, y_max, y_min.")
    x_max, x_min, y_max, y_min = (float(value) for value in analysis_coverage)
    x_span = abs(x_max - x_min)
    y_span = abs(y_max - y_min)
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Analysis Coverage must span non-zero horizontal and vertical fields.")
    height = max(1, int(round(width * y_span / x_span)))
    return width, height


def coverage_ratios(
    visual_coverage: Sequence[float], analysis_coverage: Sequence[float]
) -> Tuple[float, float]:
    """Return spatial crop ratios used during stimulus downsampling."""
    visual_coverage, analysis_coverage = validate_coverage_pair(
        visual_coverage, analysis_coverage
    )
    if tuple(visual_coverage) == tuple(analysis_coverage):
        return 1.0, 1.0
    visual = np.asarray(visual_coverage, dtype=float)
    analysis = np.asarray(analysis_coverage, dtype=float)
    ratio_x = 1 - ((visual[0] - visual[1]) - (analysis[0] - analysis[1])) / (visual[0] - visual[1])
    ratio_y = 1 - ((visual[2] - visual[3]) - (analysis[2] - analysis[3])) / (visual[2] - visual[3])
    return float(ratio_x), float(ratio_y)


def coverage_crop_bounds(
    frame_shape: Sequence[int],
    visual_coverage: Sequence[float],
    analysis_coverage: Sequence[float],
) -> Tuple[int, int, int, int]:
    """Return pixel ``(row_start, row_end, col_start, col_end)`` crop bounds.

    Coverage is expressed as ``(x_max, x_min, y_max, y_min)`` in visual-angle
    coordinates.  Image rows increase downward, so the vertical coordinate is
    intentionally converted from ``y_max`` at the top of the frame.  The
    calculation preserves the requested central analysis field rather than
    treating angular differences as pixel offsets.
    """
    if len(frame_shape) < 2:
        raise ValueError("frame_shape must provide image height and width.")
    height, width = (int(frame_shape[0]), int(frame_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("frame_shape dimensions must be positive.")
    visual, analysis = validate_coverage_pair(visual_coverage, analysis_coverage)
    visual_x_min, visual_x_max = sorted(visual[:2])
    visual_y_min, visual_y_max = sorted(visual[2:])
    analysis_x_min, analysis_x_max = sorted(analysis[:2])
    analysis_y_min, analysis_y_max = sorted(analysis[2:])
    x_span = visual_x_max - visual_x_min
    y_span = visual_y_max - visual_y_min

    col_start = int(np.floor((analysis_x_min - visual_x_min) * width / x_span))
    col_end = int(np.ceil((analysis_x_max - visual_x_min) * width / x_span))
    row_start = int(np.floor((visual_y_max - analysis_y_max) * height / y_span))
    row_end = int(np.ceil((visual_y_max - analysis_y_min) * height / y_span))
    row_start, row_end = max(0, row_start), min(height, row_end)
    col_start, col_end = max(0, col_start), min(width, col_end)
    if row_end <= row_start or col_end <= col_start:
        raise ValueError("Coverage crop is empty after conversion to image pixels.")
    return row_start, row_end, col_start, col_end


__all__ = [
    "centered_screen_visual_coverage",
    "coverage_crop_bounds",
    "coverage_ratios",
    "downsampled_grid_dimensions",
    "read_movie_metadata",
    "validate_coverage_pair",
]
