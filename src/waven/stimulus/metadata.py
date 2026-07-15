"""Authoritative stimulus-movie metadata and geometry helpers.

These functions are intentionally UI-independent: the GUI, CLI pipeline, and
alignment code can all derive dimensions and duration from the same source.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple, Union

import cv2
import numpy as np


MoviePath = Union[str, Path]


def read_movie_metadata(path: MoviePath) -> Dict[str, float]:
    """Read validated width, height, frame count, FPS, and duration from a movie."""
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


def downsampled_grid_dimensions(metadata: Dict[str, float], percent: float) -> Tuple[int, int]:
    """Return ``(width, height)`` from movie metadata and a usable percentage.

    A zero-sized stimulus grid has no scientific interpretation and previously
    collapsed to a misleading one-pixel cache.  Clamp legacy zero-percent
    configurations to the smallest supported setting (1%) instead.
    """
    percent = max(1.0, min(100.0, float(percent)))
    return (
        max(1, int(round(float(metadata["width"]) * percent / 100.0))),
        max(1, int(round(float(metadata["height"]) * percent / 100.0))),
    )


def coverage_ratios(
    visual_coverage: Sequence[float], analysis_coverage: Sequence[float]
) -> Tuple[float, float]:
    """Return spatial crop ratios used during stimulus downsampling."""
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
    if len(visual_coverage) != 4 or len(analysis_coverage) != 4:
        raise ValueError("Coverage values must each contain x_max, x_min, y_max, y_min.")

    vx_max, vx_min, vy_max, vy_min = (float(value) for value in visual_coverage)
    ax_max, ax_min, ay_max, ay_min = (float(value) for value in analysis_coverage)
    visual_x_min, visual_x_max = sorted((vx_min, vx_max))
    visual_y_min, visual_y_max = sorted((vy_min, vy_max))
    analysis_x_min, analysis_x_max = sorted((ax_min, ax_max))
    analysis_y_min, analysis_y_max = sorted((ay_min, ay_max))
    x_span = visual_x_max - visual_x_min
    y_span = visual_y_max - visual_y_min
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Visual Coverage must span a non-zero horizontal and vertical field.")
    tolerance = 1e-9
    if (
        analysis_x_min < visual_x_min - tolerance
        or analysis_x_max > visual_x_max + tolerance
        or analysis_y_min < visual_y_min - tolerance
        or analysis_y_max > visual_y_max + tolerance
    ):
        raise ValueError("Analysis Coverage must lie within Visual Coverage.")

    col_start = int(np.floor((analysis_x_min - visual_x_min) * width / x_span))
    col_end = int(np.ceil((analysis_x_max - visual_x_min) * width / x_span))
    row_start = int(np.floor((visual_y_max - analysis_y_max) * height / y_span))
    row_end = int(np.ceil((visual_y_max - analysis_y_min) * height / y_span))
    row_start, row_end = max(0, row_start), min(height, row_end)
    col_start, col_end = max(0, col_start), min(width, col_end)
    if row_end <= row_start or col_end <= col_start:
        raise ValueError("Coverage crop is empty after conversion to image pixels.")
    return row_start, row_end, col_start, col_end


__all__ = ["coverage_crop_bounds", "coverage_ratios", "downsampled_grid_dimensions", "read_movie_metadata"]
