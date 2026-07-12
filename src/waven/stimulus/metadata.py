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
    """Return ``(width, height)`` from authoritative movie metadata and a percentage."""
    percent = max(0.0, min(100.0, float(percent)))
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


__all__ = ["coverage_ratios", "downsampled_grid_dimensions", "read_movie_metadata"]
