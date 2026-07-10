"""Compare legacy flattened Gabor projection with the convolution backend.

This script builds an 8 x 8 synthetic movie, runs both decomposition backends,
and writes review artifacts to a temporary output directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.waven.wavelets.decomposition import (
    waveletDecomposition,
    waveletDecompositionConv,
    waveletDecompositionFull,
    waveletDecompositionFullConv,
)
from src.waven.wavelets.filters import makeFilterLibrary, makeFilterLibrary2


def _max_diff_record(name, legacy, convolution):
    """Return summary statistics for one legacy/convolution comparison."""
    diff = np.abs(legacy - convolution)
    max_index = tuple(int(value) for value in np.unravel_index(np.argmax(diff), diff.shape))
    return {
        "name": name,
        "shape": tuple(int(value) for value in legacy.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_abs_diff_index": max_index,
        "legacy_at_max": float(legacy[max_index]),
        "convolution_at_max": float(convolution[max_index]),
    }


def _save_heatmap(path, title, diff):
    """Save an x/y heatmap of maximum absolute difference over other axes."""
    if diff.ndim < 3:
        heatmap = diff
    else:
        # Arrays are time, x, y, ...
        reduce_axes = tuple(axis for axis in range(diff.ndim) if axis not in (1, 2))
        heatmap = diff.max(axis=reduce_axes)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(heatmap.T, origin="lower", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="max abs diff")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run(output_dir=None):
    """Run the 8 x 8 comparison and return the output directory."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="waven_conv_8x8_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_dir = output_dir / "work"
    work_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(42)
    num_frames = 4
    nx = ny = 8
    n_orientations = 4
    sigmas = np.asarray([1.0, 2.0], dtype=float)
    frequencies = np.asarray([0.08, 0.14], dtype=float)
    phases = np.asarray([0.0, np.pi / 2], dtype=float)
    xs = np.arange(nx)
    ys = np.arange(ny)
    thetas = np.asarray([(idx * np.pi) / n_orientations for idx in range(n_orientations)])
    video = rng.normal(size=(num_frames, ny, nx)).astype(np.float32)

    np.save(output_dir / "video_8x8.npy", video)

    records = []

    coarse_library = makeFilterLibrary(
        xs,
        ys,
        thetas,
        sigmas,
        phases,
        frequencies[0],
        freq=False,
    )
    coarse_library_path = work_dir / "legacy_coarse_library.npy"
    np.save(coarse_library_path, coarse_library)

    for phase_index, phase_name in ((0, "real"), (1, "imaginary")):
        legacy_path = work_dir / f"dwt_videodata_{phase_index}.npy"
        waveletDecomposition(
            video,
            phase_index,
            sigmas,
            str(work_dir),
            str(coarse_library_path),
        )
        legacy = np.load(legacy_path).copy()
        legacy_path.unlink()

        waveletDecompositionConv(
            video,
            phase_index,
            sigmas,
            str(work_dir),
            n_orientations=n_orientations,
            phase_offsets=phases,
            frame_chunk_size=2,
        )
        convolution = np.load(legacy_path).copy()
        legacy_path.unlink()

        np.save(output_dir / f"coarse_{phase_name}_legacy.npy", legacy)
        np.save(output_dir / f"coarse_{phase_name}_convolution.npy", convolution)
        np.save(output_dir / f"coarse_{phase_name}_abs_diff.npy", np.abs(legacy - convolution))
        _save_heatmap(
            output_dir / f"coarse_{phase_name}_max_abs_diff.png",
            f"Coarse {phase_name} max abs diff",
            np.abs(legacy - convolution),
        )
        records.append(_max_diff_record(f"coarse_{phase_name}", legacy, convolution))

    full_library = makeFilterLibrary2(xs, ys, thetas, sigmas, phases, frequencies)
    full_library_path = work_dir / "legacy_full_library.npy"
    np.save(full_library_path, full_library)

    for phase_index, phase_name, suffix in ((0, "real", "_r"), (1, "imaginary", "_i")):
        legacy_path = work_dir / f"dwt_videodata2{suffix}.npy"
        waveletDecompositionFull(
            video,
            phase_index,
            sigmas,
            frequencies,
            str(work_dir),
            str(full_library_path),
            library_sigmas=sigmas,
            output_format="npy",
        )
        legacy = np.load(legacy_path).copy()
        legacy_path.unlink()

        waveletDecompositionFullConv(
            video,
            phase_index,
            sigmas,
            frequencies,
            str(work_dir),
            n_orientations=n_orientations,
            phase_offsets=phases,
            output_format="npy",
            frame_chunk_size=2,
        )
        convolution = np.load(legacy_path).copy()
        legacy_path.unlink()

        np.save(output_dir / f"full_{phase_name}_legacy.npy", legacy)
        np.save(output_dir / f"full_{phase_name}_convolution.npy", convolution)
        np.save(output_dir / f"full_{phase_name}_abs_diff.npy", np.abs(legacy - convolution))
        _save_heatmap(
            output_dir / f"full_{phase_name}_max_abs_diff.png",
            f"Full {phase_name} max abs diff",
            np.abs(legacy - convolution),
        )
        records.append(_max_diff_record(f"full_{phase_name}", legacy, convolution))

    summary = {
        "movie_shape": tuple(int(value) for value in video.shape),
        "nx": nx,
        "ny": ny,
        "n_orientations": n_orientations,
        "sigmas": sigmas.tolist(),
        "frequencies": frequencies.tolist(),
        "phases": phases.tolist(),
        "records": records,
        "tolerance_note": "Differences are expected to be small because legacy libraries are stored as float16.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    with (output_dir / "diff_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    readme = output_dir / "README.md"
    readme.write_text(
        "# 8 x 8 Convolution Backend Comparison\n\n"
        "This folder compares the legacy flattened Gabor-library projection "
        "against the PyTorch convolution backend on a synthetic 8 x 8 movie.\n\n"
        "Review `summary.json` and `diff_summary.csv` for numeric error. "
        "The PNG heatmaps show the maximum absolute difference at each x/y "
        "position after reducing over time and filter axes.\n",
        encoding="utf-8",
    )
    return output_dir


def main():
    """Parse command-line arguments and run the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="Directory for saved comparison artifacts.")
    args = parser.parse_args()
    output_dir = run(args.output_dir)
    print(f"Saved 8x8 convolution comparison artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
