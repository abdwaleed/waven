"""Stress-test convolution wavelets against the legacy flattened library.

The suite covers several small but deliberately awkward cases:

* square and rectangular movies
* odd and even image dimensions
* random, binary, impulse, edge impulse, gradient, and checkerboard stimuli
* both real and imaginary phase indices
* coarse coupled-frequency and full independent-frequency outputs

Artifacts are written as JSON/CSV plus arrays for the worst observed mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

os.environ.setdefault("waven_NO_PLOTS", "1")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.waven.wavelets.decomposition import (
    build_convolution_kernel_cache,
    waveletDecomposition,
    waveletDecompositionConv,
    waveletDecompositionFull,
    waveletDecompositionFullConv,
)
from src.waven.wavelets.filters import makeFilterLibrary, makeFilterLibrary2


PHASES = np.asarray([0.0, np.pi / 2], dtype=float)


def _stimulus(kind, frames, ny, nx, seed):
    """Return a deterministic test movie with shape ``(frames, ny, nx)``."""
    rng = np.random.default_rng(seed)
    if kind == "random_normal":
        return rng.normal(size=(frames, ny, nx)).astype(np.float32)
    if kind == "binary_pm1":
        return rng.choice([-1.0, 1.0], size=(frames, ny, nx)).astype(np.float32)
    if kind == "center_impulse":
        movie = np.zeros((frames, ny, nx), dtype=np.float32)
        movie[:, ny // 2, nx // 2] = 1.0
        return movie
    if kind == "corner_impulse":
        movie = np.zeros((frames, ny, nx), dtype=np.float32)
        movie[:, 0, 0] = 1.0
        movie[:, -1, -1] = -1.0
        return movie
    if kind == "gradient":
        yy, xx = np.mgrid[0:ny, 0:nx]
        base = (xx / max(nx - 1, 1)) - (yy / max(ny - 1, 1))
        return np.stack([(idx + 1) * base for idx in range(frames)], axis=0).astype(np.float32)
    if kind == "checkerboard":
        yy, xx = np.mgrid[0:ny, 0:nx]
        base = ((xx + yy) % 2) * 2 - 1
        return np.stack([base * ((-1) ** idx) for idx in range(frames)], axis=0).astype(np.float32)
    raise ValueError(f"Unknown stimulus kind: {kind}")


def _edge_mask(shape):
    """Return a boolean x/y edge mask for an output array."""
    _, nx, ny = shape[:3]
    mask = np.zeros((nx, ny), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _record(case, stimulus_name, backend_name, phase_index, legacy, convolution, elapsed_legacy, elapsed_conv):
    """Return one numeric comparison record."""
    diff = np.abs(legacy - convolution)
    edge = _edge_mask(diff.shape)
    edge_selector = (slice(None), edge) + tuple(slice(None) for _ in range(diff.ndim - 3))
    interior_selector = (slice(None), ~edge) + tuple(slice(None) for _ in range(diff.ndim - 3))
    max_index = tuple(int(value) for value in np.unravel_index(np.argmax(diff), diff.shape))
    denom = max(float(np.max(np.abs(legacy))), 1e-12)
    return {
        "case": case["name"],
        "stimulus": stimulus_name,
        "backend": backend_name,
        "phase_index": int(phase_index),
        "shape": "x".join(str(int(value)) for value in legacy.shape),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "relative_max_diff": float(diff.max() / denom),
        "edge_max_abs_diff": float(diff[edge_selector].max()) if np.any(edge) else 0.0,
        "interior_max_abs_diff": float(diff[interior_selector].max()) if np.any(~edge) else 0.0,
        "legacy_seconds": float(elapsed_legacy),
        "convolution_seconds": float(elapsed_conv),
        "speedup_legacy_over_conv": float(elapsed_legacy / max(elapsed_conv, 1e-9)),
        "max_index": repr(max_index),
        "legacy_at_max": float(legacy[max_index]),
        "convolution_at_max": float(convolution[max_index]),
    }


def _save_worst_artifacts(output_dir, worst):
    """Save arrays and a heatmap for the worst comparison in the run."""
    if worst is None:
        return
    legacy = worst["legacy"]
    convolution = worst["convolution"]
    diff = np.abs(legacy - convolution)
    np.save(output_dir / "worst_legacy.npy", legacy)
    np.save(output_dir / "worst_convolution.npy", convolution)
    np.save(output_dir / "worst_abs_diff.npy", diff)
    reduce_axes = tuple(axis for axis in range(diff.ndim) if axis not in (1, 2))
    heatmap = diff.max(axis=reduce_axes)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(heatmap.T, origin="lower", cmap="magma")
    ax.set_title("Worst comparison max abs diff by x/y")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, label="max abs diff")
    fig.tight_layout()
    fig.savefig(output_dir / "worst_max_abs_diff_heatmap.png", dpi=160)
    plt.close(fig)


def _run_one_decomposition(func, *args, **kwargs):
    """Run a decomposition function and return elapsed seconds."""
    start = time.perf_counter()
    func(*args, **kwargs)
    return time.perf_counter() - start


def run(output_dir=None, keep_work=False):
    """Run all stress cases and return ``(output_dir, summary)``."""
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="waven_conv_stress_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = output_dir / "work"
    work_root.mkdir(exist_ok=True)

    cases = [
        {"name": "square_8", "nx": 8, "ny": 8, "frames": 4, "n_orientations": 4, "sigmas": [1.0, 2.0], "frequencies": [0.08, 0.14]},
        {"name": "odd_rect_9x7", "nx": 9, "ny": 7, "frames": 3, "n_orientations": 5, "sigmas": [0.75, 1.5], "frequencies": [0.05, 0.11]},
        {"name": "rect_12x10", "nx": 12, "ny": 10, "frames": 3, "n_orientations": 6, "sigmas": [1.0, 2.5], "frequencies": [0.04, 0.12, 0.18]},
        {"name": "larger_16x12", "nx": 16, "ny": 12, "frames": 2, "n_orientations": 8, "sigmas": [1.0, 2.0, 3.0], "frequencies": [0.04, 0.09]},
    ]
    stimuli = [
        "random_normal",
        "binary_pm1",
        "center_impulse",
        "corner_impulse",
        "gradient",
        "checkerboard",
    ]

    records = []
    worst = None

    for case_index, case in enumerate(cases):
        case_dir = work_root / case["name"]
        case_dir.mkdir(exist_ok=True)
        nx = int(case["nx"])
        ny = int(case["ny"])
        frames = int(case["frames"])
        n_orientations = int(case["n_orientations"])
        sigmas = np.asarray(case["sigmas"], dtype=float)
        frequencies = np.asarray(case["frequencies"], dtype=float)
        xs = np.arange(nx)
        ys = np.arange(ny)
        thetas = np.asarray([(idx * np.pi) / n_orientations for idx in range(n_orientations)])

        coarse_library = makeFilterLibrary(xs, ys, thetas, sigmas, PHASES, frequencies[0], freq=False)
        coarse_library_path = case_dir / "coarse_library.npy"
        np.save(coarse_library_path, coarse_library)
        full_library = makeFilterLibrary2(xs, ys, thetas, sigmas, PHASES, frequencies)
        full_library_path = case_dir / "full_library.npy"
        np.save(full_library_path, full_library)
        coarse_kernel_cache_path = build_convolution_kernel_cache(
            str(case_dir),
            "coarse",
            sigmas,
            n_orientations,
            phase_offsets=PHASES,
        )
        fine_kernel_cache_path = build_convolution_kernel_cache(
            str(case_dir),
            "fine",
            sigmas,
            n_orientations,
            phase_offsets=PHASES,
            frequencies=frequencies,
        )

        for stimulus_index, stimulus_name in enumerate(stimuli):
            movie = _stimulus(stimulus_name, frames, ny, nx, seed=1000 + case_index * 100 + stimulus_index)

            for phase_index in (0, 1):
                coarse_path = case_dir / f"dwt_videodata_{phase_index}.npy"
                elapsed_legacy = _run_one_decomposition(
                    waveletDecomposition,
                    movie,
                    phase_index,
                    sigmas,
                    str(case_dir),
                    str(coarse_library_path),
                )
                legacy = np.load(coarse_path).copy()
                coarse_path.unlink()
                elapsed_conv = _run_one_decomposition(
                    waveletDecompositionConv,
                    movie,
                    phase_index,
                    sigmas,
                    str(case_dir),
                    n_orientations=n_orientations,
                    phase_offsets=PHASES,
                    kernel_cache_path=coarse_kernel_cache_path,
                    frame_chunk_size=2,
                )
                convolution = np.load(coarse_path).copy()
                coarse_path.unlink()
                record = _record(case, stimulus_name, "coarse", phase_index, legacy, convolution, elapsed_legacy, elapsed_conv)
                records.append(record)
                if worst is None or record["max_abs_diff"] > worst["record"]["max_abs_diff"]:
                    worst = {"record": record, "legacy": legacy, "convolution": convolution}

                suffix = "_r" if phase_index == 0 else "_i"
                full_path = case_dir / f"dwt_videodata2{suffix}.npy"
                elapsed_legacy = _run_one_decomposition(
                    waveletDecompositionFull,
                    movie,
                    phase_index,
                    sigmas,
                    frequencies,
                    str(case_dir),
                    str(full_library_path),
                    library_sigmas=sigmas,
                    output_format="npy",
                )
                legacy = np.load(full_path).copy()
                full_path.unlink()
                elapsed_conv = _run_one_decomposition(
                    waveletDecompositionFullConv,
                    movie,
                    phase_index,
                    sigmas,
                    frequencies,
                    str(case_dir),
                    n_orientations=n_orientations,
                    phase_offsets=PHASES,
                    output_format="npy",
                    kernel_cache_path=fine_kernel_cache_path,
                    frame_chunk_size=2,
                )
                convolution = np.load(full_path).copy()
                full_path.unlink()
                record = _record(case, stimulus_name, "full", phase_index, legacy, convolution, elapsed_legacy, elapsed_conv)
                records.append(record)
                if record["max_abs_diff"] > worst["record"]["max_abs_diff"]:
                    worst = {"record": record, "legacy": legacy, "convolution": convolution}

    summary = {
        "num_cases": len(cases),
        "num_stimuli": len(stimuli),
        "num_comparisons": len(records),
        "max_abs_diff": max(record["max_abs_diff"] for record in records),
        "mean_abs_diff_mean": float(np.mean([record["mean_abs_diff"] for record in records])),
        "max_relative_diff": max(record["relative_max_diff"] for record in records),
        "worst_record": worst["record"] if worst else None,
        "cases": cases,
        "stimuli": stimuli,
        "convolution_kernel_cache": "enabled",
        "tolerance_note": "Legacy filters are stored as float16; differences near 1e-4 are expected.",
    }

    with (output_dir / "stress_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "stress_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    _save_worst_artifacts(output_dir, worst)
    (output_dir / "README.md").write_text(
        "# Waven Convolution Backend Stress Test\n\n"
        "This folder contains `stress_summary.json`, `stress_records.csv`, and "
        "the arrays/heatmap for the worst observed legacy-vs-convolution mismatch.\n",
        encoding="utf-8",
    )

    if not keep_work:
        shutil.rmtree(work_root, ignore_errors=True)
    return output_dir, summary


def main():
    """Parse arguments and run the stress suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=None, help="Directory for saved stress-test artifacts.")
    parser.add_argument("--keep-work", action="store_true", help="Keep temporary per-case libraries and outputs.")
    args = parser.parse_args()
    output_dir, summary = run(args.output_dir, keep_work=args.keep_work)
    print(f"Saved stress-test artifacts to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
