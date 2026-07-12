"""Strict project directory layout and lightweight artifact discovery."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v")
ARRAY_SUFFIXES = (".npy", ".zarr")
REFERENCE_NAME = ".waven_reference.json"


@dataclass(frozen=True)
class WavenProjectLayout:
    """Resolved folder layout for one Waven project root.

    Args:
        root: Project root. All generated artifacts are stored below this
            directory using stable subfolders.
    """

    root: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "WavenProjectLayout":
        """Create a layout from a root folder path."""
        return cls(Path(root).expanduser())

    @property
    def input_dir(self) -> Path:
        """Folder for source or referenced user-provided inputs."""
        return self.root / "input"

    @property
    def raw_data_dir(self) -> Path:
        """Folder for raw two-photon/ephys acquisition data."""
        return self.input_dir / "raw_data"

    @property
    def stimulus_movie_dir(self) -> Path:
        """Folder containing exactly one stimulus movie when possible."""
        return self.input_dir / "stimulus_movie"

    @property
    def neural_cache_dir(self) -> Path:
        """Folder containing aligned ``spikes`` and ``pos`` caches."""
        return self.input_dir / "neural_cache"

    @property
    def cache_dir(self) -> Path:
        """Folder for reusable generated intermediates."""
        return self.root / "cache"

    @property
    def gabor_dir(self) -> Path:
        """Folder for legacy Gabor libraries and convolution kernel caches."""
        return self.cache_dir / "gabor"

    @property
    def coarse_gabor_dir(self) -> Path:
        """Folder for coarse RF Gabor libraries."""
        return self.gabor_dir / "coarse"

    @property
    def fine_gabor_dir(self) -> Path:
        """Folder for full-model Gabor libraries."""
        return self.gabor_dir / "full"

    @property
    def gabor_kernel_dir(self) -> Path:
        """Folder for compact convolution kernel caches."""
        return self.gabor_dir / "kernels"

    @property
    def wavelet_dir(self) -> Path:
        """Folder for stimulus wavelet intermediates."""
        return self.cache_dir / "wavelets"

    @property
    def coarse_wavelet_dir(self) -> Path:
        """Folder for coarse RF wavelet caches."""
        return self.wavelet_dir / "coarse"

    @property
    def full_wavelet_dir(self) -> Path:
        """Folder for full-model wavelet caches."""
        return self.wavelet_dir / "full"

    @property
    def output_dir(self) -> Path:
        """Folder for analysis outputs and exported figures."""
        return self.root / "output"

    @property
    def plots_dir(self) -> Path:
        """Folder for plot caches and exported plots."""
        return self.output_dir / "plots"

    @property
    def recovery_dir(self) -> Path:
        """Folder for resumable task checkpoints."""
        return self.output_dir / "recovery_cache"

    @property
    def model_dir(self) -> Path:
        """Folder for model outputs."""
        return self.output_dir / "models"

    def ensure(self) -> None:
        """Create the standard project folders if they do not exist."""
        for directory in (
            self.raw_data_dir,
            self.stimulus_movie_dir,
            self.neural_cache_dir,
            self.coarse_gabor_dir,
            self.fine_gabor_dir,
            self.gabor_kernel_dir,
            self.coarse_wavelet_dir,
            self.full_wavelet_dir,
            self.plots_dir,
            self.recovery_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def dir_for_field(self, field_name: str) -> Path:
        """Return the conventional folder backing a GUI path field."""
        mapping = {
            "Dir": self.raw_data_dir,
            "Save Path": self.fine_gabor_dir,
            "Movie Path": self.stimulus_movie_dir,
            "Path Directory": self.coarse_wavelet_dir,
            "Library Path": self.fine_gabor_dir,
            "Coarse Library Path": self.coarse_gabor_dir,
            "Fine Library Path": self.fine_gabor_dir,
            "Spks Path": self.neural_cache_dir,
            "Full Model Wavelet Path": self.full_wavelet_dir,
            "Full Model Save Path": self.model_dir,
            "Plot Cache Path": self.plots_dir,
            "Recovery Cache Directory": self.recovery_dir,
        }
        return mapping[field_name]


def write_reference(folder: str | Path, target: str | Path, kind: str) -> Path:
    """Store a lightweight reference to an external artifact or folder.

    The project layout remains strict, but a user can keep large source data
    elsewhere. The resolver treats this JSON file as the canonical pointer.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    target_path = Path(target).expanduser()
    try:
        relative_target = os.path.relpath(str(target_path.resolve()), str(folder.resolve()))
    except OSError:
        relative_target = None
    payload = {"kind": kind, "target": str(target_path), "relative_target": relative_target}
    reference_path = folder / REFERENCE_NAME
    reference_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return reference_path


def read_reference(folder: str | Path, kind: Optional[str] = None) -> Optional[Path]:
    """Return the external target referenced by ``folder`` when present."""
    reference_path = Path(folder) / REFERENCE_NAME
    if not reference_path.exists():
        return None
    try:
        payload = json.loads(reference_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if kind is not None and payload.get("kind") not in {kind, None}:
        return None
    relative_target = payload.get("relative_target")
    if relative_target:
        candidate = Path(folder) / str(relative_target)
        if candidate.exists():
            return candidate
    target = payload.get("target")
    return Path(target).expanduser() if target else None


def resolve_folder_reference(folder: str | Path, kind: Optional[str] = None) -> Path:
    """Resolve a layout folder to either itself or its referenced target."""
    folder = Path(folder)
    target = read_reference(folder, kind)
    return target if target is not None and target.exists() else folder


def iter_artifacts(folder: str | Path, suffixes: Iterable[str]) -> Iterable[Path]:
    """Yield direct child artifacts matching any suffix, including references."""
    resolved = resolve_folder_reference(folder)
    if not resolved.exists():
        return
    suffixes = tuple(suffix.lower() for suffix in suffixes)
    for child in sorted(resolved.iterdir()):
        child_suffix = child.suffix.lower()
        if child_suffix in suffixes:
            yield child


def find_single_artifact(folder: str | Path, suffixes: Iterable[str], label: str) -> Path:
    """Find one artifact inside ``folder`` or raise a clear ambiguity error."""
    matches = list(iter_artifacts(folder, suffixes))
    if not matches:
        resolved = resolve_folder_reference(folder)
        raise FileNotFoundError(f"Could not find {label} in folder: {resolved}")
    if len(matches) > 1:
        joined = "\n".join(f"  - {path}" for path in matches[:10])
        raise ValueError(
            f"Found multiple {label} candidates in {resolve_folder_reference(folder)}.\n"
            f"Keep one file in this folder or move extras elsewhere:\n{joined}"
        )
    return matches[0]


def find_stimulus_movie(folder: str | Path) -> Path:
    """Return the only movie-like file in a stimulus movie folder."""
    return find_single_artifact(folder, VIDEO_SUFFIXES, "stimulus movie")


def find_named_array(folder: str | Path, stem: str) -> Optional[Path]:
    """Return ``stem.npy`` or ``stem.zarr`` from a folder when present."""
    resolved = resolve_folder_reference(folder)
    for suffix in ARRAY_SUFFIXES:
        candidate = resolved / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def conventional_gabor_path(folder: str | Path, kind: str, output_format: str = "npy") -> Path:
    """Return the conventional Gabor library path for ``kind``."""
    suffix = ".zarr" if str(output_format).lower() == "zarr" else ".npy"
    return Path(folder) / f"gabor_library_{kind}{suffix}"


def conventional_downsample_path(folder: str | Path, scale: str, output_format: str, percent: int) -> Path:
    """Return the conventional downsampled movie cache path."""
    suffix = ".zarr" if str(output_format).lower() == "zarr" else ".npy"
    return Path(folder) / f"stimulus_{scale}_downsampled_p{int(percent):03d}{suffix}"


__all__ = [
    "ARRAY_SUFFIXES",
    "VIDEO_SUFFIXES",
    "WavenProjectLayout",
    "conventional_downsample_path",
    "conventional_gabor_path",
    "find_named_array",
    "find_single_artifact",
    "find_stimulus_movie",
    "read_reference",
    "resolve_folder_reference",
    "write_reference",
]
