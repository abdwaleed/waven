"""Storage helpers for aligned neural response and position caches."""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .array_store import load_array

SUPPORTED_NEURAL_CACHE_FORMATS = {"npy", "zarr"}


def normalize_neural_cache_format(value: str) -> str:
    """Return a supported neural-cache format name."""
    fmt = str(value or "npy").strip().lower().lstrip(".")
    if fmt not in SUPPORTED_NEURAL_CACHE_FORMATS:
        raise ValueError(
            f"Unsupported neural cache format {value!r}. "
            "Choose 'npy' or 'zarr'."
        )
    return fmt


def neural_cache_path(directory: Path, stem: str, output_format: str) -> Path:
    """Return the cache path for ``stem`` in ``output_format``."""
    fmt = normalize_neural_cache_format(output_format)
    return Path(directory) / f"{stem}.{fmt}"


def _candidate_paths(directory: Path, stem: str, preferred_suffix: Optional[str] = None):
    """Yield possible neural-cache paths, putting the preferred suffix first."""
    suffixes = []
    if preferred_suffix:
        suffixes.append(preferred_suffix.lower().lstrip("."))
    suffixes.extend(fmt for fmt in ("npy", "zarr") if fmt not in suffixes)
    for suffix in suffixes:
        yield Path(directory) / f"{stem}.{suffix}"


def find_neural_cache_pair(
    directory: Path,
    spks_path: Optional[Path] = None,
) -> Optional[Tuple[Path, Path]]:
    """Find matching ``spikes`` and ``pos`` cache paths in either NPY or Zarr."""
    if spks_path is not None:
        spikes_path = Path(spks_path)
        if not spikes_path.exists():
            return None
        preferred = spikes_path.suffix
        for pos_path in _candidate_paths(spikes_path.parent, "pos", preferred):
            if pos_path.exists():
                return spikes_path, pos_path
        return None

    directory = Path(directory)
    for fmt in ("npy", "zarr"):
        spikes_path = directory / f"spikes.{fmt}"
        pos_path = directory / f"pos.{fmt}"
        if spikes_path.exists() and pos_path.exists():
            return spikes_path, pos_path
    for spikes_path in _candidate_paths(directory, "spikes"):
        if not spikes_path.exists():
            continue
        for pos_path in _candidate_paths(directory, "pos", spikes_path.suffix):
            if pos_path.exists():
                return spikes_path, pos_path
    if not directory.exists():
        return None
    array_paths = [
        path
        for path in sorted(directory.iterdir())
        if path.suffix.lower() in {".npy", ".zarr"}
    ]
    spike_candidates = [path for path in array_paths if "spike" in path.stem.lower() or "spks" in path.stem.lower()]
    pos_candidates = [path for path in array_paths if "pos" in path.stem.lower() or "position" in path.stem.lower()]
    if len(spike_candidates) == 1 and len(pos_candidates) == 1:
        return spike_candidates[0], pos_candidates[0]
    return None


def load_neural_cache_pair(
    directory: Path,
    spks_path: Optional[Path] = None,
    mmap_mode: Optional[str] = "r",
) -> Tuple[np.ndarray, np.ndarray, Path, Path]:
    """Load a ``spikes``/``pos`` cache pair from NPY or Zarr."""
    pair = find_neural_cache_pair(directory, spks_path)
    if pair is None:
        target = Path(spks_path) if spks_path is not None else Path(directory)
        raise FileNotFoundError(
            "Could not find a matching aligned neural cache pair "
            f"(spikes.npy/spikes.zarr and pos.npy/pos.zarr) at {target}."
        )
    spikes_path, pos_path = pair
    spikes = load_array(str(spikes_path), mmap_mode=mmap_mode)
    neuron_pos = load_array(str(pos_path), mmap_mode=mmap_mode)
    return spikes, neuron_pos, spikes_path, pos_path


def _save_zarr_array(path: Path, array: np.ndarray) -> None:
    """Write one dense array to a Zarr store."""
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as exc:
        raise ImportError(
            "Saving aligned neural caches as Zarr requires 'zarr' and "
            "'numcodecs'. Install project requirements or choose NPY."
        ) from exc

    arr = np.asarray(array)
    chunks = tuple(max(1, min(int(dim), 256)) for dim in arr.shape)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    try:
        store = zarr.open(
            str(path),
            mode="w",
            shape=arr.shape,
            chunks=chunks,
            dtype=arr.dtype,
            compressor=compressor,
        )
    except TypeError:
        store = zarr.open(
            str(path),
            mode="w",
            shape=arr.shape,
            chunks=chunks,
            dtype=arr.dtype,
            compressors=[compressor],
        )
    store[:] = arr


def save_aligned_neural_cache(
    neuron_pos: np.ndarray,
    spikes: np.ndarray,
    save_dir: Optional[Path],
    output_format: str = "npy",
    unit_ids: Optional[Sequence[object]] = None,
) -> Dict[str, Path]:
    """Persist aligned ``pos`` and ``spikes`` caches in NPY or Zarr format."""
    fmt = normalize_neural_cache_format(output_format)
    output_dir = Path(".") if save_dir is None else Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pos_path = neural_cache_path(output_dir, "pos", fmt)
    spikes_path = neural_cache_path(output_dir, "spikes", fmt)

    if fmt == "npy":
        np.save(pos_path, neuron_pos)
        np.save(spikes_path, spikes)
    else:
        _save_zarr_array(pos_path, neuron_pos)
        _save_zarr_array(spikes_path, spikes)
    result = {"pos": pos_path, "spikes": spikes_path}
    if unit_ids is not None:
        unit_ids_path = output_dir / "unit_ids.json"
        values = [value.item() if isinstance(value, np.generic) else value for value in unit_ids]
        with unit_ids_path.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, default=str)
        result["unit_ids"] = unit_ids_path
    return result


def load_unit_ids(directory: Path, n_neurons: Optional[int] = None) -> Optional[np.ndarray]:
    """Load preserved acquisition unit identifiers from a neural-cache folder.

    Args:
        directory: Folder containing ``unit_ids.json``.
        n_neurons: Optional expected number of identifiers.

    Returns:
        One identifier per neuron, or ``None`` for an older cache without IDs.
    """
    path = Path(directory) / "unit_ids.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        values = np.asarray(json.load(handle), dtype=object)
    if n_neurons is not None and values.size != int(n_neurons):
        raise ValueError(
            f"unit_ids.json contains {values.size} IDs but the neural cache has {n_neurons} neurons."
        )
    return values


__all__ = [
    "SUPPORTED_NEURAL_CACHE_FORMATS",
    "find_neural_cache_pair",
    "load_neural_cache_pair",
    "load_unit_ids",
    "neural_cache_path",
    "normalize_neural_cache_format",
    "save_aligned_neural_cache",
]
