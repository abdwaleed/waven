"""Storage helpers for aligned neural response and position caches."""
from __future__ import annotations

from pathlib import Path
import json
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .array_store import load_array, load_array_with_memory_fallback

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
    spike_candidates = [
        path
        for path in array_paths
        if ("spike" in path.stem.lower() or "spks" in path.stem.lower())
        and "count" not in path.stem.lower()
    ]
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
    loader = load_array_with_memory_fallback if mmap_mode is None else load_array
    if mmap_mode is None:
        spikes = loader(str(spikes_path))
        neuron_pos = loader(str(pos_path))
    else:
        spikes = loader(str(spikes_path), mmap_mode=mmap_mode)
        neuron_pos = loader(str(pos_path), mmap_mode=mmap_mode)
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
    unit_info: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Path]:
    """Persist aligned ``pos``/``spikes`` caches and optional unit metadata."""
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
    n_neurons = int(np.asarray(neuron_pos).shape[0])
    if unit_ids is not None:
        unit_ids_path = output_dir / "unit_ids.json"
        values = [value.item() if isinstance(value, np.generic) else value for value in unit_ids]
        if len(values) != n_neurons:
            raise ValueError(
                "unit_ids must contain one identifier for every neuron in the neural cache."
            )
        with unit_ids_path.open("w", encoding="utf-8") as handle:
            json.dump(values, handle, indent=2, default=str)
        result["unit_ids"] = unit_ids_path
    else:
        # A cache rebuilt from sources without acquisition identifiers must not
        # inherit labels from a previous experiment with a different unit set.
        (output_dir / "unit_ids.json").unlink(missing_ok=True)
    if unit_info is not None:
        unit_info_path = output_dir / "unit_info.json"
        values = list(unit_info)
        if len(values) != n_neurons:
            raise ValueError(
                "unit_info must contain one entry for every neuron in the neural cache."
            )
        with unit_info_path.open("w", encoding="utf-8") as handle:
            json.dump({"version": 1, "units": values}, handle, indent=2, default=str)
        result["unit_info"] = unit_info_path
    else:
        (output_dir / "unit_info.json").unlink(missing_ok=True)
    return result


def load_unit_ids(directory: Path, n_neurons: Optional[int] = None) -> Optional[np.ndarray]:
    """Load preserved acquisition unit identifiers from a neural-cache folder.

    Args:
        directory: Folder containing ``unit_ids.json``.
        n_neurons: Optional expected number of identifiers.

    Returns:
        One identifier per neuron, or ``None`` when the optional sidecar is
        absent or belongs to a different neural cache.
    """
    path = Path(directory) / "unit_ids.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Ignoring unreadable unit_ids.json ({exc}). Unit labels will fall back to indices.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not isinstance(payload, list):
        warnings.warn(
            "Ignoring invalid unit_ids.json: expected a list of identifiers. "
            "Unit labels will fall back to indices.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    values = np.asarray(payload, dtype=object)
    if n_neurons is not None and values.size != int(n_neurons):
        warnings.warn(
            f"Ignoring stale unit_ids.json: it contains {values.size} IDs but the active "
            f"neural cache has {n_neurons} neurons. Unit labels will fall back to indices.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return values


def load_unit_info(directory: Path, n_neurons: Optional[int] = None) -> Optional[List[Dict[str, Any]]]:
    """Load EPHYS unit metadata preserved beside an aligned neural cache.

    Older caches do not have this optional sidecar and return ``None``.  The
    JSON object wrapper leaves room for future metadata without changing the
    one-record-per-neuron contract used by the GUI.
    """
    path = Path(directory) / "unit_info.json"
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(
            f"Ignoring unreadable unit_info.json ({exc}).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    values = payload.get("units") if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        warnings.warn(
            "Ignoring invalid unit_info.json: expected a list of per-neuron metadata objects.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if n_neurons is not None and len(values) != int(n_neurons):
        warnings.warn(
            f"Ignoring stale unit_info.json: it contains {len(values)} records but the active "
            f"neural cache has {n_neurons} neurons.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return values


__all__ = [
    "SUPPORTED_NEURAL_CACHE_FORMATS",
    "find_neural_cache_pair",
    "load_neural_cache_pair",
    "load_unit_info",
    "load_unit_ids",
    "neural_cache_path",
    "normalize_neural_cache_format",
    "save_aligned_neural_cache",
]
