"""
Helper to load large arrays preferring zarr datasets when available.
Provides load_array(path, ...) which will try to open a .zarr dataset (or sibling .zarr)
and fall back to numpy.load(..., mmap_mode=...).

Returns either a zarr Array/Group or a numpy array/memmap depending on what was loaded.
"""
from __future__ import annotations

import os
import logging
import threading
from collections import OrderedDict
from typing import Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# This cache is deliberately opt-in and is used only by later analysis calls.
# Cache preparation always goes through ``load_array`` directly, so it keeps
# its established bounded, disk-backed behaviour.  Holding the arrays here
# (rather than relying only on the operating-system file cache) makes repeated
# Zarr reads by model fitting and PSTH/STA deterministic and substantially
# faster when the source is modest enough to fit safely in RAM.
_analysis_ram_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
_analysis_ram_cache_bytes = 0
_analysis_ram_cache_lock = threading.RLock()
_ANALYSIS_RAM_CACHE_FRACTION = 0.20

try:
    import zarr
    HAS_ZARR = True
except Exception:
    zarr = None  # type: ignore
    HAS_ZARR = False


def _is_zarr_dir(path: str) -> bool:
    """Return True if path looks like a zarr store (directory containing .zarray/.zgroup)."""
    try:
        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, ".zarray")):
                return True
            if os.path.exists(os.path.join(path, ".zgroup")):
                return True
            if os.path.exists(os.path.join(path, "zarr.json")):
                return True
        return False
    except Exception:
        return False


def load_array(
    path: str,
    *,
    mmap_mode: Optional[str] = "r",
    allow_pickle: bool = False,
    prefer_zarr: bool = True,
    raise_on_error: bool = True,
    zarr_kwargs: Optional[dict] = None,
) -> Any:
    """
    Load an array-like object from disk.

    Behavior:
    - If prefer_zarr and zarr is available and path points to a zarr store (or a sibling .zarr exists), open with zarr.open(..., mode='r').
    - Otherwise, try np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle).

    Notes:
    - If allow_pickle is True, we avoid trying zarr because pickled numpy objects are not representable in zarr.
    - Returns a zarr Array/Group or numpy array/memmap depending on what was loaded.
    - On error, either re-raises (raise_on_error=True) or returns None.
    """
    zarr_kwargs = {} if zarr_kwargs is None else dict(zarr_kwargs)
    path = str(path)

    # If caller requested pickles, prefer numpy loader
    if allow_pickle:
        logger.debug("allow_pickle=True -> using numpy.load for %s", path)
        try:
            return np.load(path, allow_pickle=True, mmap_mode=mmap_mode)
        except Exception as e:
            logger.exception("np.load failed for %s", path)
            if raise_on_error:
                raise
            return None

    # First: if the path itself is a zarr store
    if prefer_zarr and HAS_ZARR:
        try:
            if _is_zarr_dir(path):
                logger.debug("Opening zarr store at %s", path)
                return zarr.open(path, mode="r", **zarr_kwargs)

            # If path ends with .npy but a sibling .zarr exists, prefer it
            if path.endswith(".npy"):
                sibling = path[:-4] + ".zarr"
                if _is_zarr_dir(sibling):
                    logger.debug("Found sibling zarr %s for %s", sibling, path)
                    return zarr.open(sibling, mode="r", **zarr_kwargs)

            # Also accept explicit .zarr filename
            if path.endswith(".zarr") and os.path.exists(path):
                logger.debug("Opening explicit zarr path %s", path)
                return zarr.open(path, mode="r", **zarr_kwargs)

        except Exception:
            logger.exception("Failed opening as zarr, will try numpy.load for %s", path)

    # Fallback to numpy load with mmap when requested.  Do not silently retry
    # a failed mapped load as an eager load: that turns a recoverable mapping
    # problem into a potentially system-freezing allocation.
    try:
        logger.debug("Using numpy.load for %s (mmap_mode=%r)", path, mmap_mode)
        return np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
    except Exception:
        logger.exception("np.load failed for %s", path)
        if raise_on_error:
            raise
        return None


def load_array_with_memory_fallback(
    path: str,
    *,
    allow_pickle: bool = False,
    prefer_zarr: bool = True,
    zarr_kwargs: Optional[dict] = None,
) -> Any:
    """Load eagerly when possible, falling back to a read-only disk view.

    This is for callers that benefit from ordinary in-memory NumPy arrays on
    small inputs but must not crash a workstation when an old cache is large.
    Zarr stores are already opened lazily by :func:`load_array`; NPY files are
    retried with ``mmap_mode='r'`` only after an allocation-related failure.
    """
    try:
        return load_array(
            path,
            mmap_mode=None,
            allow_pickle=allow_pickle,
            prefer_zarr=prefer_zarr,
            zarr_kwargs=zarr_kwargs,
        )
    except (MemoryError, OSError) as exc:
        logger.warning(
            "Could not load %s fully into RAM (%s); using a read-only disk-backed view.",
            path,
            exc,
        )
        return load_array(
            path,
            mmap_mode="r",
            allow_pickle=allow_pickle,
            prefer_zarr=prefer_zarr,
            zarr_kwargs=zarr_kwargs,
        )


def _ram_acceleration_enabled() -> bool:
    """Return whether the optional later-analysis RAM cache is enabled."""
    # Keep ``array_store`` importable in small utility/test environments where
    # the optional runtime stack (and therefore Torch) is unavailable.
    try:
        from ..runtime.performance import enabled_feature

        return enabled_feature("RAM_ACCELERATION_CACHE", default=False)
    except Exception:
        return False


def _array_nbytes(array: Any) -> Optional[int]:
    """Return the fully materialized size of an array-like source, if known."""
    try:
        shape = tuple(int(size) for size in array.shape)
        dtype = np.dtype(array.dtype)
        return int(np.prod(shape, dtype=np.int64)) * int(dtype.itemsize)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None


def _ram_cache_limits(required_bytes: int) -> Optional[tuple[int, bool]]:
    """Return the live cache budget and headroom decision, if measurable."""
    try:
        from ..runtime.performance import available_ram_bytes, has_enough_ram

        available_bytes = int(available_ram_bytes())
        return (
            int(available_bytes * _ANALYSIS_RAM_CACHE_FRACTION),
            bool(has_enough_ram(required_bytes, safety_margin=1.50, os_reserve_fraction=0.40)),
        )
    except Exception:
        return None


def _analysis_cache_key(path: str) -> str:
    """Canonicalize an artifact path for the process-local analysis cache."""
    return os.path.normcase(os.path.realpath(os.fspath(path)))


def ram_acceleration_cache_stats() -> dict[str, int]:
    """Return the current process-local RAM-cache footprint for UI reporting."""
    with _analysis_ram_cache_lock:
        return {
            "entries": len(_analysis_ram_cache),
            "bytes": int(_analysis_ram_cache_bytes),
        }


def clear_ram_acceleration_cache() -> dict[str, int]:
    """Release references held by the optional later-analysis RAM cache.

    Callers may still hold an array returned by
    :func:`load_array_with_ram_acceleration`; in that case Python releases the
    corresponding memory once those callers finish.  This function nevertheless
    makes the cache itself immediately eligible for collection.
    """
    global _analysis_ram_cache_bytes
    with _analysis_ram_cache_lock:
        released = {
            "entries": len(_analysis_ram_cache),
            "bytes": int(_analysis_ram_cache_bytes),
        }
        _analysis_ram_cache.clear()
        _analysis_ram_cache_bytes = 0
    return released


def load_array_with_ram_acceleration(
    path: str,
    *,
    mmap_mode: Optional[str] = "r",
    allow_pickle: bool = False,
    prefer_zarr: bool = True,
    raise_on_error: bool = True,
    zarr_kwargs: Optional[dict] = None,
    cache_label: str = "analysis array",
) -> Any:
    """Load a later-analysis array with an optional guarded RAM copy.

    The default remains a disk-backed NumPy/Zarr object.  When the Advanced
    Session Config RAM acceleration switch is enabled, a source is copied once
    into an immutable process-local cache only when it fits a conservative
    cumulative budget (20% of currently available memory) and retains desktop
    headroom.  Any eligibility or allocation failure returns the original
    disk-backed source, so enabling this option never makes an analysis fail.

    This function is intentionally *not* used by cache-preparation code.  It
    is for repeated reads in later analysis stages such as Run Model, Run Full
    Model, and PSTH/STA.
    """
    global _analysis_ram_cache_bytes
    source = load_array(
        path,
        mmap_mode=mmap_mode,
        allow_pickle=allow_pickle,
        prefer_zarr=prefer_zarr,
        raise_on_error=raise_on_error,
        zarr_kwargs=zarr_kwargs,
    )
    if source is None or not _ram_acceleration_enabled():
        return source

    required_bytes = _array_nbytes(source)
    if required_bytes is None:
        print(f"[RAM cache] Keeping {cache_label} disk-backed: its RAM size is unknown.")
        return source
    if required_bytes == 0:
        return source

    limits = _ram_cache_limits(required_bytes)
    if limits is None:
        logger.warning("RAM acceleration check unavailable for %s", path)
        return source
    cache_budget, has_headroom = limits

    key = _analysis_cache_key(path)
    with _analysis_ram_cache_lock:
        cached = _analysis_ram_cache.get(key)
        if cached is not None:
            _analysis_ram_cache.move_to_end(key)
            print(f"[RAM cache] Reusing {cache_label} ({required_bytes / 1024**3:.2f} GiB).")
            return cached
        projected_bytes = _analysis_ram_cache_bytes + required_bytes

    if required_bytes > cache_budget or projected_bytes > cache_budget:
        print(
            f"[RAM cache] Keeping {cache_label} disk-backed "
            f"({required_bytes / 1024**3:.2f} GiB exceeds the safe "
            f"{cache_budget / 1024**3:.2f} GiB session budget)."
        )
        return source
    if not has_headroom:
        print(
            f"[RAM cache] Keeping {cache_label} disk-backed "
            f"({required_bytes / 1024**3:.2f} GiB would leave insufficient RAM headroom)."
        )
        return source

    try:
        # ``np.array(..., copy=True)`` forces a real resident copy for both
        # Zarr arrays and NPY memmaps.  The immutable flag prevents a later
        # analysis routine from silently changing the shared cache.
        resident = np.array(source, copy=True, subok=False)
        resident.setflags(write=False)
    except Exception as exc:
        print(f"[RAM cache] Could not cache {cache_label}; continuing disk-backed ({exc}).")
        return source

    with _analysis_ram_cache_lock:
        # A second GUI worker could have filled this entry while the copy was
        # in progress.  Prefer the first completed copy and drop ours.
        cached = _analysis_ram_cache.get(key)
        if cached is not None:
            _analysis_ram_cache.move_to_end(key)
            return cached
        # Recheck against the live budget after the potentially lengthy copy.
        # If another cache entry appeared, return the normal source rather
        # than growing past the promised session ceiling.
        live_limits = _ram_cache_limits(0)
        live_budget = live_limits[0] if live_limits is not None else 0
        if _analysis_ram_cache_bytes + resident.nbytes > live_budget:
            print(f"[RAM cache] Cache budget changed while loading {cache_label}; continuing disk-backed.")
            return source
        _analysis_ram_cache[key] = resident
        _analysis_ram_cache_bytes += int(resident.nbytes)
        _analysis_ram_cache.move_to_end(key)
    print(
        f"[RAM cache] Cached {cache_label}: {resident.nbytes / 1024**3:.2f} GiB "
        f"({ram_acceleration_cache_stats()['entries']} session array(s))."
    )
    return resident


__all__ = [
    "clear_ram_acceleration_cache",
    "load_array",
    "load_array_with_memory_fallback",
    "load_array_with_ram_acceleration",
    "ram_acceleration_cache_stats",
]

