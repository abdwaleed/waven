"""Regression tests for the opt-in, later-analysis RAM cache."""

from __future__ import annotations

import numpy as np

from waven.storage import array_store


def setup_function(_function):
    """Keep process-local cache state from leaking between tests."""
    array_store.clear_ram_acceleration_cache()


def teardown_function(_function):
    """Release test arrays before the next test allocates its fixture."""
    array_store.clear_ram_acceleration_cache()


def test_ram_acceleration_reuses_one_immutable_resident_copy(tmp_path, monkeypatch):
    """A safe later-analysis input is copied once and reused by path."""
    path = tmp_path / "small_analysis_input.npy"
    expected = np.arange(24, dtype=np.float32).reshape(4, 6)
    np.save(path, expected)
    monkeypatch.setattr(array_store, "_ram_acceleration_enabled", lambda: True)
    monkeypatch.setattr(
        array_store,
        "_ram_cache_limits",
        lambda _required: (64 * 1024**2, True),
    )

    first = array_store.load_array_with_ram_acceleration(str(path), cache_label="test input")
    second = array_store.load_array_with_ram_acceleration(str(path), cache_label="test input")

    assert isinstance(first, np.ndarray)
    assert not isinstance(first, np.memmap)
    assert first is second
    assert not first.flags.writeable
    np.testing.assert_array_equal(first, expected)
    assert array_store.ram_acceleration_cache_stats() == {
        "entries": 1,
        "bytes": expected.nbytes,
    }


def test_ram_acceleration_keeps_an_oversized_input_disk_backed(tmp_path, monkeypatch):
    """The cache budget is checked before copying a disk-backed NPY input."""
    path = tmp_path / "oversized_analysis_input.npy"
    np.save(path, np.arange(64, dtype=np.float32))
    monkeypatch.setattr(array_store, "_ram_acceleration_enabled", lambda: True)
    monkeypatch.setattr(array_store, "_ram_cache_limits", lambda _required: (1, True))

    loaded = array_store.load_array_with_ram_acceleration(str(path), cache_label="oversized test input")

    assert isinstance(loaded, np.memmap)
    assert array_store.ram_acceleration_cache_stats() == {"entries": 0, "bytes": 0}
    del loaded


def test_ram_acceleration_disabled_retains_existing_disk_backed_loading(tmp_path, monkeypatch):
    """Default-off behavior remains an ordinary memory-mapped load."""
    path = tmp_path / "default_analysis_input.npy"
    np.save(path, np.arange(8, dtype=np.float32))
    monkeypatch.setattr(array_store, "_ram_acceleration_enabled", lambda: False)

    loaded = array_store.load_array_with_ram_acceleration(str(path))

    assert isinstance(loaded, np.memmap)
    assert array_store.ram_acceleration_cache_stats() == {"entries": 0, "bytes": 0}
    del loaded
