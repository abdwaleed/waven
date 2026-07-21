"""Regression coverage for neural-cache metadata and Zarr trial selection."""

from __future__ import annotations

import json

import numpy as np
import pytest

from waven.storage.array_store import read_first_axis_indices
from waven.storage.neural_cache import (
    load_unit_ids,
    load_unit_info,
    save_aligned_neural_cache,
)


def test_read_first_axis_indices_matches_numpy_advanced_selection():
    source = np.arange(4 * 5 * 3).reshape(4, 5, 3)

    selected = read_first_axis_indices(source, [3, 1], slice(1, 4), slice(1, 3))

    np.testing.assert_array_equal(selected, source[[3, 1], 1:4, 1:3])


def test_read_first_axis_indices_has_basic_indexing_fallback():
    source_values = np.arange(4 * 5 * 3).reshape(4, 5, 3)

    class BasicIndexOnlyArray:
        def __getitem__(self, selection):
            if isinstance(selection[0], np.ndarray):
                raise IndexError("advanced first-axis indexing is unsupported")
            return source_values[selection]

    selected = read_first_axis_indices(
        BasicIndexOnlyArray(),
        [3, 1],
        slice(1, 4),
        slice(1, 3),
    )

    np.testing.assert_array_equal(selected, source_values[[3, 1], 1:4, 1:3])


def test_read_first_axis_indices_supports_disk_backed_zarr(tmp_path):
    zarr = pytest.importorskip("zarr")
    source_values = np.arange(4 * 5 * 3).reshape(4, 5, 3)
    source = zarr.open(
        str(tmp_path / "spikes.zarr"),
        mode="w",
        shape=source_values.shape,
        chunks=(1, 3, 3),
        dtype=source_values.dtype,
    )
    source[:] = source_values

    selected = read_first_axis_indices(source, [3, 1], slice(1, 4), slice(1, 3))

    np.testing.assert_array_equal(selected, source_values[[3, 1], 1:4, 1:3])


def test_stale_optional_unit_sidecars_do_not_block_analysis(tmp_path):
    (tmp_path / "unit_ids.json").write_text(json.dumps(["old-a", "old-b"]), encoding="utf-8")
    (tmp_path / "unit_info.json").write_text(
        json.dumps({"version": 1, "units": [{"unit": "old"}]}),
        encoding="utf-8",
    )

    with pytest.warns(RuntimeWarning, match="Ignoring stale unit_ids"):
        assert load_unit_ids(tmp_path, n_neurons=3) is None
    with pytest.warns(RuntimeWarning, match="Ignoring stale unit_info"):
        assert load_unit_info(tmp_path, n_neurons=3) is None


def test_saving_without_optional_metadata_clears_old_sidecars(tmp_path):
    positions = np.zeros((2, 3), dtype=np.float32)
    spikes = np.zeros((2, 4, 2), dtype=np.float32)
    save_aligned_neural_cache(
        positions,
        spikes,
        tmp_path,
        unit_ids=["unit-a", "unit-b"],
        unit_info=[{"unit": "unit-a"}, {"unit": "unit-b"}],
    )

    save_aligned_neural_cache(positions, spikes, tmp_path)

    assert not (tmp_path / "unit_ids.json").exists()
    assert not (tmp_path / "unit_info.json").exists()
