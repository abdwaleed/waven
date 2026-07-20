"""Contracts for non-optional performance paths."""

import pytest

from waven.runtime.performance import enabled_feature


@pytest.mark.parametrize(
    "feature",
    (
        "AUTOTUNE",
        "PREFETCH",
        "ASYNC_WRITER",
        "TIME_MAJOR_CONV",
        "2P_PARALLEL_IO",
        "RF_GPU",
    ),
)
def test_core_performance_paths_ignore_legacy_disable_environment(monkeypatch, feature):
    """Saved sessions cannot disable the performance paths that are now core."""
    monkeypatch.setenv(f"WAVEN_{feature}", "0")

    assert enabled_feature(feature, default=False)

