"""Application entry points for interactive Waven tools.

Configuration loading stays importable in headless environments. The Tk GUI is
only imported when its two public entry points are explicitly requested.
"""
from __future__ import annotations

from typing import Any

from .launcher import LaunchSettings, launch_from_project, load_launch_settings


def __getattr__(name: str) -> Any:
    """Lazily expose Tk entry points without forcing GUI dependencies at import."""
    if name in {"run", "select_workflow"}:
        from .gui import run, select_workflow

        globals().update(run=run, select_workflow=select_workflow)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "LaunchSettings",
    "launch_from_project",
    "load_launch_settings",
    "run",
    "select_workflow",
]
