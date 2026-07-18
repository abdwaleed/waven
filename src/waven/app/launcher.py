"""Project-root configuration loading for interactive Waven sessions.

The root ``ui.py`` script is intentionally only a small, user-friendly error
boundary.  This module owns the deterministic part of startup so it is usable
from tests, other launchers, and future non-Tk front ends.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from ..config import WORKFLOW_EPHYS


@dataclass(frozen=True)
class LaunchSettings:
    """GUI-ready settings resolved from one project configuration file."""

    project_root: Path
    param_defaults: Mapping[str, Any]
    gabor_defaults: Mapping[str, Any]
    workflow: Optional[str]
    gui_options: Mapping[str, Any]


def load_launch_settings(
    project_root: Union[Path, str],
    config_name: str = "pipeline_config.json",
) -> LaunchSettings:
    """Read optional GUI defaults without importing any GUI backend.

    ``{PROJECT_ROOT}`` in JSON values is resolved to ``project_root``. Legacy
    flat configuration keys remain supported so existing experiments launch
    unchanged.
    """
    root = Path(project_root).resolve()
    config_path = root / config_name
    if not config_path.exists():
        return LaunchSettings(root, {}, {}, None, {})

    raw_text = config_path.read_text(encoding="utf-8")
    resolved_text = raw_text.replace("{PROJECT_ROOT}", str(root).replace("\\", "/"))
    payload = json.loads(resolved_text) if resolved_text.strip() else {}
    if not isinstance(payload, dict):
        raise ValueError(f"{config_name} must contain a JSON object when provided.")

    workflow = payload.get("workflow") or None
    gui_options = dict(payload.get("gui") or {})
    gabor_defaults = dict(payload.get("gabor") or payload.get("gabor_param") or {})
    common = dict(payload.get("common") or {})
    legacy = dict(payload.get("analysis") or payload.get("param_defaults") or {})
    workflow_section = (
        payload.get("ephys") or {}
        if workflow == WORKFLOW_EPHYS
        else payload.get("two_photon") or payload.get("2p") or payload.get("two_p") or {}
    )
    workflow_values = dict(workflow_section)
    return LaunchSettings(
        root,
        {**legacy, **common, **workflow_values},
        gabor_defaults,
        workflow,
        gui_options,
    )


def launch_from_project(
    project_root: Union[Path, str],
    *,
    runner: Optional[Callable[..., Any]] = None,
) -> Any:
    """Load project settings and invoke the GUI runner.

    The optional ``runner`` injection makes startup routing testable without
    constructing a Tk root.
    """
    settings = load_launch_settings(project_root)
    if runner is None:
        from .gui import run

        runner = run
    return runner(
        dict(settings.param_defaults),
        dict(settings.gabor_defaults),
        workflow=settings.workflow,
        gui_options=dict(settings.gui_options),
    )
