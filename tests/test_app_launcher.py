"""Tests for the GUI-launch configuration boundary."""

from __future__ import annotations

import json

from waven.app.launcher import launch_from_project, load_launch_settings


def test_load_launch_settings_merges_legacy_common_and_ephys_values(tmp_path):
    (tmp_path / "pipeline_config.json").write_text(
        json.dumps(
            {
                "workflow": "ephys",
                "analysis": {"Number of Frames": "10"},
                "common": {"Project Root": "{PROJECT_ROOT}/experiment"},
                "ephys": {"Sampling Rate (samples / sec)": "30000"},
                "gabor": {"N_thetas": "12"},
                "gui": {"theme": "dark", "export_dpi": 150},
            }
        ),
        encoding="utf-8",
    )

    settings = load_launch_settings(tmp_path)

    assert settings.workflow == "ephys"
    assert settings.param_defaults["Number of Frames"] == "10"
    assert settings.param_defaults["Sampling Rate (samples / sec)"] == "30000"
    assert settings.param_defaults["Project Root"].endswith("/experiment")
    assert settings.gabor_defaults == {"N_thetas": "12"}
    assert settings.gui_options == {"theme": "dark", "export_dpi": 150}


def test_launch_from_project_passes_resolved_settings_without_creating_a_gui(tmp_path):
    (tmp_path / "pipeline_config.json").write_text('{"workflow": "2p"}', encoding="utf-8")
    received = {}

    def runner(params, gabor, *, workflow, gui_options):
        received.update(params=params, gabor=gabor, workflow=workflow, gui_options=gui_options)
        return "started"

    assert launch_from_project(tmp_path, runner=runner) == "started"
    assert received == {"params": {}, "gabor": {}, "workflow": "2p", "gui_options": {}}
