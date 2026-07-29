"""Focused checks for headless, process-safe graph rendering."""

from __future__ import annotations

from waven.gui_support.export_rendering import (
    MAX_EXPORT_DPI,
    MIN_EXPORT_DPI,
    normalise_export_dpi,
    render_individual_bundle,
)


def test_export_dpi_is_numeric_and_bounded():
    assert normalise_export_dpi("150") == 150
    assert normalise_export_dpi("not-a-number") == 200
    assert normalise_export_dpi(-1) == MIN_EXPORT_DPI
    assert normalise_export_dpi(9_999) == MAX_EXPORT_DPI


def test_headless_spike_train_writes_centered_png_and_svg(tmp_path):
    png_path = tmp_path / "spike.png"
    svg_path = tmp_path / "spike.svg"

    result = render_individual_bundle(
        {
            "dpi": 100,
            "graphs": [
                {
                    "graph_kind": "spike_train",
                    "title": "Trial-averaged Spike Train",
                    "payload": {"spike_train": [0.0, 1.0, 0.25, 0.5]},
                    "outputs": {"png": str(png_path), "svg": str(svg_path)},
                }
            ],
        }
    )

    assert result["graphs"] == 1
    assert png_path.exists() and png_path.stat().st_size > 0
    assert svg_path.exists() and svg_path.stat().st_size > 0
