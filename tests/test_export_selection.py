"""Unit tests for graph-by-graph GUI export selection."""

from waven.gui_support.export_selection import (
    classify_export_record,
    classify_individual_axis,
    graph_payload,
)
from waven.gui_support.helpers import _export_safe_name


def test_current_display_records_are_classified_for_checkbox_selection():
    assert classify_export_record("All neurons", "Neuron Layout") == "layout"
    assert classify_export_record("All neurons", "Population Retinotopy Maps") == "retinotopy"
    assert classify_export_record("All neurons", "OSI and gOSI All Cells Distribution") == "selectivity"
    assert classify_export_record("Individual neuron", "Spike Train") == "spike_train"
    assert classify_export_record("Individual neuron", "Selected Neuron Tuning") == "tuning_dashboard"
    assert classify_export_record("Individual neuron", "PSTH-weighted Spike-Triggered Averages") == "sta"
    assert classify_export_record("Individual neuron", "Run Full Model diagnostics neuron 7") == "model_diagnostics"


def test_batch_axis_classification_keeps_dashboard_graphs_separate():
    assert classify_individual_axis("Individual neuron", "Selected Neuron Tuning", "Receptive Field") == "receptive_field"
    assert classify_individual_axis("Individual neuron", "Selected Neuron Tuning", "Elevation (deg)") == "elevation"
    assert classify_individual_axis("Individual neuron", "Selected Neuron Tuning", "Azimuth (deg)") == "azimuth"
    assert classify_individual_axis(
        "Individual neuron", "Selected Neuron Tuning", "Orientation tuning from firing rate (OSI 0.4)"
    ) == "orientation_firing_rate"
    assert classify_individual_axis("Individual neuron", "PSTH-weighted STA", "STA lag 100 ms") == "sta"


def test_export_safe_name_limits_long_titles_without_collisions():
    long_title = "Orientation correlation firing rate OSI 0.030594 gOSI 0.027696 " * 4
    compact = _export_safe_name(long_title, maximum_length=56)

    assert len(compact) <= 56
    assert compact != _export_safe_name(long_title + "different", maximum_length=56)
    assert all(character.isalnum() or character in "-_." for character in compact)


def test_single_graph_payload_excludes_sibling_dashboard_arrays():
    payload = {
        "source": "Inspect Single Neuron",
        "neuron_id": 4,
        "rf2d": [[1]],
        "azimuth_correlation_tuning": [1, 2],
        "elevation_correlation_tuning": [3, 4],
    }

    graph = graph_payload(payload, "azimuth")

    assert graph["neuron_id"] == 4
    assert graph["azimuth_correlation_tuning"] == [1, 2]
    assert "rf2d" not in graph
    assert "elevation_correlation_tuning" not in graph


def test_orientation_axis_payload_uses_its_precomputed_reference_like_record():
    correlation_record = {"unit_id": "shank0_unit0", "curve_kind": "correlation"}
    rate_record = {"unit_id": "shank0_unit0", "curve_kind": "firing_rate"}
    payload = {
        "source": "Inspect Single Neuron",
        "neuron_id": 0,
        "orientation_correlation_export": correlation_record,
        "orientation_firing_rate_export": rate_record,
    }

    correlation = graph_payload(payload, "orientation_correlation")
    firing_rate = graph_payload(payload, "orientation_firing_rate")

    assert correlation["orientation_export"] == correlation_record
    assert firing_rate["orientation_export"] == rate_record
