"""Unit tests for graph-by-graph GUI export selection."""

from waven.gui_support.export_selection import (
    classify_export_record,
    classify_individual_axis,
    graph_payload,
)


def test_current_display_records_are_classified_for_checkbox_selection():
    assert classify_export_record("All neurons", "Neuron Layout") == "layout"
    assert classify_export_record("All neurons", "Population Retinotopy Maps") == "retinotopy"
    assert classify_export_record("All neurons", "OSI and gOSI All Cells Distribution") == "selectivity"
    assert classify_export_record("Individual neuron", "Spike Train") == "spike_train"
    assert classify_export_record("Individual neuron", "Selected Neuron Tuning") == "tuning_dashboard"
    assert classify_export_record("Individual neuron", "PSTH-weighted Spike-Triggered Averages") == "sta"


def test_batch_axis_classification_keeps_dashboard_graphs_separate():
    assert classify_individual_axis("Individual neuron", "Selected Neuron Tuning", "Receptive Field") == "receptive_field"
    assert classify_individual_axis("Individual neuron", "Selected Neuron Tuning", "Elevation (deg)") == "elevation"
    assert classify_individual_axis("Individual neuron", "Selected Neuron Tuning", "Azimuth (deg)") == "azimuth"
    assert classify_individual_axis(
        "Individual neuron", "Selected Neuron Tuning", "Orientation tuning from firing rate (OSI 0.4)"
    ) == "orientation_firing_rate"
    assert classify_individual_axis("Individual neuron", "PSTH-weighted STA", "STA lag 100 ms") == "sta"


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
