"""Pure helpers for filtering GUI graph exports.

The GUI deliberately keeps these rules separate from Tk widgets so the two
export sections can identify the requested visual axes and retain only the
payload values that belong to each graph.
"""
from __future__ import annotations

from typing import Any, Dict


ALL_NEURON_GRAPH_OPTIONS = (
    ("layout", "Neuron layout"),
    ("retinotopy", "Population retinotopy"),
    ("selectivity", "Orientation selectivity"),
    ("other", "Other all-neuron graphs"),
)

SINGLE_NEURON_GRAPH_OPTIONS = (
    ("spike_train", "Spike train"),
    ("receptive_field", "Receptive-field map"),
    ("elevation", "Elevation profile"),
    ("azimuth", "Azimuth profile"),
    ("orientation_correlation", "Orientation correlation"),
    ("orientation_firing_rate", "Orientation tuning (firing rate)"),
    ("size_tuning", "Size tuning"),
    ("spatial_frequency", "Spatial-frequency tuning"),
    ("sta", "PSTH-weighted STA lag maps"),
    ("run_model_amplitude", "Run Model amplitude tuning (fit for every neuron)"),
    ("run_model_phase", "Run Model phase tuning (fit for every neuron)"),
    ("run_model_drift", "Run Model drift tuning (fit for every neuron)"),
    ("run_full_model_amplitude", "Run Full Model amplitude tuning (fit for every neuron)"),
    ("run_full_model_phase", "Run Full Model phase tuning (fit for every neuron)"),
    ("run_full_model_drift", "Run Full Model drift tuning (fit for every neuron)"),
    ("other", "Other individual graphs"),
)


def classify_export_record(tab: str, title: str) -> str:
    """Return the checkbox category for a displayed figure record."""
    tab = str(tab or "").casefold()
    title = str(title or "").casefold()
    if tab == "all neurons":
        if "layout" in title or "position" in title:
            return "layout"
        if "retinotopy" in title:
            return "retinotopy"
        if "osi" in title or "selectivity" in title:
            return "selectivity"
        return "other"
    if "spike-triggered" in title or "psth-weighted sta" in title:
        return "sta"
    if "spike" in title:
        return "spike_train"
    model_prefix = None
    if "run full model" in title and "tuning" in title:
        model_prefix = "run_full_model"
    elif "run model" in title and "tuning" in title:
        model_prefix = "run_model"
    if model_prefix and "amplitude" in title:
        return f"{model_prefix}_amplitude"
    if model_prefix and "phase" in title:
        return f"{model_prefix}_phase"
    if model_prefix and "drift" in title:
        return f"{model_prefix}_drift"
    if "tuning" in title:
        return "tuning_dashboard"
    return "other"


def classify_individual_axis(tab: str, figure_title: str, axis_title: str) -> str:
    """Return the category for one graph/axis in an individual-neuron figure."""
    title = str(axis_title or figure_title or "").casefold()
    model_prefix = None
    if "run full model" in title:
        model_prefix = "run_full_model"
    elif "run model" in title:
        model_prefix = "run_model"
    if model_prefix and "amplitude" in title:
        return f"{model_prefix}_amplitude"
    if model_prefix and "phase" in title:
        return f"{model_prefix}_phase"
    if model_prefix and "drift" in title:
        return f"{model_prefix}_drift"
    if "sta" in title or "spike-triggered" in title:
        return "sta"
    if "spike" in title:
        return "spike_train"
    if "receptive field" in title:
        return "receptive_field"
    if "elevation" in title:
        return "elevation"
    if "azimuth" in title:
        return "azimuth"
    if "firing rate" in title and "orientation" in title:
        return "orientation_firing_rate"
    if "orientation" in title:
        return "orientation_correlation"
    if "size" in title:
        return "size_tuning"
    if "frequency" in title:
        return "spatial_frequency"
    return "other"


_PAYLOAD_KEYS = {
    "spike_train": ("spike_train", "trial_spikes"),
    "receptive_field": ("rf2d", "rf_extent_degrees", "best_params"),
    "elevation": ("elevation_correlation_tuning", "elevation_degrees"),
    "azimuth": ("azimuth_correlation_tuning", "azimuth_degrees"),
    "orientation_correlation": (
        "orientation_correlation_tuning",
        "orientation_correlation_ci_95",
        "orientation_correlation_export",
    ),
    "orientation_firing_rate": (
        "orientation_firing_rate_tuning",
        "osi",
        "gosi",
        "osi_source",
        "orientation_firing_rate_export",
    ),
    "size_tuning": ("size_tuning", "size_correlation_ci_95", "size_degrees"),
    "spatial_frequency": ("frequency_tuning", "frequency_tuning_available", "frequency_cpd"),
    "sta": (
        "sta_maps", "sta_lag_frames", "sta_lag_ms", "sta_variances",
        "sta_peak_lag_frame", "sta_peak_lag_ms", "sta_peak_variance",
    ),
    "run_model_amplitude": ("paper_tuning",),
    "run_model_phase": ("paper_tuning",),
    "run_model_drift": ("paper_tuning",),
    "run_full_model_amplitude": ("paper_tuning",),
    "run_full_model_phase": ("paper_tuning",),
    "run_full_model_drift": ("paper_tuning",),
}


def graph_payload(payload: Any, graph_kind: str) -> Dict[str, Any]:
    """Return only data relevant to one exported graph, never its sibling axes."""
    if not isinstance(payload, dict):
        return {"graph_kind": graph_kind}
    result = {key: payload[key] for key in ("source", "neuron_id") if key in payload}
    result["graph_kind"] = graph_kind
    for key in _PAYLOAD_KEYS.get(graph_kind, ()):
        if key in payload:
            result[key] = payload[key]
    if graph_kind == "orientation_correlation" and "orientation_correlation_export" in result:
        result["orientation_export"] = result["orientation_correlation_export"]
    elif graph_kind == "orientation_firing_rate" and "orientation_firing_rate_export" in result:
        result["orientation_export"] = result["orientation_firing_rate_export"]
    return result
