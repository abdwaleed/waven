"""Headless, process-safe rendering helpers for graph exports.

The GUI owns interactive Matplotlib canvases and must therefore never hand a
live figure to a process.  These helpers receive either a compact, explicit
graph payload or a previously-created figure snapshot and write complete files
from a clean Agg canvas.  Keeping this module free of Tk state is important on
Windows, where process workers start by importing modules afresh.
"""
from __future__ import annotations

import os
import pickle
import uuid
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Mapping

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


DEFAULT_EXPORT_DPI = 200
MIN_EXPORT_DPI = 72
MAX_EXPORT_DPI = 600


def normalise_export_dpi(value: Any, default: int = DEFAULT_EXPORT_DPI) -> int:
    """Return a bounded export DPI from a GUI or JSON value."""
    try:
        dpi = int(round(float(value)))
    except (TypeError, ValueError):
        dpi = int(default)
    return max(MIN_EXPORT_DPI, min(MAX_EXPORT_DPI, dpi))


def _atomic_savefig(figure: Figure, path: str, image_format: str, dpi: int, **kwargs: Any) -> None:
    """Write one visual atomically so cancelled/failed jobs leave no final stub."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid.uuid4().hex}.tmp"
    )
    try:
        figure.savefig(
            temporary,
            format=image_format,
            dpi=normalise_export_dpi(dpi),
            **kwargs,
        )
        os.replace(temporary, destination)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _new_figure(figsize: tuple[float, float] = (6.4, 4.8)) -> tuple[Figure, Any]:
    """Create a centered, standalone canvas with layout reserved for labels."""
    figure = Figure(figsize=figsize, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot(111)


def _as_float_array(value: Any) -> np.ndarray:
    """Coerce a numeric export field without allowing an object array to leak in."""
    return np.asarray(value, dtype=float).reshape(-1)


def _plot_spike_train(axis: Any, payload: Mapping[str, Any], title: str) -> None:
    values = _as_float_array(payload.get("spike_train", ()))
    axis.plot(np.arange(values.size), values, color="#111827", linewidth=1.3,
              label="Trial-averaged activity")
    axis.set_title(title)
    axis.set_xlabel("Frame index")
    axis.set_ylabel("Activity (a.u.)")
    if values.size:
        axis.legend(fontsize=8)


def _plot_receptive_field(figure: Figure, axis: Any, payload: Mapping[str, Any], title: str) -> None:
    matrix = np.asarray(payload.get("rf2d", ()), dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("Receptive-field export payload does not contain a 2D map.")
    finite = matrix[np.isfinite(matrix)]
    if not finite.size:
        raise ValueError("Receptive-field export map contains no finite values.")
    limit = max(float(np.max(np.abs(finite))), 1e-8)
    image = axis.imshow(
        matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="equal",
        origin="upper", interpolation="nearest",
    )
    extent = payload.get("rf_extent_degrees")
    if isinstance(extent, (list, tuple)) and len(extent) == 4:
        x_left, x_right, y_bottom, y_top = (float(item) for item in extent)
        axis.set_xticks(
            np.linspace(0, max(0, matrix.shape[1] - 1), 3),
            np.round(np.linspace(x_left, x_right, 3), 2),
        )
        axis.set_yticks(
            np.linspace(0, max(0, matrix.shape[0] - 1), 3),
            np.round(np.linspace(y_bottom, y_top, 3), 2),
        )
        axis.set_xlabel("Azimuth (deg)")
        axis.set_ylabel("Elevation (deg)")
    else:
        axis.set_xlabel("Stimulus x (px)")
        axis.set_ylabel("Stimulus y (px)")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="RF correlation (r)")


def _plot_profile(axis: Any, payload: Mapping[str, Any], title: str, values_key: str,
                  positions_key: str, xlabel: str) -> None:
    values = _as_float_array(payload.get(values_key, ()))
    positions = _as_float_array(payload.get(positions_key, ()))
    x_values = positions if positions.size == values.size else np.arange(values.size)
    axis.plot(x_values, values, color="#2563EB", linewidth=1.5)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("RF correlation (r)")


def _close_periodic_curve(angles: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Close an orientation curve at 180 degrees when its samples are open."""
    if not angles.size or not values.size:
        return angles, values
    count = min(angles.size, values.size)
    angles = angles[:count]
    values = values[:count]
    if angles[-1] < 180.0 - 1e-9:
        return np.append(angles, 180.0), np.append(values, values[0])
    return angles, values


def _plot_orientation(axis: Any, payload: Mapping[str, Any], title: str, *, firing_rate: bool) -> None:
    record = payload.get("orientation_export")
    tuning = record.get("tuning", {}) if isinstance(record, Mapping) else {}
    angles = _as_float_array(tuning.get("orientations", ()))
    values = _as_float_array(tuning.get("mean_values", ()))
    if not values.size:
        values = _as_float_array(
            payload.get(
                "orientation_firing_rate_tuning" if firing_rate else "orientation_correlation_tuning",
                (),
            )
        )
    if not angles.size:
        angles = np.linspace(0.0, 180.0, values.size, endpoint=False) if values.size else np.array([])
    angles, values = _close_periodic_curve(angles, values)
    errors = _as_float_array(
        tuning.get("sem_values", payload.get("orientation_correlation_ci_95", ()))
    )
    if errors.size:
        _unused_angles, errors = _close_periodic_curve(angles[:-1] if angles.size > errors.size else angles, errors)
        if errors.size != values.size:
            errors = np.array([])
    color, error_color = ("#DC2626", "#FCA5A5") if firing_rate else ("#2563EB", "#60A5FA")
    if errors.size:
        axis.errorbar(angles, values, yerr=errors, fmt="o-", color=color,
                      ecolor=error_color, capsize=3, linewidth=1.35)
    else:
        axis.plot(angles, values, "o-", color=color, linewidth=1.35)
    axis.set_title(title)
    axis.set_xlim(0, 180)
    axis.set_xticks((0, 90, 180))
    axis.set_xlabel("Orientation (deg)")
    axis.set_ylabel(
        str(tuning.get("value_label") or ("Firing rate (Hz)" if firing_rate else "RF correlation (r)"))
    )


def _plot_size(axis: Any, payload: Mapping[str, Any], title: str) -> None:
    values = _as_float_array(payload.get("size_tuning", ()))
    sizes = _as_float_array(payload.get("size_degrees", ()))
    x_values = sizes if sizes.size == values.size else np.arange(values.size)
    errors = _as_float_array(payload.get("size_correlation_ci_95", ()))
    if errors.size == values.size:
        axis.errorbar(x_values, values, yerr=errors, fmt="o-", color="#7C3AED",
                      ecolor="#C4B5FD", capsize=3, linewidth=1.35)
    else:
        axis.plot(x_values, values, "o-", color="#7C3AED", linewidth=1.35)
    axis.set_title(title)
    axis.set_xlabel("Size (deg)")
    axis.set_ylabel("RF correlation (r)")


def _plot_frequency(axis: Any, payload: Mapping[str, Any], title: str) -> None:
    values = _as_float_array(payload.get("frequency_tuning", ()))
    frequencies = _as_float_array(payload.get("frequency_cpd", ()))
    x_values = frequencies if frequencies.size == values.size else np.arange(values.size)
    axis.plot(x_values, values, "o-", color="#111827", linewidth=1.35)
    axis.set_title(title)
    axis.set_xlabel("Spatial frequency (cycles/deg)")
    axis.set_ylabel("RF correlation (r)")


def _plot_sta(figure: Figure, axis: Any, payload: Mapping[str, Any], title: str) -> None:
    matrix = np.asarray(payload.get("sta_map", ()), dtype=float)
    if matrix.ndim != 2 or not matrix.size:
        raise ValueError("STA export payload does not contain a 2D lag map.")
    limits = payload.get("sta_color_limits")
    if isinstance(limits, (tuple, list)) and len(limits) == 2:
        minimum, maximum = (float(value) for value in limits)
    else:
        minimum, maximum = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        finite = matrix[np.isfinite(matrix)]
        minimum, maximum = (
            (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
        )
    if maximum <= minimum:
        maximum = minimum + 1e-6
    image = axis.imshow(matrix, cmap="coolwarm", vmin=minimum, vmax=maximum,
                        aspect="equal", origin="upper", interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("Stimulus x (px)")
    axis.set_ylabel("Stimulus y (px)")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)


def _draw_individual_graph(figure: Figure, axis: Any, job: Mapping[str, Any]) -> None:
    """Draw exactly one requested graph on a dedicated, correctly padded canvas."""
    kind = str(job["graph_kind"])
    title = str(job["title"])
    payload = job.get("payload") or {}
    if kind == "spike_train":
        _plot_spike_train(axis, payload, title)
    elif kind == "receptive_field":
        _plot_receptive_field(figure, axis, payload, title)
    elif kind == "elevation":
        _plot_profile(axis, payload, title, "elevation_correlation_tuning", "elevation_degrees", "Elevation (deg)")
    elif kind == "azimuth":
        _plot_profile(axis, payload, title, "azimuth_correlation_tuning", "azimuth_degrees", "Azimuth (deg)")
    elif kind == "orientation_correlation":
        _plot_orientation(axis, payload, title, firing_rate=False)
    elif kind == "orientation_firing_rate":
        _plot_orientation(axis, payload, title, firing_rate=True)
    elif kind == "size_tuning":
        _plot_size(axis, payload, title)
    elif kind == "spatial_frequency":
        _plot_frequency(axis, payload, title)
    elif kind == "sta":
        _plot_sta(figure, axis, payload, title)
    else:
        raise ValueError(f"No headless renderer is available for graph kind '{kind}'.")
    # Dashboard-axis cropping used to leave long titles and labels pressed
    # against neighbouring panels.  A standalone canvas can consistently wrap
    # the title and reserve a little breathing room around every graph.
    axis.set_title(axis.get_title(), fontsize=10.5, pad=10, wrap=True)
    axis.tick_params(labelsize=9)


def render_individual_bundle(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Render selected individual-neuron graphs in one worker process.

    Each graph starts as a fresh figure instead of a cropped dashboard axis.
    This preserves a stable artboard, reserves layout space for all text, and
    lets several neuron bundles render on separate CPU cores.
    """
    started = perf_counter()
    dpi = normalise_export_dpi(job.get("dpi"))
    written: list[str] = []
    graphs = job.get("graphs") or ()
    for graph in graphs:
        figure, axis = _new_figure(tuple(graph.get("figsize", (6.4, 4.8))))
        try:
            _draw_individual_graph(figure, axis, graph)
            outputs = graph.get("outputs") or {}
            if outputs.get("png"):
                # Do not crop a dashboard: this fixed, standalone artboard is
                # deliberately centered and leaves constrained-layout padding.
                _atomic_savefig(figure, str(outputs["png"]), "png", dpi)
                written.append(str(outputs["png"]))
            if outputs.get("svg"):
                _atomic_savefig(figure, str(outputs["svg"]), "svg", dpi)
                written.append(str(outputs["svg"]))
        finally:
            figure.clear()
    return {
        "graphs": len(graphs),
        "files": written,
        "seconds": perf_counter() - started,
    }


def render_figure_snapshot(job: Mapping[str, Any]) -> Dict[str, Any]:
    """Render a complete pre-existing figure in a separate process.

    This is used for aggregate/dashboard and model exports where no explicit
    data-first renderer exists yet.  Whole figures are never cropped by axis,
    and ``pad_inches`` protects titles, labels, and legends.
    """
    started = perf_counter()
    serialized = job.get("figure_pickle_bytes")
    if not isinstance(serialized, (bytes, bytearray)):
        raise ValueError("Figure export job is missing its serialized figure.")
    figure = pickle.loads(serialized)
    FigureCanvasAgg(figure)
    dpi = normalise_export_dpi(job.get("dpi"))
    written: list[str] = []
    try:
        outputs = job.get("outputs") or {}
        if outputs.get("png"):
            _atomic_savefig(
                figure, str(outputs["png"]), "png", dpi,
                bbox_inches="tight", pad_inches=0.12,
            )
            written.append(str(outputs["png"]))
        if outputs.get("svg"):
            _atomic_savefig(
                figure, str(outputs["svg"]), "svg", dpi,
                bbox_inches="tight", pad_inches=0.12,
            )
            written.append(str(outputs["svg"]))
    finally:
        figure.clear()
    return {
        "graphs": 1,
        "files": written,
        "seconds": perf_counter() - started,
    }
