"""Tkinter GUI for the waven Gabor-wavelet analysis pipeline.

Provides a staged workflow (Gabor bank → stimulus wavelets → neural RF analysis),
live terminal output, and embedded matplotlib visualizations.
"""
import json
import gzip
import hashlib
import io
import pickle
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
import tempfile
import shutil
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import threading
import time
import sys
import gc
from . import LoadPinkNoise as lpn
import traceback
import customtkinter as ctk

# --- DPI Awareness ---
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    myappid = 'neuro.gabor.toolkit.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

from .config import (
    AnalysisConfig,
    DEFAULT_COMMON_PARAMS,
    DEFAULT_EPHYS_PARAMS,
    DEFAULT_TWO_PHOTON_PARAMS,
    WORKFLOW_2P,
    WORKFLOW_EPHYS,
    coarse_grid_dimensions,
    parse_literal,
)
from .WaveletGenerator import *
from .LoadPinkNoise import *
from .Analysis_Utils import *
from .performance import video_downsample_chunk_size
from .pipeline import smooth_best_positions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
import zarr
from numcodecs import Blosc
from .wavelet_io import convert_npy_to_zarr


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        text = self.widget.get()
        if not text: return
        x, y, cx, cy = self.widget.bbox("insert") or (0,0,0,0)
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 20
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tk.Label(tw, text=text, justify=tk.LEFT,
                      background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                      font=("Segoe UI Variable Display", "9", "normal"))
        label.pack(ipadx=1)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw: tw.destroy()

def _parse_data_dir(value):
    try:
        parsed = parse_literal(value, "Dir")
    except ValueError:
        parsed = value
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(path) for path in parsed]
    return [str(parsed)]


def _format_bytes(size):
    if size is None or size < 0:
        return "0.00 GB"
    return f"{size / (1024 ** 3):.2f} GB"


def _build_size_text(gb_bytes):
    return _format_bytes(gb_bytes)


def _folder_size_bytes(path):
    total = 0
    if not path or not os.path.exists(path):
        return None
    if os.path.isfile(path):
        return os.path.getsize(path)
    for root_dir, _, filenames in os.walk(path):
        for filename in filenames:
            file_path = os.path.join(root_dir, filename)
            try:
                total += os.path.getsize(file_path)
            except OSError:
                pass
    return total


def _zarr_output_path(path):
    path = str(path).strip()
    if not path:
        return path
    if path.endswith(".zarr"):
        return path
    stem, ext = os.path.splitext(path)
    return (stem if ext else path) + ".zarr"


def _estimate_blosc_bytes(raw_bytes, sample):
    if raw_bytes <= 0:
        return 0
    arr = np.ascontiguousarray(sample)
    if arr.nbytes == 0:
        return raw_bytes
    try:
        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
        encoded = compressor.encode(arr)
        ratio = max(len(encoded) / arr.nbytes, 0.02)
        return int(raw_bytes * ratio)
    except Exception:
        return int(raw_bytes * 0.75)


def _safe_name(value):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value)).strip("_") or "plot"


def _prompt_workflow() -> str:
    """Ask the user to choose a data workflow before opening the main GUI."""
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    prompt_root = ctk.CTk()
    prompt_root.withdraw()
    dialog = ctk.CTkToplevel(prompt_root)
    dialog.title("Select Data Workflow")
    dialog.geometry("420x180")
    dialog.resizable(False, False)
    dialog.configure(fg_color="#F3F6F8")
    dialog.transient(prompt_root)

    choice = {"workflow": None}

    ctk.CTkLabel(
        dialog,
        text="Which neural data type are you analyzing?",
        text_color="#1F2937",
        font=ctk.CTkFont(size=14, weight="bold"),
    ).pack(pady=(24, 12))

    button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
    button_frame.pack(pady=8)

    def select(workflow):
        choice["workflow"] = workflow
        dialog.destroy()

    ctk.CTkButton(
        button_frame,
        text="Two-Photon (2p)",
        width=150,
        height=34,
        corner_radius=6,
        command=lambda: select(WORKFLOW_2P),
    ).pack(side=tk.LEFT, padx=8)
    ctk.CTkButton(
        button_frame,
        text="Electrophysiology (Ephys)",
        width=185,
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=lambda: select(WORKFLOW_EPHYS),
    ).pack(side=tk.LEFT, padx=8)

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.grab_set()
    prompt_root.wait_window(dialog)
    prompt_root.destroy()

    if choice["workflow"] is None:
        raise SystemExit(0)
    return choice["workflow"]


def select_workflow() -> str:
    """Public launch-time workflow selector used by script entry points."""
    return _prompt_workflow()


def run(param_defaults, gabor_param, workflow=None):
    if workflow is None:
        workflow = _prompt_workflow()

    GABOR_LABELS = {
        "N_thetas": "Orientation Count",
        "Sigmas": "Filter Sizes (px)",
        "Frequencies": "Spatial Frequencies (cyc/px)",
        "Phases": "Phases (degrees)",
        "NX": "Stimulus Width (px)",
        "NY": "Stimulus Height (px)",
        "Save Path": "Gabor Library Output",
    }

    ANALYSIS_LABELS = {
        "Dir": "Data Directory (.tiff or .pkl/.din)",
        "Path Directory": "Wavelet Output Directory",
        "Experiment Info": "Experiment ID (mouse, date, #)",
        "Number of Planes": "Imaging Planes",
        "Block End": "Session Block Start Frame",
        "screen_x": "Display Width (px)",
        "screen_y": "Display Height (px)",
        "NX": "Analysis Grid Width (px)",
        "NY": "Analysis Grid Height (px)",
        "Resolution": "Microscope Resolution (µm/px)",
        "Sampling Rate (samples / sec)": "Recording Sampling Rate (Hz)",
        "Sigmas": "RF Filter Sizes (px)",
        "Sigmas Full Model": "Full-Model Filter Sizes (px)",
        "Frequencies": "Stimulus Frequencies (cyc/px)",
        "Visual Coverage": "Visual Field Coverage (°)",
        "Analysis Coverage": "Analysis Field Coverage (°)",
        "Hz": "Stimulus Frame Rate (Hz)",
        "Number of Frames": "Frames per Trial",
        "Number of Trials to Keep": "Trials to Retain",
        "Movie Path": "Stimulus Movie (.mp4)",
        "Library Path": "Gabor Library Path",
        "Spks Path": "Pre-aligned Spikes (.npy, optional)",
        "Full Model Wavelet Path": "Full-Model Wavelet Store",
        "Full Model Save Path": "Full-Model Results Directory",
        "Plot Cache Path": "Plot Cache File",
        "Recovery Cache Directory": "Recovery Checkpoint Directory",
        "Neuron ID": "Neuron Index",
    }

    workflow_param_keys = AnalysisConfig.gui_param_keys(workflow)
    merged_param_defaults = dict(DEFAULT_COMMON_PARAMS)
    if workflow == WORKFLOW_2P:
        merged_param_defaults.update(DEFAULT_TWO_PHOTON_PARAMS)
    else:
        merged_param_defaults.update(DEFAULT_EPHYS_PARAMS)
    merged_param_defaults.update(param_defaults)
    filtered_param_defaults = {
        key: merged_param_defaults.get(key, "")
        for key in workflow_param_keys
    }

    FIELD_LABELS = {**GABOR_LABELS, **ANALYSIS_LABELS}

    BROWSE_KIND = {
        "Save Path": "savefile",
        "Movie Path": "file",
        "Library Path": "file",
        "Spks Path": "file",
        "Path Directory": "dir",
        "Dir": "dir",
        "Full Model Wavelet Path": "dir",
        "Full Model Save Path": "dir",
        "Plot Cache Path": "savefile",
        "Recovery Cache Directory": "dir",
    }

    BROWSE_ICONS = {"file": "📄", "savefile": "📄", "dir": "📁"}

    class RedirectText:
        def __init__(self, widget): self.widget = widget
        def write(self, string):
            self.widget.insert(tk.END, string)
            self.widget.see(tk.END)
        def flush(self): pass

    task_state = {"name": None, "start": None, "detail": None}

    def flash_taskbar():
        try:
            FLASHW_ALL = 3
            class FLASHWINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", ctypes.c_uint),
                    ("hwnd", ctypes.c_void_p),
                    ("dwFlags", ctypes.c_uint),
                    ("uCount", ctypes.c_uint),
                    ("dwTimeout", ctypes.c_uint),
                ]

            hwnd = root.winfo_id()
            fwi = FLASHWINFO(ctypes.sizeof(FLASHWINFO), hwnd, FLASHW_ALL, 3, 0)
            ctypes.windll.user32.FlashWindowEx(ctypes.byref(fwi))
        except Exception:
            pass

    def update_progress(percent=None, message=None, detail=None):
        if percent is None:
            progress_bar.configure(mode="indeterminate")
            progress_bar.start(12)
        else:
            progress_bar.stop()
            progress_bar.configure(mode="determinate", value=max(0, min(percent, 100)))

        if detail is not None:
            task_state["detail"] = detail

        if message is None and task_state["name"]:
            message = task_state["name"]
        if message is not None:
            suffix = f" - {task_state['detail']}" if task_state.get("detail") else ""
            status_var.set(f"{message}{suffix}")

        if percent is not None and task_state.get("start") and percent > 0 and percent < 100:
            elapsed = time.time() - task_state["start"]
            eta = elapsed * (100 - percent) / percent
            status_var.set(f"{message}{suffix} · ETA {int(eta)}s")

        root.update_idletasks()

    def show_terminal_half():
        """Reveal the log pane and allocate the lower half of the window to it."""
        try:
            if frame_log not in right_pane.panes():
                right_pane.add(frame_log, weight=1)
            root.update_idletasks()
            total_h = right_pane.winfo_height() or root.winfo_height()
            top_h = max(120, int(total_h * 0.68))
            right_pane.sashpos(0, top_h)
        except Exception:
            pass

    def begin_task(task_name):
        task_state["name"] = task_name
        task_state["start"] = time.time()
        task_state["detail"] = None
        show_terminal_half()
        update_progress(None, f"Running: {task_name}")
        for btn in all_buttons:
            btn.configure(state=tk.DISABLED)
        print(f"\n--- {task_name} ---")

    def end_task():
        if task_state["name"]:
            progress_bar.stop()
            progress_bar.configure(mode="determinate", value=100)
            print(f"--- Finished: {task_state['name']} ---\n")
            flash_taskbar()
        task_state["name"] = None
        task_state["start"] = None
        task_state["detail"] = None
        status_var.set("Ready")
        for btn in all_buttons:
            btn.configure(state=tk.NORMAL)

    def run_in_thread(func, task_name=None):
        label = task_name or func.__name__.replace("_", " ").title()

        def wrapper(*args, **kwargs):
            root.after(0, lambda: begin_task(label))

            def thread_target():
                success = False
                _start_recovery_checkpoint(label)
                try:
                    result = func(*args, **kwargs)
                    success = result is not False
                except Exception as exc:
                    # Print a clear header for the error
                    print(f"\n[!] AN ERROR OCCURRED IN TASK: {label}")
                    # This will dump the full "most recent call last" traceback 
                    # straight to your GUI terminal!
                    traceback.print_exc() 
                finally:
                    _finish_recovery_checkpoint(success)
                    root.after(0, end_task)

            threading.Thread(target=thread_target, daemon=True).start()

        return wrapper
    
    temp_directories = []
    current_wavelet_dir = [None]
    analysis_state = {}
    embedded_canvases = []
    figure_export_records = []
    active_recovery_dir = {"path": None}

    def _field_value(entries, key, default=""):
        entry = entries.get(key) if isinstance(entries, dict) else None
        if entry is None:
            return default
        try:
            return entry.get()
        except Exception:
            return default

    def _plot_cache_path():
        value = _field_value(param_entries, "Plot Cache Path", "").strip()
        if value.lower() in ("", "none", "null"):
            save_dir = _field_value(param_entries, "Full Model Save Path", "").strip()
            if save_dir.lower() in ("", "none", "null"):
                movie_path = _field_value(param_entries, "Movie Path", "").strip()
                save_dir = os.path.dirname(movie_path) or "."
            value = os.path.join(save_dir, "plot_cache.pkl.gz")
        return value

    def _recovery_root():
        value = _field_value(param_entries, "Recovery Cache Directory", "").strip()
        if value.lower() in ("", "none", "null"):
            save_dir = _field_value(param_entries, "Full Model Save Path", "").strip()
            if save_dir.lower() in ("", "none", "null"):
                movie_path = _field_value(param_entries, "Movie Path", "").strip()
                save_dir = os.path.dirname(movie_path) or "."
            value = os.path.join(save_dir, "recovery_cache")
        return value

    def _load_plot_cache():
        path = _plot_cache_path()
        if not os.path.exists(path):
            return {"version": 1, "entries": {}}
        try:
            with gzip.open(path, "rb") as handle:
                cache = pickle.load(handle)
            if not isinstance(cache, dict):
                return {"version": 1, "entries": {}}
            cache.setdefault("version", 1)
            cache.setdefault("entries", {})
            return cache
        except Exception as exc:
            print(f"Could not load plot cache {path}: {exc}")
            return {"version": 1, "entries": {}}

    def _save_plot_cache(cache):
        path = _plot_cache_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with gzip.open(tmp_path, "wb") as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        print(f"Updated plot cache: {path}")

    def _cache_fingerprint(extra=None):
        fields = {
            "workflow": workflow,
            "gabor": {key: _field_value(gabor_entries, key) for key in sorted(gabor_entries)},
            "analysis": {
                key: _field_value(param_entries, key)
                for key in sorted(param_entries)
                if key not in {"Neuron ID", "Plot Cache Path", "Recovery Cache Directory"}
            },
            "extra": extra or {},
        }
        payload = json.dumps(fields, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_key(kind, neuron_id=None):
        if neuron_id is None:
            return kind
        return f"{kind}:neuron:{int(neuron_id)}"

    def _figure_records(items):
        records = []
        for tab_name, title, fig in items:
            image_buffer = io.BytesIO()
            fig.savefig(image_buffer, format="png", dpi=150, bbox_inches="tight")
            records.append(
                {
                    "tab": tab_name,
                    "title": title,
                    "png": image_buffer.getvalue(),
                    "export_payload": _cache_safe_payload(getattr(fig, "_waven_export_payload", None)),
                    "export_artist_data": _extract_figure_data(fig),
                }
            )
        return records

    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _json_safe(val) for key, val in value.items()}
        return repr(value)

    def _cache_safe_payload(payload):
        try:
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            return payload
        except Exception as exc:
            return {
                "cache_note": f"Original payload was not pickleable for plot cache: {exc}",
                "summary": _json_safe(payload),
            }

    def _extract_figure_data(fig):
        axes_data = []
        for ax_index, ax in enumerate(fig.axes):
            axis_record = {
                "index": ax_index,
                "title": ax.get_title(),
                "xlabel": ax.get_xlabel(),
                "ylabel": ax.get_ylabel(),
                "xlim": list(ax.get_xlim()),
                "ylim": list(ax.get_ylim()),
                "lines": [],
                "images": [],
                "collections": [],
            }
            for line_index, line in enumerate(ax.lines):
                axis_record["lines"].append(
                    {
                        "index": line_index,
                        "label": line.get_label(),
                        "x": np.asarray(line.get_xdata(orig=False)),
                        "y": np.asarray(line.get_ydata(orig=False)),
                    }
                )
            for image_index, image in enumerate(ax.images):
                axis_record["images"].append(
                    {
                        "index": image_index,
                        "array": np.asarray(image.get_array()),
                        "extent": _json_safe(image.get_extent()),
                        "cmap": image.get_cmap().name if image.get_cmap() else None,
                        "clim": list(image.get_clim()),
                    }
                )
            for collection_index, collection in enumerate(ax.collections):
                collection_record = {"index": collection_index}
                try:
                    collection_record["offsets"] = np.asarray(collection.get_offsets())
                except Exception:
                    pass
                try:
                    values = collection.get_array()
                    if values is not None:
                        collection_record["values"] = np.asarray(values)
                except Exception:
                    pass
                try:
                    collection_record["sizes"] = np.asarray(collection.get_sizes())
                except Exception:
                    pass
                try:
                    collection_record["facecolors"] = np.asarray(collection.get_facecolors())
                except Exception:
                    pass
                axis_record["collections"].append(collection_record)
            axes_data.append(axis_record)
        return {
            "figure_size_inches": list(fig.get_size_inches()),
            "dpi": fig.dpi,
            "axes": axes_data,
        }

    def _add_array_exports(prefix, value, arrays, metadata):
        key = _safe_name(prefix)
        if isinstance(value, np.ndarray):
            if value.dtype != object:
                arrays[key] = value
            else:
                metadata[key] = {"array_shape": list(value.shape), "dtype": str(value.dtype)}
            return
        if isinstance(value, np.generic):
            metadata[key] = value.item()
            return
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                _add_array_exports(f"{prefix}_{child_key}", child_value, arrays, metadata)
            return
        if isinstance(value, (list, tuple)):
            try:
                arr = np.asarray(value)
                if arr.dtype != object:
                    arrays[key] = arr
                    return
            except Exception:
                pass
            for index, child_value in enumerate(value):
                _add_array_exports(f"{prefix}_{index}", child_value, arrays, metadata)
            return
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
        else:
            metadata[key] = repr(value)

    def _tab_name_for_parent(parent_container):
        try:
            if parent_container is frame_plot_all:
                return "All neurons"
            if parent_container is frame_plot_individual:
                return "Individual neuron"
        except NameError:
            pass
        return "Plots"

    def _figure_title(fig, fallback="Plot"):
        for ax in fig.axes:
            if ax.get_title():
                return ax.get_title()
        return fallback

    def _set_figure_export_payload(fig, payload):
        setattr(fig, "_waven_export_payload", payload)

    def _model_result_payload(model_name, result, neuron_id):
        if model_name == "run_Model":
            names = ["predictions", "nonlinear_params", "rho_phi_params", "metrics", "interpolators"]
        else:
            names = [
                "predictions",
                "params",
                "nonlinear_params",
                "rho_phi_params",
                "metrics",
                "orientation_selectivity",
                "interpolators",
            ]
        payload = {"source": model_name, "neuron_id": neuron_id}
        if isinstance(result, tuple):
            for name, value in zip(names, result):
                payload[name] = value
        else:
            payload["result"] = result
        return payload

    def _active_export_records(tab_name=None):
        active = []
        for record in list(figure_export_records):
            section = record.get("section")
            try:
                if section is not None and not section.winfo_exists():
                    continue
            except Exception:
                continue
            if tab_name is None or record.get("tab") == tab_name:
                active.append(record)
        return active

    def _export_figure_record(record, base_dir, index=None):
        fig = record["figure"]
        tab_name = record.get("tab") or "Plots"
        title = record.get("title") or _figure_title(fig)
        prefix_bits = []
        if index is not None:
            prefix_bits.append(f"{index:02d}")
        prefix_bits.extend([_safe_name(tab_name), _safe_name(title)])
        export_name = "_".join(prefix_bits)
        graph_dir = os.path.join(base_dir, export_name)
        os.makedirs(graph_dir, exist_ok=True)

        png_path = os.path.join(graph_dir, f"{export_name}.png")
        svg_path = os.path.join(graph_dir, f"{export_name}.svg")
        npz_path = os.path.join(graph_dir, f"{export_name}_data.npz")
        pickle_path = os.path.join(graph_dir, f"{export_name}_data.pkl")
        figure_pickle_path = os.path.join(graph_dir, f"{export_name}_figure.pkl")
        manifest_path = os.path.join(graph_dir, f"{export_name}_manifest.json")

        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        fig.savefig(svg_path, format="svg", bbox_inches="tight")

        current_figure_data = _extract_figure_data(fig)
        cached_artist_data = getattr(fig, "_waven_cached_artist_data", None)
        figure_data = cached_artist_data or current_figure_data
        payload = getattr(fig, "_waven_export_payload", None)
        bundle = {
            "tab": tab_name,
            "title": title,
            "figure_data": figure_data,
            "current_figure_data": current_figure_data,
            "source_artist_data": cached_artist_data,
            "payload": payload,
        }

        arrays = {}
        metadata = {}
        _add_array_exports("figure", figure_data, arrays, metadata)
        _add_array_exports("current_figure", current_figure_data, arrays, metadata)
        if cached_artist_data is not None:
            _add_array_exports("source_artist", cached_artist_data, arrays, metadata)
        _add_array_exports("payload", payload, arrays, metadata)
        np.savez_compressed(npz_path, **arrays)
        pickle_data_saved = True
        try:
            with open(pickle_path, "wb") as handle:
                pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            pickle_data_saved = False
            safe_bundle = {
                "tab": tab_name,
                "title": title,
                "figure_data": _json_safe(figure_data),
                "current_figure_data": _json_safe(current_figure_data),
                "source_artist_data": _json_safe(cached_artist_data),
                "payload": _json_safe(payload),
                "pickle_note": f"Original payload was not pickleable: {exc}",
            }
            with open(pickle_path, "wb") as handle:
                pickle.dump(safe_bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved summarized data pickle for '{title}' because the full payload was not pickleable: {exc}")

        figure_pickle_saved = False
        try:
            with open(figure_pickle_path, "wb") as handle:
                pickle.dump(fig, handle, protocol=pickle.HIGHEST_PROTOCOL)
            figure_pickle_saved = True
        except Exception as exc:
            print(f"Could not pickle matplotlib figure '{title}': {exc}")

        manifest = {
            "title": title,
            "tab": tab_name,
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "files": {
                "png": os.path.basename(png_path),
                "svg": os.path.basename(svg_path),
                "npz_data": os.path.basename(npz_path),
                "pickle_data": os.path.basename(pickle_path),
                "pickle_data_contains_full_payload": pickle_data_saved,
                "figure_pickle": os.path.basename(figure_pickle_path) if figure_pickle_saved else None,
            },
            "array_keys": sorted(arrays),
            "metadata": _json_safe(metadata),
            "axes": _json_safe(figure_data.get("axes", [])),
            "current_axes": _json_safe(current_figure_data.get("axes", [])),
            "source_axes": _json_safe(cached_artist_data.get("axes", [])) if cached_artist_data else None,
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return graph_dir

    def export_single_graph(record):
        title = record.get("title") or "plot"
        export_dir = filedialog.askdirectory(title=f"Select Folder for Export {title}")
        if not export_dir:
            return
        try:
            graph_dir = _export_figure_record(record, export_dir, index=1)
            print(f"Exported graph '{title}' with reusable data to: {graph_dir}")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not export {title}: {exc}")
            print(f"Failed to export graph '{title}': {exc}")

    def _export_displayed_results(tab_name=None):
        records = _active_export_records(tab_name=tab_name)
        if not records:
            label = tab_name or "displayed"
            messagebox.showinfo("No Results", f"No {label} results are available to export.")
            print(f"No {label} figures are available to export.")
            return
        title = f"Select Folder for {tab_name} Export" if tab_name else "Select Folder for Displayed Result Export"
        selected_dir = filedialog.askdirectory(title=title)
        if not selected_dir:
            return
        export_label = _safe_name(tab_name or "displayed_results")
        export_root = os.path.join(selected_dir, f"waven_{export_label}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(export_root, exist_ok=True)
        exported = []
        try:
            for index, record in enumerate(records, start=1):
                tab_dir = os.path.join(export_root, _safe_name(record.get("tab") or "Plots"))
                os.makedirs(tab_dir, exist_ok=True)
                exported.append(_export_figure_record(record, tab_dir, index=index))
            manifest = {
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "count": len(exported),
                "tab": tab_name,
                "graphs": exported,
            }
            with open(os.path.join(export_root, "export_manifest.json"), "w", encoding="utf-8") as handle:
                json.dump(_json_safe(manifest), handle, indent=2)
            print(f"Exported {len(exported)} displayed graph(s) with reusable data to: {export_root}")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not export displayed results: {exc}")
            print(f"Failed to export displayed results: {exc}")

    def export_all_displayed_results():
        _export_displayed_results()

    def export_all_neurons_results():
        _export_displayed_results("All neurons")

    def export_individual_neuron_results():
        _export_displayed_results("Individual neuron")

    def _render_figure_records(records, message="Loaded plots from cache.", clear=True):
        def render():
            if clear:
                clear_plot_tab(frame_plot_all)
                clear_plot_tab(frame_plot_individual)
                embedded_canvases.clear()
            for record in records or []:
                try:
                    if "png" in record:
                        image = plt.imread(io.BytesIO(record["png"]), format="png")
                        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
                        ax.imshow(image)
                        ax.axis("off")
                    else:
                        fig = pickle.loads(record["figure"])
                    if "export_payload" in record:
                        _set_figure_export_payload(fig, record.get("export_payload"))
                    if "export_artist_data" in record:
                        setattr(fig, "_waven_cached_artist_data", record.get("export_artist_data"))
                except Exception as exc:
                    print(f"Could not restore cached figure: {exc}")
                    continue
                parent = frame_plot_all if record.get("tab") == "all" else frame_plot_individual
                embed_interactive_figure(fig, parent, record.get("title"))
            print(message)
        root.after(0, render)

    def _state_for_plot_cache(state):
        excluded = {"wavelets_complex"}
        cached = {}
        for key, value in state.items():
            if key in excluded:
                continue
            cached[key] = value
        return cached

    def _get_cached_entry(kind, neuron_id=None, extra=None):
        cache = _load_plot_cache()
        entry = cache.get("entries", {}).get(_cache_key(kind, neuron_id))
        if not entry:
            return None
        if entry.get("fingerprint") != _cache_fingerprint(extra=extra):
            return None
        return entry

    def _put_cached_entry(kind, entry, neuron_id=None, extra=None):
        cache = _load_plot_cache()
        cache.setdefault("entries", {})[_cache_key(kind, neuron_id)] = {
            **entry,
            "fingerprint": _cache_fingerprint(extra=extra),
            "saved_at": time.time(),
        }
        _save_plot_cache(cache)

    def _start_recovery_checkpoint(task_name):
        root_dir = _recovery_root()
        os.makedirs(root_dir, exist_ok=True)
        task_dir = os.path.join(root_dir, _safe_name(task_name))
        os.makedirs(task_dir, exist_ok=True)
        active_recovery_dir["path"] = task_dir
        manifest = {
            "task": task_name,
            "status": "running",
            "started_at": time.time(),
            "cwd": os.getcwd(),
        }
        with open(os.path.join(task_dir, "manifest.json"), "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        return task_dir

    def _write_recovery_step(step, **data):
        task_dir = active_recovery_dir.get("path")
        if not task_dir:
            return
        payload = {"step": step, "time": time.time(), **data}
        with open(os.path.join(task_dir, f"{_safe_name(step)}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _recovery_subdir(name):
        task_dir = active_recovery_dir.get("path") or _recovery_root()
        path = os.path.join(task_dir, _safe_name(name))
        os.makedirs(path, exist_ok=True)
        return path

    def _finish_recovery_checkpoint(success):
        task_dir = active_recovery_dir.get("path")
        active_recovery_dir["path"] = None
        if not task_dir:
            return
        manifest_path = os.path.join(task_dir, "manifest.json")
        try:
            manifest = {}
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            manifest["status"] = "complete" if success else "failed"
            manifest["finished_at"] = time.time()
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
        except Exception:
            pass
        if success:
            try:
                shutil.rmtree(task_dir)
                print(f"Removed successful recovery checkpoint: {task_dir}")
            except Exception as exc:
                print(f"Could not remove successful recovery checkpoint {task_dir}: {exc}")
        else:
            print(f"Kept recovery checkpoint after failure: {task_dir}")

    def create_gabor():
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        nx = int(gabor_entries["NX"].get())
        ny = int(gabor_entries["NY"].get())
        n_theta = int(gabor_entries["N_thetas"].get())
        offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
        path_save = gabor_entries["Save Path"].get()
        xs = np.arange(nx)
        ys = np.arange(ny)
        thetas = np.array([(i * np.pi) / n_theta for i in range(n_theta)])
        sigmas = np.array(sigmas)
        offsets = np.array(offsets)
        frequencies = np.array(frequencies)

        update_progress(5, "Building Gabor library", "Estimating filter bank")
        if frequencies.size and np.any(frequencies != 0):
            L = makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies)
        else:
            frequency = frequencies[0] if frequencies.size else 0
            L = makeFilterLibrary(xs, ys, thetas, sigmas, offsets, frequency, freq=False)

        update_progress(70, "Saving library", "Writing output file")
        os.makedirs(os.path.dirname(path_save) or ".", exist_ok=True)
        if gabor_format_var.get() == "zarr":
            output_path = os.path.splitext(path_save)[0] + ".zarr"
            print(f"Saving Gabor library into Zarr container: {output_path}")
            zarr.save(output_path, L)
        else:
            output_path = path_save
            np.save(output_path, L)
            print(f"Gabor library saved to: {output_path}")
        if "Library Path" in param_entries:
            param_entries["Library Path"].delete(0, tk.END)
            param_entries["Library Path"].insert(0, output_path)
            refresh_size_estimates()
        update_progress(100, "Gabor library complete")

    def run_wavelet():
        movpath = param_entries["Movie Path"].get().strip()
        if not movpath:
            print("Error: Movie Path is required.")
            return False

        parent_dir = os.path.dirname(movpath) or "."
        output_format = wavelet_format_var.get()
        is_zarr_wavelet = output_format == "zarr"
        wavelet_folder = parent_dir

        current_wavelet_dir[0] = wavelet_folder

        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        lib_path = param_entries["Library Path"].get()
        nx = int(param_entries["NX"].get())
        ny = int(param_entries["NY"].get())
        n_thetas = int(gabor_entries["N_thetas"].get())

        coarse_cache_present = os.path.exists(
            os.path.join(wavelet_folder, "dwt_downsampled_videodata.npy")
        )
        highres_present = (
            os.path.exists(os.path.join(wavelet_folder, "dwt_videodata_0.npy"))
            and os.path.exists(os.path.join(wavelet_folder, "dwt_videodata_1.npy"))
        )
        if coarse_cache_present:
            print(f"Found existing coarse RF wavelet output in {wavelet_folder}. Reusing it.")
        elif highres_present:
            print(f"Found existing high-resolution wavelets in {wavelet_folder}. Skipping decomposition.")
        else:
            print("Step 1/2: Downsampling stimulus movie and real phase decomposition...")
            _write_recovery_step("wavelet_downsampling_started", wavelet_folder=wavelet_folder)
            visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
            analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")

            if visual_coverage != analysis_coverage:
                visual_coverage = np.array(visual_coverage)
                analysis_coverage = np.array(analysis_coverage)
                ratio_x = 1 - ((visual_coverage[0] - visual_coverage[1]) - (analysis_coverage[0] - analysis_coverage[1])) / (visual_coverage[0] - visual_coverage[1])
                ratio_y = 1 - ((visual_coverage[2] - visual_coverage[3]) - (analysis_coverage[2] - analysis_coverage[3])) / (visual_coverage[2] - visual_coverage[3])
            else:
                ratio_x = ratio_y = 1

            downsample_video_binary(
                movpath,
                visual_coverage,
                analysis_coverage,
                shape=(ny, nx),
                chunk_size=video_downsample_chunk_size(),
                ratios=(ratio_x, ratio_y),
            )
            videodata = np.load(movpath[:-4] + '_downsampled.npy')
            videodata = videodata.astype(int) - np.logical_not(videodata).astype(int)

            waveletDecomposition(videodata, 0, sigmas, wavelet_folder, lib_path)
            _write_recovery_step("coarse_phase_real_complete", path=os.path.join(wavelet_folder, "dwt_videodata_0.npy"))
            print("Step 2/2: Wavelet decomposition (imaginary phase)...")
            waveletDecomposition(videodata, 1, sigmas, wavelet_folder, lib_path)
            _write_recovery_step("coarse_phase_imaginary_complete", path=os.path.join(wavelet_folder, "dwt_videodata_1.npy"))

        coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
        print(
            f"Coarse RF grid derived from config: {coarse_nx} x {coarse_ny} "
            f"(20% of {nx} x {ny})"
        )

        full_output_target = param_entries["Full Model Wavelet Path"].get().strip()
        if not full_output_target:
            full_output_target = parent_dir
        if is_zarr_wavelet:
            full_work_dir = _recovery_subdir("full_model_npy_before_zarr")
            full_output = full_work_dir
            os.makedirs(full_output_target, exist_ok=True)
            print(f"Using recovery-backed full-model NPY folder before Zarr conversion: {full_output}")
        else:
            full_output = full_output_target
            os.makedirs(full_output, exist_ok=True)

        print("Step 3: Generating coarse wavelet cache for RF analysis...")
        lpn.coarseWavelet(
            wavelet_folder,
            False,
            nx0=nx,
            ny0=ny,
            no=n_thetas,
            ns=len(sigmas),
            nf=1,
            nx=coarse_nx,
            ny=coarse_ny,
            chunk_size=None,
        )
        _write_recovery_step(
            "coarse_cache_complete",
            path=os.path.join(wavelet_folder, "dwt_downsampled_videodata.npy"),
        )
        for intermediate_name in ("dwt_videodata_0.npy", "dwt_videodata_1.npy"):
            intermediate_path = os.path.join(wavelet_folder, intermediate_name)
            try:
                if os.path.exists(intermediate_path):
                    os.remove(intermediate_path)
                    print(f"Removed intermediate coarse phase file: {intermediate_path}")
            except Exception as exc:
                print(f"Could not remove intermediate file {intermediate_path}: {exc}")

        sigmas_full = parse_literal(
            param_entries["Sigmas Full Model"].get(),
            "Sigmas Full Model",
        )
        print("Step 4: Generating full-resolution wavelets for run_Full_Model...")
        videodata = np.load(movpath[:-4] + '_downsampled.npy')
        videodata = videodata.astype(int) - np.logical_not(videodata).astype(int)
        for phase in (0, 1):
            suffix = "_r" if phase == 0 else "_i"
            target = os.path.join(full_output, f"dwt_videodata2{suffix}.npy")
            if os.path.exists(target):
                print(f"Found existing full-model wavelets: {target}")
                continue
            waveletDecompositionFull(
                videodata,
                phase,
                sigmas_full,
                frequencies,
                full_output,
                lib_path,
                library_sigmas=sigmas,
            )
            _write_recovery_step(f"full_model_phase_{phase}_complete", path=target)

        if is_zarr_wavelet:
            print("Step 5: Converting full-model wavelets to compressed Zarr...")
            convert_npy_to_zarr(
                full_output,
                full_output_target,
                int(param_entries["Hz"].get()),
                n_orientations=n_thetas,
                n_sigmas=len(sigmas_full),
                n_frequencies=len(frequencies),
            )
            _write_recovery_step("full_model_zarr_complete", path=full_output_target)
            try:
                shutil.rmtree(full_output)
                print(f"Removed temporary full-model NPY conversion folder: {full_output}")
            except Exception as exc:
                print(f"Could not remove temporary full-model folder {full_output}: {exc}")
            print(
                f"All wavelet files are ready. Coarse NPY cache: {wavelet_folder} | "
                f"Full-model Zarr: {full_output_target}"
            )
        else:
            print(f"All wavelet files are ready. Coarse: {wavelet_folder} | Full-model NPY: {full_output}")
    
    def embed_interactive_figure(fig, parent_container, title=None):
        """Embed a matplotlib figure with navigation toolbar in ``parent_container``."""
        section = ctk.CTkFrame(parent_container, corner_radius=8, fg_color="#FFFFFF")
        section.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=8, padx=8)
        graph_title = title or _figure_title(fig)
        header = ctk.CTkFrame(section, fg_color="transparent")
        header.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(8, 2))
        if graph_title:
            ctk.CTkLabel(
                header,
                text=graph_title,
                text_color="#1F2937",
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w",
            ).pack(side=tk.LEFT, fill=tk.X, expand=True)

        canvas = FigureCanvasTkAgg(fig, master=section)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, section)
        toolbar.update()

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        embedded_canvases.append(canvas)
        record = {
            "canvas": canvas,
            "figure": fig,
            "title": graph_title,
            "tab": _tab_name_for_parent(parent_container),
            "section": section,
        }
        figure_export_records.append(record)
        ctk.CTkButton(
            header,
            text=f"Export {graph_title}",
            width=150,
            height=28,
            corner_radius=6,
            fg_color="#4B5563",
            hover_color="#374151",
            command=lambda r=record: export_single_graph(r),
        ).pack(side=tk.RIGHT, padx=(10, 0))
        return canvas

    def clear_plot_tab(parent):
        tab_name = _tab_name_for_parent(parent)
        for widget in parent.winfo_children():
            widget.destroy()
        figure_export_records[:] = [
            record for record in figure_export_records
            if record.get("tab") != tab_name
        ]

    def switch_to_individual_tab(flash=True):
        try:
            plot_tabs.set("Individual neuron")
        except Exception:
            pass
        if flash:
            try:
                individual_update_label.configure(
                    text=f"Updated neuron {param_entries['Neuron ID'].get()}",
                    fg_color="#F97316",
                    text_color="#111827",
                )
                root.after(
                    1400,
                    lambda: individual_update_label.configure(
                        fg_color="#FFF7ED",
                        text_color="#9A3412",
                    ),
                )
            except Exception:
                pass

    def embed_captured_figures(figures, parent, title_prefix):
        if not figures:
            ctk.CTkLabel(
                parent,
                text="No matplotlib figures were produced for this run.",
                text_color="#6B7280",
            ).pack(anchor="w", padx=12, pady=12)
            return
        for index, fig in enumerate(figures, start=1):
            embed_interactive_figure(fig, parent, f"{title_prefix} {index}")

    def capture_new_figures(callback):
        before = set(plt.get_fignums())
        result = callback()
        after = set(plt.get_fignums())
        figures = [plt.figure(num) for num in sorted(after - before)]
        return result, figures

    def gui_trailing_sep(path):
        path = str(path)
        if path.endswith(("/", "\\")):
            return path
        return path + os.sep
    
    def plot_data():
        rf_extra = {"selected_neuron": _field_value(param_entries, "Neuron ID", "")}
        cached = _get_cached_entry("coarse_rf", extra=rf_extra)
        if cached:
            cached_state = cached.get("analysis_state")
            if isinstance(cached_state, dict):
                analysis_state.clear()
                analysis_state.update(cached_state)
                current_wavelet_dir[0] = cached_state.get("wavelet_dir", current_wavelet_dir[0])
            _render_figure_records(
                cached.get("figures"),
                message=f"Loaded coarse RF plots from cache: {_plot_cache_path()}",
            )
            return

        try:
            data_dirs = _parse_data_dir(param_entries["Dir"].get())
            exp_info = parse_literal(param_entries["Experiment Info"].get(), "Experiment Info")
            sigmas = np.array(parse_literal(param_entries["Sigmas"].get(), "Sigmas"))
            frequencies = np.array(parse_literal(gabor_entries["Frequencies"].get(), "Frequencies"))
            nf = len(frequencies)
            visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
            analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")
            block_end = int(param_entries["Block End"].get())
            nx = int(param_entries["NX"].get())
            ny = int(param_entries["NY"].get())
            coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
            n_orientations = int(gabor_entries["N_thetas"].get())
            ns = len(sigmas)
            spks_path = param_entries["Spks Path"].get()
            nb_frames = int(param_entries["Number of Frames"].get())
            movpath = param_entries["Movie Path"].get()
            screen_ratio = abs(visual_coverage[0] - visual_coverage[1]) / nx
            xM, xm, yM, ym = analysis_coverage
            if workflow == WORKFLOW_2P:
                n_planes = int(param_entries["Number of Planes"].get())
                resolution = float(param_entries["Resolution"].get())
                sampling_rate = None
            else:
                n_planes = None
                resolution = None
                sampling_rate = float(
                    param_entries["Sampling Rate (samples / sec)"].get()
                )
        except Exception as e:
            print(f"Invalid input: {e}")
            return False

        pathdata = os.path.join(data_dirs[0], exp_info[0], exp_info[1], str(exp_info[2]))
        pathsuite2p = os.path.join(pathdata, 'suite2p')
        deg_per_pix = abs(xM - xm) / nx
        sigmas_deg = np.trunc(2 * deg_per_pix * sigmas * 100) / 100

        if spks_path.strip().lower() in ("", "none", "null"):
            from . import time_alignment as ta

            try:
                aligned = ta.load_aligned_spikes(
                    workflow,
                    experiment_info=exp_info,
                    data_dir=Path(data_dirs[0]),
                    data_dir_strings=data_dirs,
                    suite2p_dir=Path(pathsuite2p),
                    block_end=block_end,
                    n_planes=n_planes,
                    nb_frames=nb_frames,
                    resolution=resolution,
                    sampling_rate=sampling_rate,
                    threshold=1.25,
                    method='frame2ttl',
                )
            except NotImplementedError as exc:
                print(exc)
                return False
            spks = aligned.spikes
            neuron_pos = aligned.neuron_pos
            if workflow == WORKFLOW_2P:
                neuron_pos[:, 1] = abs(neuron_pos[:, 1] - np.max(neuron_pos[:, 1]))
        else:
            try:
                spks = np.load(spks_path)
                parent_dir = os.path.dirname(spks_path)
                neuron_pos = np.load(os.path.join(parent_dir, 'pos.npy'))
            except Exception as e:
                print(f"File not found: {e}")
                return False

        print("Loading neural data and coarse wavelets...")
        _write_recovery_step("loading_neural_data")
        respcorr = repetability_trial3(spks, neuron_pos, plotting=False)
        skewness = np.array(compute_skewness_neurons(spks, plotting=False))
        filter_mask = np.logical_and(respcorr >= 0.2, skewness <= 20)

        point_alphas = np.where(filter_mask, 1.0, 0.05)

        parent_dir = current_wavelet_dir[0] or os.path.dirname(movpath)
        try:
            wavelets_downsampled = np.load(os.path.join(parent_dir, 'dwt_downsampled_videodata.npy'))
            w_c_downsampled = wavelets_downsampled[2]
            del wavelets_downsampled
            gc.collect()
        except Exception as e:
            print(f"Decomposition loading failed: {e}")
            return False
            
        n_frames = min(nb_frames, w_c_downsampled.shape[0], spks.shape[1])

        if w_c_downsampled.ndim == 6:
            rf_nf = w_c_downsampled.shape[5]
            rf_frequencies = frequencies[:rf_nf]
        else:
            rf_nf = 1
            rf_frequencies = frequencies[:1]

        rfs_gabor = PearsonCorrelationPinkNoise(w_c_downsampled[:n_frames].reshape(n_frames, -1),
                                                np.mean(spks[:, :n_frames], axis=0),
                                                neuron_pos, coarse_nx, coarse_ny, ns, rf_nf, analysis_coverage, screen_ratio, sigmas_deg, rf_frequencies,
                                                n_orientations=n_orientations,
                                                plotting=False)
        _write_recovery_step("coarse_rf_complete", wavelet_dir=parent_dir)
        analysis_state.clear()
        analysis_state.update(
            spks=spks,
            neuron_pos=neuron_pos,
            rfs_gabor=rfs_gabor,
            wavelets_complex=w_c_downsampled,
            sigmas=sigmas,
            sigmas_deg=sigmas_deg,
            frequencies=frequencies,
            rf_frequencies=rf_frequencies,
            analysis_coverage=analysis_coverage,
            visual_coverage=visual_coverage,
            screen_ratio=screen_ratio,
            nx=nx,
            ny=ny,
            coarse_nx=coarse_nx,
            coarse_ny=coarse_ny,
            n_orientations=n_orientations,
            nb_frames=nb_frames,
            wavelet_dir=parent_dir,
        )

        def render_gui_plots():
            clear_plot_tab(frame_plot_all)
            clear_plot_tab(frame_plot_individual)
            embedded_canvases.clear()
            plt.close('all')

            fig1, ax1 = plt.subplots(figsize=(6, 5), constrained_layout=True)
            ax1.scatter(neuron_pos[:, 0], neuron_pos[:, 1], c='k', alpha=0.3, label="Neurons", picker=True, rasterized=True)
            ax1.set_title("Neuron Positions (µm)")
            ax1.set_xlabel("X (µm)")
            ax1.set_ylabel("Y (µm)")

            fig10, axes10 = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
            ax10 = axes10.ravel()
            maxes1 = rfs_gabor[2]
            plt.rcParams['axes.facecolor'] = 'none'

            map_specs = [
                (0, maxes1[0], 'jet', 'Azimuth (°)'),
                (1, maxes1[1], 'jet_r', 'Elevation (°)'),
                (2, maxes1[2], 'hsv', 'Orientation (°)'),
                (3, maxes1[3], 'coolwarm', 'Size (°)'),
            ]
            for idx, values, cmap, title in map_specs:
                scatter = ax10[idx].scatter(
                    neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=values,
                    cmap=cmap, alpha=point_alphas, rasterized=True, picker=True,
                )
                fig10.colorbar(scatter, ax=ax10[idx], fraction=0.046)
                ax10[idx].set_title(title)
                ax10[idx].set_xlabel("X (µm)")
                ax10[idx].set_ylabel("Y (µm)")

            fig2, ax2 = plt.subplots(figsize=(10, 2.5), constrained_layout=True)
            ax2.set_title("Trial-averaged Spike Train")

            fig3 = plt.figure(figsize=(8, 11), constrained_layout=True)
            gs = fig3.add_gridspec(4, 2)
            ax3_0 = fig3.add_subplot(gs[0, :])
            ax3_1 = fig3.add_subplot(gs[1, 0])
            ax3_2 = fig3.add_subplot(gs[1, 1])
            ax3_3 = fig3.add_subplot(gs[2, 0])
            ax3_4 = fig3.add_subplot(gs[2, 1])
            ax3_5 = fig3.add_subplot(gs[3, :])
            ax3 = [ax3_0, ax3_1, ax3_2, ax3_3, ax3_4, ax3_5]

            def draw_individual_neuron(neuron_id):
                try:
                    entry_neuron.delete(0, tk.END)
                    entry_neuron.insert(0, str(neuron_id))

                    spike_train = np.mean(spks[:, :, neuron_id], axis=0)
                    ax2.clear()
                    ax2.plot(spike_train, label=f"Neuron {neuron_id} Spike Times")
                    ax2.legend()
                    _set_figure_export_payload(
                        fig2,
                        {
                            "source": "Inspect Single Neuron",
                            "neuron_id": neuron_id,
                            "spike_train": spike_train,
                            "trial_spikes": spks[:, :, neuron_id],
                        },
                    )
                    canvas2.draw()

                    rf2d, x_tuning, y_tuning, ori_tun, s_tuning, f_tuning = PlotTuningCurve(rfs_gabor, neuron_id, analysis_coverage, sigmas_deg, screen_ratio, frequencies, show=False)
                    for ax in ax3: ax.clear()

                    heat = ax3[0].imshow(rf2d, cmap='coolwarm', aspect='equal')
                    ax3[0].set_xticks(
                        np.linspace(0, rf2d.shape[1] - 1, 3),
                        np.round(np.linspace(xM, xm, 3), 2),
                    )
                    ax3[0].set_yticks(
                        np.linspace(0, rf2d.shape[0] - 1, 3),
                        np.round(np.linspace(yM, ym, 3), 2),
                    )
                    ax3[0].set_title('Receptive Field')
                    ax3[0].set_xlabel("Azimuth (deg)")
                    ax3[0].set_ylabel("Elevation (deg)")
                    ax3[1].plot(x_tuning[::-1], c='k')
                    ax3[1].set_title('Elevation (deg)')
                    ax3[1].set_xticks([0, rf2d.shape[0]], [ym, yM])
                    ax3[2].plot(y_tuning, c='k')
                    ax3[2].set_title('Azimuth')
                    ax3[2].set_xticks([0, rf2d.shape[1]], [xM, xm])
                    ax3[3].plot(ori_tun, 'o-', c='k')
                    ax3[3].set_title('Orientation')
                    n_ori = rfs_gabor[0].shape[3]
                    ax3[3].set_xticks(
                        [0, max(1, n_ori // 2), max(2, n_ori - 1)],
                        [0, 90, 180],
                    )
                    ax3[4].plot(s_tuning, 'o-', c='k')
                    ax3[4].set_title('Size (deg)')
                    ax3[4].set_xticks([0, len(sigmas) - 1], [sigmas_deg[0], sigmas_deg[-1]])
                    ax3[5].plot(f_tuning, 'o-', c='k')
                    ax3[5].set_title('Spatial Frequency')
                    ax3[5].set_xticks(range(len(frequencies)), [round(f, 3) for f in frequencies])
                    _set_figure_export_payload(
                        fig3,
                        {
                            "source": "Inspect Single Neuron",
                            "neuron_id": neuron_id,
                            "rf2d": rf2d,
                            "x_tuning": x_tuning,
                            "y_tuning": y_tuning,
                            "orientation_tuning": ori_tun,
                            "size_tuning": s_tuning,
                            "frequency_tuning": f_tuning,
                            "best_params": np.asarray(rfs_gabor[1])[:, neuron_id],
                            "retinotopy": np.asarray(rfs_gabor[2])[:, neuron_id],
                        },
                    )
                    canvas3.draw()
                    switch_to_individual_tab(flash=True)
                except Exception as e:
                    print(f"Error drawing selected neuron: {e}")

            def onpick(event):
                try:
                    draw_individual_neuron(int(event.ind[0]))
                except Exception as e:
                    print(f"Error drawing pick event: {e}")

            fig1.canvas.mpl_connect('pick_event', onpick)
            fig10.canvas.mpl_connect('pick_event', onpick)

            global canvas2, canvas3

            _set_figure_export_payload(
                fig1,
                {
                    "source": "Run Coarse RF Analysis",
                    "neuron_pos": neuron_pos,
                    "response_correlation": respcorr,
                    "skewness": skewness,
                    "filter_mask": filter_mask,
                },
            )
            _set_figure_export_payload(
                fig10,
                {
                    "source": "Run Coarse RF Analysis",
                    "retinotopy": np.asarray(rfs_gabor[2]),
                    "best_params": np.asarray(rfs_gabor[1]),
                    "neuron_pos": neuron_pos,
                    "sigmas_deg": sigmas_deg,
                    "frequencies": rf_frequencies,
                    "analysis_coverage": analysis_coverage,
                    "visual_coverage": visual_coverage,
                },
            )
            embed_interactive_figure(fig1, frame_plot_all, title="Neuron Layout")
            embed_interactive_figure(fig10, frame_plot_all, title="Population Retinotopy Maps")
            canvas2 = embed_interactive_figure(fig2, frame_plot_individual, title="Spike Train")
            canvas3 = embed_interactive_figure(fig3, frame_plot_individual, title="Selected Neuron Tuning")

            def click_RF():
                try:
                    neuron_id = int(param_entries["Neuron ID"].get())
                    draw_individual_neuron(neuron_id)
                except Exception as e:
                    print(f"Failed to plot RF: {e}")

            btn_runRF.configure(command=click_RF)
            try:
                draw_individual_neuron(int(param_entries["Neuron ID"].get()))
            except Exception:
                pass
            _put_cached_entry(
                "coarse_rf",
                {
                    "analysis_state": _state_for_plot_cache(analysis_state),
                    "figures": _figure_records(
                        [
                            ("all", "Neuron Layout", fig1),
                            ("all", "Population Retinotopy Maps", fig10),
                            ("individual", "Spike Train", fig2),
                            ("individual", "Selected Neuron Tuning", fig3),
                        ]
                    ),
                },
                extra=rf_extra,
            )
            print("Plots rendered successfully.")

        root.after(0, render_gui_plots)

    def _selected_neuron_id():
        neuron_id = int(param_entries["Neuron ID"].get())
        if "spks" in analysis_state and not (0 <= neuron_id < analysis_state["spks"].shape[2]):
            raise ValueError(f"Neuron ID {neuron_id} is outside the loaded range.")
        return neuron_id

    def _require_rf_state():
        if not analysis_state:
            raise RuntimeError("Run Coarse RF Analysis before model plotting.")
        return analysis_state

    def _append_model_figures(figures, title_prefix):
        def render():
            switch_to_individual_tab(flash=True)
            embed_captured_figures(figures, frame_plot_individual, title_prefix)
        root.after(0, render)

    def plot_run_model_outputs():
        state = _require_rf_state()
        neuron_id = _selected_neuron_id()
        cache_extra = {"model": "run_Model", "neuron": neuron_id}
        cached = _get_cached_entry("run_model", neuron_id=neuron_id, extra=cache_extra)
        if cached:
            _render_figure_records(
                cached.get("figures"),
                message=f"Loaded run_Model plots for neuron {neuron_id} from cache.",
                clear=False,
            )
            return
        wavelet_dir = state["wavelet_dir"]
        wavelets_downsampled = np.load(
            os.path.join(wavelet_dir, "dwt_downsampled_videodata.npy"),
            mmap_mode="r",
        )
        w_r = wavelets_downsampled[0]
        w_i = wavelets_downsampled[1]
        raw_best_params = np.array(state["rfs_gabor"][1])
        smoothed_best_params = smooth_best_positions(
            raw_best_params,
            state["neuron_pos"],
        )
        dt1 = min(int(param_entries["Number of Frames"].get()), state["spks"].shape[1])
        frames_per_minute = int(param_entries["Hz"].get()) * 60

        def call_model():
            return run_Model(
                smoothed_best_params[:, [neuron_id]],
                raw_best_params[:, [neuron_id]],
                state["spks"][:, :, [neuron_id]],
                w_i,
                w_r,
                dt1=dt1,
                n_min=5,
                double_wavelet_model=False,
                plotting=True,
                frames_per_minute=frames_per_minute,
            )

        result, figures = capture_new_figures(call_model)
        del wavelets_downsampled
        gc.collect()
        model_payload = _model_result_payload("run_Model", result, neuron_id)
        for fig in figures:
            _set_figure_export_payload(fig, model_payload)
        _put_cached_entry(
            "run_model",
            {
                "figures": _figure_records(
                    [("individual", f"run_Model neuron {neuron_id} {i}", fig) for i, fig in enumerate(figures, start=1)]
                )
            },
            neuron_id=neuron_id,
            extra=cache_extra,
        )
        _append_model_figures(figures, f"run_Model neuron {neuron_id}")

    def plot_run_full_model_outputs():
        state = _require_rf_state()
        neuron_id = _selected_neuron_id()
        cache_extra = {"model": "run_Full_Model", "neuron": neuron_id}
        cached = _get_cached_entry("run_full_model", neuron_id=neuron_id, extra=cache_extra)
        if cached:
            _render_figure_records(
                cached.get("figures"),
                message=f"Loaded run_Full_Model plots for neuron {neuron_id} from cache.",
                clear=False,
            )
            return
        raw_best_params = np.array(state["rfs_gabor"][1])
        smoothed_best_params = smooth_best_positions(
            raw_best_params,
            state["neuron_pos"],
        )
        sigmas_full = np.array(parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"))
        frequencies = np.array(parse_literal(gabor_entries["Frequencies"].get(), "Frequencies"))
        wavelet_path = param_entries["Full Model Wavelet Path"].get().strip() or state["wavelet_dir"]
        save_path = param_entries["Full Model Save Path"].get().strip() or state["wavelet_dir"]
        os.makedirs(save_path, exist_ok=True)
        frames_per_minute = int(param_entries["Hz"].get()) * 60

        def call_full_model():
            return run_Full_Model(
                raw_best_params,
                smoothed_best_params,
                state["spks"],
                [neuron_id],
                np.array([(i * np.pi) / state["n_orientations"] for i in range(state["n_orientations"])]),
                sigmas_full,
                frequencies,
                state["visual_coverage"],
                state["neuron_pos"],
                wavelet_path=gui_trailing_sep(wavelet_path),
                savepath=gui_trailing_sep(save_path),
                n_min=5,
                tt=[0, min(state["nb_frames"], state["spks"].shape[1])],
                memmapping=True,
                train_idx=[0, 2],
                test_idx=[1, 3],
                double_wavelet_model=False,
                lastmin=False,
                plotting=True,
                frames_per_minute=frames_per_minute,
                hz=int(param_entries["Hz"].get()),
            )

        result, figures = capture_new_figures(call_full_model)
        model_payload = _model_result_payload("run_Full_Model", result, neuron_id)
        for fig in figures:
            _set_figure_export_payload(fig, model_payload)
        _put_cached_entry(
            "run_full_model",
            {
                "figures": _figure_records(
                    [("individual", f"run_Full_Model neuron {neuron_id} {i}", fig) for i, fig in enumerate(figures, start=1)]
                )
            },
            neuron_id=neuron_id,
            extra=cache_extra,
        )
        _append_model_figures(figures, f"run_Full_Model neuron {neuron_id}")

    def click_save():
        try:
            state = _require_rf_state()
        except Exception as exc:
            messagebox.showinfo("No Retinotopy Data", str(exc))
            print(f"Retinotopy export skipped: {exc}")
            return
        path = filedialog.asksaveasfilename(
            title="Export Retinotopy Matrix",
            defaultextension=".npy",
            filetypes=[("NumPy array", "*.npy"), ("Compressed NumPy archive", "*.npz")],
            initialfile="retinotopy_matrix.npy",
        )
        if not path:
            return
        try:
            retinotopy = np.asarray(state["rfs_gabor"][2])
            if path.lower().endswith(".npz"):
                np.savez_compressed(
                    path,
                    retinotopy=retinotopy,
                    best_params=np.asarray(state["rfs_gabor"][1]),
                    neuron_pos=np.asarray(state["neuron_pos"]),
                )
            else:
                np.save(path, retinotopy)
            print(f"Exported retinotopy matrix to: {path}")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not export retinotopy matrix: {exc}")
            print(f"Failed to export retinotopy matrix: {exc}")

    def export_plots():
        if not embedded_canvases:
            messagebox.showinfo("No Plots", "Run an analysis before exporting plots.")
            print("No embedded plots are available to export.")
            return
        export_dir = filedialog.askdirectory(title="Select Folder for SVG Export")
        if not export_dir:
            return
        exported = 0
        try:
            for index, canvas in enumerate(embedded_canvases, start=1):
                fig = getattr(canvas, "figure", None)
                if fig is None:
                    continue
                title = ""
                try:
                    axes_titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
                    title = axes_titles[0] if axes_titles else f"plot_{index:02d}"
                except Exception:
                    title = f"plot_{index:02d}"
                filename = f"{index:02d}_{_safe_name(title)}.svg"
                fig.savefig(os.path.join(export_dir, filename), format="svg", bbox_inches="tight")
                exported += 1
            print(f"Exported {exported} SVG plot(s) to: {export_dir}")
        except Exception as exc:
            messagebox.showerror("Export Failed", f"Could not export SVG plots: {exc}")
            print(f"Failed to export SVG plots: {exc}")

    def save_app_state():
        state = {
            "workflow": workflow,
            "gabor": {key: entry.get() for key, entry in gabor_entries.items()},
            "analysis": {key: entry.get() for key, entry in param_entries.items()},
            "save_options": {
                "gabor_format": gabor_format_var.get(),
                "wavelet_format": wavelet_format_var.get(),
            },
        }
        path = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="waven_configuration.json",
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(state, handle, indent=2)
            print(f"Saved GUI state to: {path}")
        except Exception as exc:
            messagebox.showerror("Save Failed", f"Could not save GUI state: {exc}")
            print(f"Failed to save GUI state: {exc}")

    def load_app_state():
        path = filedialog.askopenfilename(
            title="Load Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            if state.get("workflow") and state["workflow"] != workflow:
                print(
                    f"Warning: loaded workflow {state['workflow']} does not match current workflow {workflow}."
                )
            for key, value in state.get("gabor", {}).items():
                if key in gabor_entries:
                    gabor_entries[key].delete(0, tk.END)
                    gabor_entries[key].insert(0, str(value))
            for key, value in state.get("analysis", {}).items():
                if key in param_entries:
                    param_entries[key].delete(0, tk.END)
                    param_entries[key].insert(0, str(value))
            save_options = state.get("save_options", {})
            gabor_format_var.set(save_options.get("gabor_format", gabor_format_var.get()))
            wavelet_format_var.set(save_options.get("wavelet_format", wavelet_format_var.get()))
            refresh_size_estimates()
            print(f"Loaded GUI state from: {path}")
        except Exception as exc:
            messagebox.showerror("Load Failed", f"Could not load GUI state: {exc}")
            print(f"Failed to load GUI state: {exc}")


    def cleanup_temporary_directories():
        for temp_dir in list(temp_directories):
            if os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Removed temporary folder: {temp_dir}")
                except Exception as exc:
                    print(f"Could not remove temporary folder {temp_dir}: {exc}")
        temp_directories.clear()

    def quit_app():
        cleanup_temporary_directories()
        root.quit()
        root.destroy()

    def browse_path(entry_widget, kind):
        """Open a file or directory picker appropriate for the field type."""
        if kind == "file":
            path = filedialog.askopenfilename(title="Select file")
        elif kind == "savefile":
            path = filedialog.asksaveasfilename(title="Select output file")
        else:
            path = filedialog.askdirectory(title="Select directory")
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)
            refresh_size_estimates()

    def estimate_gabor_library_size():
        try:
            nx = int(gabor_entries["NX"].get())
            ny = int(gabor_entries["NY"].get())
            n_theta = int(gabor_entries["N_thetas"].get())
            sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
            offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
            frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
            sigmas = np.asarray(sigmas)
            offsets = np.asarray(offsets)
            frequencies = np.asarray(frequencies)
            if frequencies.size and np.any(frequencies != 0):
                num_f = len(frequencies)
                ndim_label = "frequency bank"
            else:
                num_f = 1
                ndim_label = "single-frequency bank"
            sample = makeGaborFilter(
                0,
                0,
                0,
                sigmas[0] if sigmas.size else 1,
                offsets[0] if offsets.size else 0,
                frequencies[0] if frequencies.size else 0,
                lx=nx,
                ly=ny,
                freq=True,
            ).astype(np.float16, copy=False).ravel()
            flat_size = sample.size
            total_bytes = nx * ny * n_theta * max(1, len(sigmas)) * num_f * max(1, len(offsets)) * flat_size * np.dtype(np.float16).itemsize
            if gabor_format_var.get() == "zarr":
                zarr_path = _zarr_output_path(gabor_entries["Save Path"].get())
                exact_bytes = _folder_size_bytes(zarr_path)
                if exact_bytes is not None:
                    gabor_size_label.configure(text=f"Current Zarr size: {_format_bytes(exact_bytes)}")
                else:
                    zarr_bytes = _estimate_blosc_bytes(total_bytes, sample)
                    gabor_size_label.configure(
                        text=f"Estimated Zarr disk size: {_format_bytes(zarr_bytes)} ({ndim_label})"
                    )
            else:
                gabor_size_label.configure(text=f"Estimated NPY disk size: {_format_bytes(total_bytes)} ({ndim_label})")
        except Exception:
            gabor_size_label.configure(text="Estimated disk size: 0.00 GB")

    def estimate_wavelet_size():
        try:
            n_frames_entry = param_entries.get("Number of Frames")
            n_frames = int(n_frames_entry.get()) if n_frames_entry is not None else 0
            nx = int(param_entries["NX"].get())
            ny = int(param_entries["NY"].get())
            n_thetas = int(gabor_entries["N_thetas"].get())
            sigmas = parse_literal(param_entries["Sigmas"].get(), "Sigmas")
            sigmas_full = parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model")
            frequencies_entry = param_entries.get("Frequencies")
            frequencies = parse_literal(
                frequencies_entry.get() if frequencies_entry is not None else gabor_entries["Frequencies"].get(),
                "Frequencies",
            )
            n_sigmas = len(sigmas)
            n_sigmas_full = len(sigmas_full)
            n_frequencies = len(frequencies)
            coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
            bytes_per_float = np.dtype(np.float32).itemsize
            highres_coarse_bytes = 2 * n_frames * nx * ny * n_thetas * n_sigmas * bytes_per_float
            coarse_cache_bytes = 3 * n_frames * coarse_nx * coarse_ny * n_thetas * n_sigmas * bytes_per_float
            full_model_raw_bytes = 2 * n_frames * nx * ny * n_thetas * n_sigmas_full * n_frequencies * bytes_per_float
            if wavelet_format_var.get() == "zarr":
                full_path = param_entries.get("Full Model Wavelet Path")
                full_dir = full_path.get().strip() if full_path is not None else ""
                if full_dir:
                    exact_i = _folder_size_bytes(os.path.join(full_dir, "dwt_videodata2_i.zarr"))
                    exact_r = _folder_size_bytes(os.path.join(full_dir, "dwt_videodata2_r.zarr"))
                else:
                    exact_i = exact_r = None
                if exact_i is not None and exact_r is not None:
                    full_model_bytes = exact_i + exact_r
                    label = "Current wavelet disk usage"
                else:
                    sample = np.sin(np.linspace(0, 8 * np.pi, 20000, dtype=np.float32))
                    full_model_bytes = _estimate_blosc_bytes(full_model_raw_bytes, sample)
                    label = "Estimated wavelet disk usage"
                total_bytes = coarse_cache_bytes + full_model_bytes
                wavelet_size_label.configure(
                    text=(
                        f"{label}: {_format_bytes(total_bytes)} "
                        f"(Zarr full model + coarse RF output; temp recovery peak +{_format_bytes(highres_coarse_bytes)})"
                    )
                )
            else:
                total_bytes = coarse_cache_bytes + full_model_raw_bytes
                wavelet_size_label.configure(
                    text=(
                        f"Estimated wavelet disk usage: {_format_bytes(total_bytes)} "
                        f"(NPY final output; temp recovery peak +{_format_bytes(highres_coarse_bytes)})"
                    )
                )
        except Exception:
            wavelet_size_label.configure(text="Estimated wavelet disk usage: 0.00 GB")

    def refresh_size_estimates():
        estimate_gabor_library_size()
        estimate_wavelet_size()

    GABOR_ESTIMATE_KEYS = {"NX", "NY", "N_thetas", "Sigmas", "Phases", "Frequencies", "Save Path"}
    WAVELET_ESTIMATE_KEYS = {"Number of Frames", "NX", "NY", "N_thetas", "Sigmas", "Sigmas Full Model", "Frequencies", "Full Model Wavelet Path"}

    def add_config_row(parent, key, default, entries_dict, row, bg, labels_map):
        """Render one labeled configuration row with an optional typed browse button."""
        label_text = labels_map.get(key, key)
        ctk.CTkLabel(parent, text=label_text, text_color=text_color, font=ctk.CTkFont(size=12)).grid(
            row=row, column=0, sticky="w", pady=4,
        )

        entry_wrap = ctk.CTkFrame(parent, fg_color="transparent")
        entry_wrap.grid(row=row, column=1, pady=3, padx=(10, 0), sticky="ew")
        entry_wrap.columnconfigure(0, weight=1)

        entry = ctk.CTkEntry(entry_wrap, height=30, corner_radius=6, border_width=1)
        entry.insert(0, default)
        entry.grid(row=0, column=0, sticky="ew")
        ToolTip(entry)
        entries_dict[key] = entry

        if key in GABOR_ESTIMATE_KEYS:
            entry.bind("<KeyRelease>", lambda event: refresh_size_estimates())
        if key in WAVELET_ESTIMATE_KEYS:
            entry.bind("<KeyRelease>", lambda event: refresh_size_estimates())

        browse_kind = BROWSE_KIND.get(key)
        if browse_kind:
            ctk.CTkButton(
                entry_wrap,
                text="...",
                width=34,
                height=30,
                corner_radius=6,
                fg_color="#E5E7EB",
                hover_color="#D1D5DB",
                text_color=text_color,
                command=lambda e=entry, k=browse_kind: browse_path(e, k),
            ).grid(row=0, column=1, padx=(5, 0))

    # --- Root Window Setup & Theming ---
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    workflow_label = "Two-Photon" if workflow == WORKFLOW_2P else "Electrophysiology"
    root.title(f"Neuron Analysis Toolkit — {workflow_label}")

    def on_closing():
        if messagebox.askokcancel("Quit", "Are you sure you want to close the application? Unsaved temporary data will be removed."):
            cleanup_temporary_directories()
            root.quit()
            root.destroy()
            os._exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.geometry("1600x1000")

    try:
        icon_base64 = ''
        app_icon = tk.PhotoImage(data=icon_base64)
        root.iconphoto(True, app_icon)
    except Exception as e:
        print(f"Failed to load custom icon: {e}")
    
    bg_color = "#F3F6F8"
    frame_color = "#FFFFFF"
    text_color = "#1F2937"
    primary_btn = "#2563EB"
    success_btn = "#047857"
    danger_btn = "#DC2626"
    muted_text = "#6B7280"
    
    style = ttk.Style()
    style.theme_use("clam")
    
    main_font = ("Segoe UI Variable Display", 10)
    bold_font = ("Segoe UI Variable Display", 11, "bold")
    
    style.configure(".", font=main_font, background=bg_color, foreground=text_color)
    style.configure("TFrame", background=frame_color)
    style.configure("TLabelframe", background=frame_color, bordercolor="#E1DFDD", lightcolor="#FFFFFF", darkcolor="#E1DFDD")
    style.configure("TLabelframe.Label", background=frame_color, font=bold_font, foreground="#0078D4")
    
    style.configure("Primary.TButton", background=primary_btn, foreground="white", font=main_font, padding=(10, 6), borderwidth=0)
    style.map("Primary.TButton", background=[("active", "#005A9E")])
    style.configure("Success.TButton", background=success_btn, foreground="white", font=main_font, padding=(10, 6), borderwidth=0)
    style.map("Success.TButton", background=[("active", "#0B5B2E")])
    style.configure("Danger.TButton", background=danger_btn, foreground="white", font=main_font, padding=(10, 6), borderwidth=0)
    style.map("Danger.TButton", background=[("active", "#A4262C")])
    style.configure("Browse.TButton", font=("Segoe UI Emoji", 10), padding=2)

    root.configure(fg_color=bg_color)

    # --- View Menu ---
    menubar = tk.Menu(root)
    view_menu = tk.Menu(menubar, tearoff=0)

    def toggle_terminal():
        try:
            if frame_log in right_pane.panes():
                right_pane.forget(frame_log)
            else:
                right_pane.add(frame_log, weight=1)
                show_terminal_half()
        except Exception:
            pass

    left_panel_visible = [True]

    def toggle_left_panel():
        try:
            if left_frame in paned_h.panes():
                paned_h.forget(left_frame)
                left_panel_visible[0] = False
            else:
                paned_h.add(left_frame, weight=1)
                left_panel_visible[0] = True
        except Exception:
            pass

    view_menu.add_command(label="Toggle Left Panel", command=toggle_left_panel)
    view_menu.add_command(label="Toggle Terminal", command=toggle_terminal)
    menubar.add_cascade(label="View", menu=view_menu)
    root.config(menu=menubar)

    content_root = ctk.CTkFrame(root, fg_color=bg_color, corner_radius=0)
    content_root.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 0))

    status_frame = ctk.CTkFrame(content_root, fg_color="transparent")
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(6, 0))
    status_var = tk.StringVar(value="Ready")
    ctk.CTkLabel(status_frame, textvariable=status_var, text_color=muted_text).pack(side=tk.LEFT)
    progress_bar = ttk.Progressbar(status_frame, mode="determinate", length=260, maximum=100)
    progress_bar.pack(side=tk.RIGHT, padx=(8, 0))

    # Split layout using paned windows so left and terminal are resizable
    paned_h = ttk.PanedWindow(content_root, orient=tk.HORIZONTAL)
    paned_h.pack(fill=tk.BOTH, expand=True)

    # Left pane (resizable horizontally)
    left_frame = ttk.Frame(paned_h, style="TFrame")
    paned_h.add(left_frame, weight=1)

    # Right pane is a vertical paned window so the terminal is resizable vertically
    right_pane = ttk.PanedWindow(paned_h, orient=tk.VERTICAL)
    paned_h.add(right_pane, weight=3)

    # Container for left content (scrollable inside)
    container_left = ctk.CTkFrame(left_frame, fg_color=frame_color, corner_radius=8)
    container_left.pack(fill=tk.BOTH, expand=True)

    # Right-top visualization area
    container_right_top = ctk.CTkFrame(right_pane, fg_color=frame_color, corner_radius=8)
    right_pane.add(container_right_top, weight=3)

    # Right-bottom terminal
    frame_log = ctk.CTkFrame(right_pane, fg_color="#FFFFFF", corner_radius=8)
    right_pane.add(frame_log, weight=1)

    terminal_toolbar = ctk.CTkFrame(frame_log, fg_color="transparent")
    terminal_toolbar.pack(fill=tk.X, padx=10, pady=(8, 4))
    ctk.CTkLabel(
        terminal_toolbar,
        text="Terminal",
        text_color=primary_btn,
        font=ctk.CTkFont(size=13, weight="bold"),
    ).pack(side=tk.LEFT)
    btn_clear_terminal = ctk.CTkButton(
        terminal_toolbar,
        text="Clear",
        width=64,
        height=26,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=lambda: text_log.delete("1.0", tk.END),
    )
    btn_clear_terminal.pack(side=tk.RIGHT)
    log_scroll = ttk.Scrollbar(frame_log)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10), pady=(0, 10))
    text_log = tk.Text(
        frame_log, height=10, bg="#1E1E1E", fg="#CCCCCC",
        font=("Consolas", 10), yscrollcommand=log_scroll.set, relief="flat",
    )
    text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
    log_scroll.config(command=text_log.yview)
    sys.stdout = RedirectText(text_log)
    sys.stderr = RedirectText(text_log)

    # Set sensible initial sash positions after layout
    root.update_idletasks()
    try:
        paned_h.sashpos(0, 420)
        right_pane.sashpos(0, int(root.winfo_height() * 0.68))
    except Exception:
        pass

    # Left panel scrollable content
    frame_left = ctk.CTkScrollableFrame(
        container_left,
        fg_color=frame_color,
        corner_radius=8,
        scrollbar_button_color="#CBD5E1",
        scrollbar_button_hover_color="#94A3B8",
    )
    frame_left.pack(fill=tk.BOTH, expand=True)

    plot_tabs = ctk.CTkTabview(
        container_right_top,
        corner_radius=8,
        fg_color=frame_color,
        segmented_button_fg_color="#E5E7EB",
        segmented_button_selected_color=primary_btn,
        segmented_button_selected_hover_color="#1D4ED8",
        segmented_button_unselected_color="#E5E7EB",
        segmented_button_unselected_hover_color="#D1D5DB",
        text_color=text_color,
    )
    plot_tabs.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
    all_neurons_tab = plot_tabs.add("All neurons")
    individual_neuron_tab = plot_tabs.add("Individual neuron")
    frame_plot_all = ctk.CTkScrollableFrame(all_neurons_tab, fg_color="#F9FAFB", corner_radius=8)
    frame_plot_all.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
    individual_update_label = ctk.CTkLabel(
        individual_neuron_tab,
        text="No neuron selected",
        fg_color="#FFF7ED",
        text_color="#9A3412",
        corner_radius=6,
        height=30,
    )
    individual_update_label.pack(fill=tk.X, padx=4, pady=(4, 2))
    frame_plot_individual = ctk.CTkScrollableFrame(individual_neuron_tab, fg_color="#F9FAFB", corner_radius=8)
    frame_plot_individual.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # --- Session configuration ---
    frame_session = ttk.LabelFrame(frame_left, text="Session Configuration", padding=15)
    frame_session.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    btn_load_state = ctk.CTkButton(
        frame_session,
        text="Load Configuration",
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=load_app_state,
    )
    btn_load_state.pack(fill=tk.X, pady=3)
    btn_save_state = ctk.CTkButton(
        frame_session,
        text="Save Configuration",
        fg_color="#4B5563",
        hover_color="#374151",
        command=save_app_state,
    )
    btn_save_state.pack(fill=tk.X, pady=3)

    # --- 1 · Gabor filter bank ---
    frame_gabor = ttk.LabelFrame(frame_left, text="1 · Gabor Filter Bank", padding=15)
    frame_gabor.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_gabor.columnconfigure(1, weight=1)

    gabor_entries = {}
    for i, (label, default) in enumerate(gabor_param.items()):
        add_config_row(frame_gabor, label, default, gabor_entries, i, frame_color, GABOR_LABELS)

    gabor_format_var = tk.StringVar(value="npy")
    btn_submit_gabor = ctk.CTkButton(
        frame_gabor,
        text="Build Gabor Library",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(create_gabor, "Gabor library construction"),
    )
    btn_submit_gabor.grid(row=len(gabor_param), column=0, columnspan=2, pady=(15, 0), sticky="ew")

    format_frame = ctk.CTkFrame(frame_gabor, fg_color="transparent")
    format_frame.grid(row=len(gabor_param)+1, column=0, columnspan=2, pady=(10, 0), sticky="w")
    ctk.CTkLabel(format_frame, text="Library format:", text_color=muted_text).pack(side=tk.LEFT)

    def _set_gabor_format(val):
        gabor_format_var.set(val)
        try:
            refresh_size_estimates()
        except NameError:
            pass

    gabor_format_segment = ctk.CTkSegmentedButton(
        format_frame,
        values=["npy", "zarr"],
        variable=gabor_format_var,
        command=_set_gabor_format,
        height=26,
        corner_radius=6,
        border_width=1,
        fg_color="#E5E7EB",
        selected_color=primary_btn,
        selected_hover_color="#1D4ED8",
        unselected_color="#F3F4F6",
        unselected_hover_color="#E5E7EB",
        text_color="#FFFFFF",
        text_color_disabled="#9CA3AF",
    )
    gabor_format_segment.pack(side=tk.LEFT, padx=(10, 0))
    gabor_format_segment.set(gabor_format_var.get())
    try:
        gabor_format_var.trace_add("write", lambda *a: refresh_size_estimates())
    except Exception:
        pass

    gabor_size_label = ttk.Label(
        frame_gabor,
        text="Estimated disk size: 0.00 GB",
        font=main_font,
        background=frame_color,
        foreground=text_color,
    )
    gabor_size_label.grid(row=len(gabor_param)+2, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # --- 2 · Stimulus wavelet pipeline ---
    frame_processing = ttk.LabelFrame(frame_left, text="2 · Stimulus Wavelet Pipeline", padding=15)
    frame_processing.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    wavelet_format_var = tk.StringVar(value="zarr")
    btn_submit_wavelet = ctk.CTkButton(
        frame_processing,
        text="Run Wavelet Decomposition",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(run_wavelet, "Stimulus wavelet decomposition"),
    )
    btn_submit_wavelet.pack(fill=tk.X, pady=3)

    format_frame_wavelet = ctk.CTkFrame(frame_processing, fg_color="transparent")
    format_frame_wavelet.pack(anchor="w", pady=(10, 0))
    ctk.CTkLabel(format_frame_wavelet, text="Full-model format:", text_color=muted_text).pack(side=tk.LEFT)

    def _set_wavelet_format(val):
        wavelet_format_var.set(val)
        try:
            refresh_size_estimates()
        except NameError:
            pass

    wavelet_format_segment = ctk.CTkSegmentedButton(
        format_frame_wavelet,
        values=["npy", "zarr"],
        variable=wavelet_format_var,
        command=_set_wavelet_format,
        height=26,
        corner_radius=6,
        border_width=1,
        fg_color="#E5E7EB",
        selected_color=primary_btn,
        selected_hover_color="#1D4ED8",
        unselected_color="#F3F4F6",
        unselected_hover_color="#E5E7EB",
        text_color="#FFFFFF",
        text_color_disabled="#9CA3AF",
    )
    wavelet_format_segment.pack(side=tk.LEFT, padx=(10, 0))
    wavelet_format_segment.set(wavelet_format_var.get())
    try:
        wavelet_format_var.trace_add("write", lambda *a: refresh_size_estimates())
    except Exception:
        pass

    wavelet_size_label = ttk.Label(
        frame_processing,
        text="Estimated wavelet disk usage: 0.00 GB",
        font=main_font,
        background=frame_color,
        foreground=text_color,
    )
    wavelet_size_label.pack(anchor="w", pady=(6, 0))

    ttk.Separator(frame_processing, orient="horizontal").pack(fill=tk.X, pady=8)

    # --- Experiment configuration ---
    frame_params = ttk.LabelFrame(
        frame_left,
        text=f"Experiment Configuration ({workflow_label})",
        padding=15,
    )
    frame_params.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_params.columnconfigure(1, weight=1)

    param_entries = {}
    PARAMETER_GROUPS = {
        "Data Input": [
            "Dir",
            "Experiment Info",
            "Movie Path",
            "Library Path",
            "Spks Path",
            "Path Directory",
            "Full Model Wavelet Path",
            "Full Model Save Path",
            "Plot Cache Path",
            "Recovery Cache Directory",
        ],
        "Acquisition & Timing": [
            "Number of Planes",
            "Sampling Rate (samples / sec)",
            "Hz",
            "Number of Frames",
            "Number of Trials to Keep",
            "Block End",
        ],
        "Spatial & Wavelet": [
            "screen_x",
            "screen_y",
            "NX",
            "NY",
            "Resolution",
            "Visual Coverage",
            "Analysis Coverage",
            "Sigmas",
            "Sigmas Full Model",
            "Frequencies",
        ],
    }

    for section_title, section_keys in PARAMETER_GROUPS.items():
        section_frame = ttk.LabelFrame(frame_params, text=section_title, padding=(10, 8))
        section_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        section_frame.columnconfigure(1, weight=1)
        section_row = 0
        for key in section_keys:
            if key not in workflow_param_keys:
                continue
            default = filtered_param_defaults.get(key, "")
            add_config_row(section_frame, key, default, param_entries, section_row, frame_color, ANALYSIS_LABELS)
            section_row += 1
    refresh_size_estimates()

    # --- 3 · Neural & RF analysis ---
    frame_analysis = ttk.LabelFrame(frame_left, text="3 · Neural & RF Analysis", padding=15)
    frame_analysis.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    btn_submit_plot = ctk.CTkButton(
        frame_analysis,
        text="Run Coarse RF Analysis",
        height=34,
        corner_radius=6,
        fg_color=success_btn,
        hover_color="#065F46",
        command=run_in_thread(plot_data, "Coarse receptive-field analysis"),
    )
    btn_submit_plot.pack(fill=tk.X, pady=(0, 10))

    ttk.Separator(frame_analysis, orient="horizontal").pack(fill=tk.X, pady=8)

    rf_wrap = ttk.Frame(frame_analysis, style="TFrame")
    rf_wrap.pack(fill=tk.X, pady=(0, 10))
    rf_wrap.columnconfigure(1, weight=1)

    ctk.CTkLabel(
        rf_wrap, text=FIELD_LABELS["Neuron ID"], text_color=text_color,
        font=ctk.CTkFont(size=12, weight="bold"),
    ).grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))

    entry_neuron = ctk.CTkEntry(rf_wrap, height=30, corner_radius=6, border_width=1)
    entry_neuron.insert(0, '1173')
    entry_neuron.grid(row=0, column=1, sticky="ew")
    param_entries['Neuron ID'] = entry_neuron

    btn_runRF = ctk.CTkButton(
        frame_analysis,
        text="Inspect Single Neuron",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
    )
    btn_runRF.pack(fill=tk.X)

    btn_run_model_plots = ctk.CTkButton(
        frame_analysis,
        text="Run Simple Model Plots",
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=run_in_thread(plot_run_model_outputs, "run_Model plot capture"),
    )
    btn_run_model_plots.pack(fill=tk.X, pady=(8, 0))
    btn_run_full_model_plots = ctk.CTkButton(
        frame_analysis,
        text="Run Full Model Plots",
        height=34,
        corner_radius=6,
        fg_color="#7C3AED",
        hover_color="#5B21B6",
        command=run_in_thread(plot_run_full_model_outputs, "run_Full_Model plot capture"),
    )
    btn_run_full_model_plots.pack(fill=tk.X, pady=(6, 0))

    # --- 4 · Export ---
    frame_export = ttk.LabelFrame(frame_left, text="4 · Export", padding=15)
    frame_export.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    btn_export_all_results = ctk.CTkButton(
        frame_export,
        text="Export All Displayed Results",
        fg_color="#4B5563",
        hover_color="#374151",
        command=export_all_displayed_results,
    )
    btn_export_all_results.pack(fill=tk.X, pady=3)
    btn_export_all_neurons = ctk.CTkButton(
        frame_export,
        text="Export All Neurons Tab",
        fg_color="#4B5563",
        hover_color="#374151",
        command=export_all_neurons_results,
    )
    btn_export_all_neurons.pack(fill=tk.X, pady=3)
    btn_export_individual_neuron = ctk.CTkButton(
        frame_export,
        text="Export Individual Neuron Tab",
        fg_color="#4B5563",
        hover_color="#374151",
        command=export_individual_neuron_results,
    )
    btn_export_individual_neuron.pack(fill=tk.X, pady=3)

    # --- Global Controls ---
    frame_controls = ttk.Frame(frame_left, style="TFrame")
    frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    all_buttons = [
        btn_submit_gabor,
        btn_submit_wavelet,
        btn_submit_plot,
        btn_runRF,
        btn_run_model_plots,
        btn_run_full_model_plots,
        btn_export_all_results,
        btn_export_all_neurons,
        btn_export_individual_neuron,
        btn_save_state,
        btn_load_state,
    ]

    root.mainloop()
