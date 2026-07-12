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
import traceback
import customtkinter as ctk
import psutil

# --- DPI Awareness ---
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    myappid = 'neuro.gabor.toolkit.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

from ..config import (
    AnalysisConfig,
    DEFAULT_COMMON_PARAMS,
    DEFAULT_EPHYS_PARAMS,
    DEFAULT_TWO_PHOTON_PARAMS,
    WORKFLOW_2P,
    WORKFLOW_EPHYS,
    coarse_grid_dimensions,
    parse_literal,
)
import numpy as np

from ..gui_support import (
    ToolTip,
    _build_size_text,
    _folder_size_bytes,
    _format_bytes,
    _normalise_gabor_params,
    _ordered_float_union,
    _parse_data_dir,
    _safe_name,
    _zarr_output_path,
)
from ..runtime.keep_awake import KeepAwake
from ..runtime.task_control import OperationCancelled, check_cancelled, format_duration
from ..project_layout import (
    WavenProjectLayout,
    conventional_downsample_path,
    conventional_gabor_path,
    find_single_artifact,
    find_stimulus_movie,
    resolve_folder_reference,
    write_reference,
)
from ..storage.neural_cache import find_neural_cache_pair, load_neural_cache_pair, load_unit_ids
from ..stimulus.metadata import coverage_ratios, downsampled_grid_dimensions, read_movie_metadata
_ANALYSIS_IMPORTS_READY = False
_GABOR_IMPORTS_READY = False
_WAVELET_IMPORTS_READY = False
_RF_IMPORTS_READY = False
_MODEL_IMPORTS_READY = False
_PLOT_IMPORTS_READY = False
plt = None
FigureCanvasTkAgg = None
NavigationToolbar2Tk = None


def _ensure_plot_imports():
    """Function for ensure plot imports."""
    global _PLOT_IMPORTS_READY, plt, FigureCanvasTkAgg, NavigationToolbar2Tk
    if _PLOT_IMPORTS_READY:
        return
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg as _FigureCanvasTkAgg
    from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as _NavigationToolbar2Tk
    import matplotlib.pyplot as _plt

    FigureCanvasTkAgg = _FigureCanvasTkAgg
    NavigationToolbar2Tk = _NavigationToolbar2Tk
    plt = _plt
    _PLOT_IMPORTS_READY = True


def _ensure_gabor_imports(label="Gabor library construction"):
    """Function for ensure gabor imports.

    Args:
        label: Input value for this operation.
    """
    global _GABOR_IMPORTS_READY
    global makeFilterLibrary, makeFilterLibrary2, makeGaborFilter
    if _GABOR_IMPORTS_READY:
        return
    from ..wavelets.filters import (
        makeFilterLibrary as _makeFilterLibrary,
        makeFilterLibrary2 as _makeFilterLibrary2,
        makeGaborFilter as _makeGaborFilter,
    )

    makeFilterLibrary = _makeFilterLibrary
    makeFilterLibrary2 = _makeFilterLibrary2
    makeGaborFilter = _makeGaborFilter
    _GABOR_IMPORTS_READY = True


def _ensure_wavelet_imports(label="stimulus wavelet generation"):
    """Function for ensure wavelet imports.

    Args:
        label: Input value for this operation.
    """
    global _WAVELET_IMPORTS_READY
    global coarseWavelet, downsample_video_binary, waveletDecomposition, waveletDecompositionFull
    global build_convolution_kernel_cache, convolution_kernel_cache_path
    global waveletDecompositionConv, waveletPowerDecompositionConv, waveletDecompositionFullConv
    global video_downsample_chunk_size, convert_npy_to_zarr
    if _WAVELET_IMPORTS_READY:
        return
    from ..stimulus import coarseWavelet as _coarseWavelet
    from ..wavelets.decomposition import (
        build_convolution_kernel_cache as _build_convolution_kernel_cache,
        convolution_kernel_cache_path as _convolution_kernel_cache_path,
        downsample_video_binary as _downsample_video_binary,
        waveletDecomposition as _waveletDecomposition,
        waveletDecompositionConv as _waveletDecompositionConv,
        waveletPowerDecompositionConv as _waveletPowerDecompositionConv,
        waveletDecompositionFull as _waveletDecompositionFull,
        waveletDecompositionFullConv as _waveletDecompositionFullConv,
    )
    from ..runtime.performance import video_downsample_chunk_size as _video_downsample_chunk_size
    from ..storage.wavelet_zarr import convert_npy_to_zarr as _convert_npy_to_zarr

    coarseWavelet = _coarseWavelet
    build_convolution_kernel_cache = _build_convolution_kernel_cache
    convolution_kernel_cache_path = _convolution_kernel_cache_path
    downsample_video_binary = _downsample_video_binary
    waveletDecomposition = _waveletDecomposition
    waveletDecompositionConv = _waveletDecompositionConv
    waveletPowerDecompositionConv = _waveletPowerDecompositionConv
    waveletDecompositionFull = _waveletDecompositionFull
    waveletDecompositionFullConv = _waveletDecompositionFullConv
    video_downsample_chunk_size = _video_downsample_chunk_size
    convert_npy_to_zarr = _convert_npy_to_zarr
    _WAVELET_IMPORTS_READY = True


def _ensure_rf_imports(label="coarse RF analysis"):
    """Function for ensure rf imports.

    Args:
        label: Input value for this operation.
    """
    global _RF_IMPORTS_READY
    global compute_skewness_neurons, PearsonCorrelationPinkNoise, PlotTuningCurve
    global repetability_trial3, firing_rate_orientation_tuning
    if _RF_IMPORTS_READY:
        return
    _ensure_plot_imports()
    from ..analysis.receptive_fields import (
        compute_skewness_neurons as _compute_skewness_neurons,
        PearsonCorrelationPinkNoise as _PearsonCorrelationPinkNoise,
        repetability_trial3 as _repetability_trial3,
    )
    from ..analysis.nonlinear_models import (
        PlotTuningCurve as _PlotTuningCurve,
    )
    from ..analysis.orientation_selectivity import (
        firing_rate_orientation_tuning as _firing_rate_orientation_tuning,
    )

    compute_skewness_neurons = _compute_skewness_neurons
    PearsonCorrelationPinkNoise = _PearsonCorrelationPinkNoise
    PlotTuningCurve = _PlotTuningCurve
    repetability_trial3 = _repetability_trial3
    firing_rate_orientation_tuning = _firing_rate_orientation_tuning
    _RF_IMPORTS_READY = True


def _ensure_model_imports(label="model plot capture"):
    """Function for ensure model imports.

    Args:
        label: Input value for this operation.
    """
    global _MODEL_IMPORTS_READY
    global run_Model, run_Full_Model, smooth_best_positions
    if _MODEL_IMPORTS_READY:
        return
    _ensure_plot_imports()
    from ..analysis.model_runs import (
        run_Model as _run_Model,
        run_Full_Model as _run_Full_Model,
    )
    from ..pipeline import smooth_best_positions as _smooth_best_positions

    run_Model = _run_Model
    run_Full_Model = _run_Full_Model
    smooth_best_positions = _smooth_best_positions
    _MODEL_IMPORTS_READY = True


def _ensure_analysis_imports(label="analysis"):
    """Function for ensure analysis imports.

    Args:
        label: Input value for this operation.
    """
    global _ANALYSIS_IMPORTS_READY
    if _ANALYSIS_IMPORTS_READY:
        return
    _ensure_gabor_imports(label)
    _ensure_wavelet_imports(label)
    _ensure_rf_imports(label)
    _ensure_model_imports(label)
    _ANALYSIS_IMPORTS_READY = True



def select_workflow() -> str:
    """Return the default GUI workflow for backward-compatible callers."""
    return WORKFLOW_2P


def run(param_defaults=None, gabor_param=None, workflow=None, gui_options=None):
    """Function for run.

    Args:
        param_defaults: Input value for this operation.
        gabor_param: Input value for this operation.
        workflow: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    param_defaults = dict(param_defaults or {})
    gui_options = dict(gui_options or {})
    if workflow not in (WORKFLOW_2P, WORKFLOW_EPHYS):
        workflow = WORKFLOW_2P
    gabor_param = _normalise_gabor_params(gabor_param)

    GABOR_LABELS = {
        "N_thetas": "Orientation Count",
        "Sigmas": "Filter Sizes (px)",
        "Frequencies": "Spatial Frequencies (cyc/px)",
        "Phases": "Phases (degrees)",
        "Save Path": "Gabor Cache Folder",
        "Coarse Library Path": "Coarse Gabor Folder",
        "Fine Library Path": "Fine Gabor Folder",
    }

    ANALYSIS_LABELS = {
        "Project Root": "Project Root Folder",
        "Dir": "Raw Data Folder",
        "Path Directory": "Coarse Wavelet Folder",
        "Experiment Info": "Experiment ID (mouse, date, #)",
        "Number of Planes": "Imaging Planes",
        "Block End": "Session Block Start Frame",
        "Resolution": "Microscope Resolution (µm/px)",
        "Sampling Rate (samples / sec)": "Recording Sampling Rate (Hz)",
        "Sigmas": "RF Filter Sizes (px)",
        "Sigmas Full Model": "Full-Model Filter Sizes (px)",
        "Frequencies": "Stimulus Frequencies (cyc/px)",
        "Visual Coverage": "Visual Field Coverage (°)",
        "Analysis Coverage": "Analysis Field Coverage (°)",
        "Number of Frames": "Frames per Trial",
        "Movie Path": "Stimulus Movie Folder",
        "Library Path": "Gabor Library Folder",
        "Coarse Library Path": "Coarse Gabor Folder",
        "Fine Library Path": "Fine Gabor Folder",
        "Spks Path": "Neural Cache Folder",
        "Full Model Wavelet Path": "Full-Model Wavelet Folder",
        "Full Model Save Path": "Full-Model Results Directory",
        "Plot Cache Path": "Plot Cache Folder",
        "Recovery Cache Directory": "Recovery Checkpoint Directory",
        "Train Trial Indices": "Train Trials",
        "Test Trial Indices": "Test Trials",
        "Use Last Minute Holdout": "Last-Minute Holdout",
        "Neuron ID": "Neuron Index",
    }

    def workflow_display_name(value):
        """Function for workflow display name.

        Args:
            value: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        return "Two-Photon" if value == WORKFLOW_2P else "Electrophysiology"

    def workflow_defaults(value):
        """Function for workflow defaults.

        Args:
            value: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        keys = AnalysisConfig.gui_param_keys(value)
        return keys, {key: param_defaults.get(key, "") for key in keys}

    FIELD_LABELS = {**GABOR_LABELS, **ANALYSIS_LABELS}

    BROWSE_KIND = {
        "Project Root": "dir",
        "Save Path": "dir",
        "Movie Path": "dir",
        "Library Path": "dir",
        "Coarse Library Path": "dir",
        "Fine Library Path": "dir",
        "Spks Path": "dir",
        "Path Directory": "dir",
        "Dir": "dir",
        "Full Model Wavelet Path": "dir",
        "Full Model Save Path": "dir",
        "Plot Cache Path": "dir",
        "Recovery Cache Directory": "dir",
    }

    INPUT_HINTS = {
        "Experiment Info": "Tuple: ('subject', 'YYYY-MM-DD', experiment_number)",
        "Number of Planes": "Integer, e.g. 1",
        "Sampling Rate (samples / sec)": "Number in Hz, e.g. 30000",
        "Train Trial Indices": "'auto' or zero-based list, e.g. [0, 2]",
        "Test Trial Indices": "'auto' or zero-based list, e.g. [1]",
        "Use Last Minute Holdout": "Boolean: True or False",
        "Block End": "Integer acquisition block index, e.g. 0",
        "Resolution": "Micrometers per pixel, e.g. 1.3671",
        "Visual Coverage": "[left, right, top, bottom] in degrees",
        "Analysis Coverage": "[left, right, top, bottom] in degrees",
        "N_thetas": "Integer orientation bins, e.g. 18",
        "Sigmas": "Numeric list in analysis pixels, e.g. [2, 4, 8]",
        "Sigmas Full Model": "Numeric list in analysis pixels, e.g. [2, 4, 8, 12]",
        "Frequencies": "Numeric list in cycles/pixel, e.g. [0.02, 0.06, 0.1]",
        "Phases": "Numeric list in degrees, e.g. [0, 90]",
        "Neuron ID": "Zero-based neuron index, e.g. 1173",
    }

    BROWSE_ICONS = {"file": "📄", "savefile": "📄", "dir": "📁"}

    class RedirectText:
        """Container for RedirectText."""
        def __init__(self, widget, max_lines=5000, flush_ms=50):
            """Function for init.

            Args:
                widget: Input value for this operation.
                max_lines: Input value for this operation.
                flush_ms: Input value for this operation.
            """
            self.widget = widget
            self.max_lines = max_lines
            self.flush_ms = flush_ms
            self._buffer = []
            self._lock = threading.Lock()
            self._flush_pending = False
            self.widget.after(self.flush_ms, self._poll_flush)

        def write(self, string):
            """Function for write.

            Args:
                string: Input value for this operation.
            """
            if not string:
                return
            with self._lock:
                self._buffer.append(string)
                self._flush_pending = True

        def _poll_flush(self):
            """Flush buffered worker output from Tk's main thread."""
            try:
                self._flush()
            finally:
                try:
                    self.widget.after(self.flush_ms, self._poll_flush)
                except Exception:
                    pass

        def _flush(self):
            """Function for flush."""
            with self._lock:
                chunk = "".join(self._buffer)
                self._buffer.clear()
                self._flush_pending = False
            if not chunk:
                return
            try:
                at_bottom = self.widget.yview()[1] >= 0.98
                self.widget.mark_set(tk.INSERT, tk.END)
                self._insert_terminal_chunk(chunk)
                line_count = int(float(self.widget.index("end-1c").split(".")[0]))
                if line_count > self.max_lines:
                    self.widget.delete("1.0", f"{line_count - self.max_lines}.0")
                if at_bottom:
                    self.widget.see(tk.END)
            except Exception:
                pass

        def _insert_terminal_chunk(self, chunk):
            """Function for insert terminal chunk.

            Args:
                chunk: Input value for this operation.
            """
            for part in chunk.splitlines(keepends=True):
                if "\r" in part:
                    before, _, after = part.rpartition("\r")
                    if before:
                        self.widget.insert(tk.END, before.replace("\r", ""))
                    self.widget.delete("end-1c linestart", "end-1c lineend")
                    if after:
                        self.widget.insert(tk.END, after.replace("\r", ""))
                else:
                    self.widget.insert(tk.END, part)

        def flush(self):
            """Function for flush."""
            if threading.current_thread() is threading.main_thread():
                self._flush()

        def isatty(self):
            """Function for isatty.

            Returns:
                Result produced by the operation.
            """
            return True

        def writable(self):
            """Function for writable.

            Returns:
                Result produced by the operation.
            """
            return True

        @property
        def encoding(self):
            """Function for encoding.

            Returns:
                Result produced by the operation.
            """
            return "utf-8"

    class TaskResourceMonitor:
        """Container for TaskResourceMonitor."""
        def __init__(self, label):
            """Function for init.

            Args:
                label: Input value for this operation.
            """
            self.label = label
            self.process = psutil.Process(os.getpid())
            self.start_time = None
            self.start_perf = None
            self.start_cpu = None
            self.start_io = None
            self.start_net = None
            self.peak_rss = 0
            self.peak_cuda_allocated = 0
            self.peak_cuda_reserved = 0
            self._stop_event = threading.Event()
            self._thread = None

        def start(self):
            """Function for start."""
            self.start_time = time.time()
            self.start_perf = time.perf_counter()
            self.start_cpu = self.process.cpu_times()
            self.start_io = self._io_counters()
            self.start_net = self._net_counters()
            self.peak_rss = self._rss()
            self._reset_cuda_peaks()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()

        def stop(self):
            """Function for stop.

            Returns:
                Result produced by the operation.
            """
            self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=1)
            return self.summary()

        def _sample_loop(self):
            """Function for sample loop."""
            while not self._stop_event.wait(1.0):
                self.peak_rss = max(self.peak_rss, self._rss())
                cuda_allocated, cuda_reserved = self._cuda_peaks()
                self.peak_cuda_allocated = max(self.peak_cuda_allocated, cuda_allocated)
                self.peak_cuda_reserved = max(self.peak_cuda_reserved, cuda_reserved)

        def _rss(self):
            """Function for rss.

            Returns:
                Result produced by the operation.
            """
            try:
                return int(self.process.memory_info().rss)
            except Exception:
                return 0

        def _io_counters(self):
            """Function for io counters.

            Returns:
                Result produced by the operation.
            """
            try:
                return self.process.io_counters()
            except Exception:
                return None

        def _net_counters(self):
            """Function for net counters.

            Returns:
                Result produced by the operation.
            """
            try:
                return psutil.net_io_counters()
            except Exception:
                return None

        def _reset_cuda_peaks(self):
            """Function for reset cuda peaks."""
            try:
                import torch

                if torch.cuda.is_available():
                    for idx in range(torch.cuda.device_count()):
                        torch.cuda.reset_peak_memory_stats(idx)
            except Exception:
                pass

        def _cuda_peaks(self):
            """Function for cuda peaks.

            Returns:
                Result produced by the operation.
            """
            allocated = 0
            reserved = 0
            try:
                import torch

                if torch.cuda.is_available():
                    for idx in range(torch.cuda.device_count()):
                        allocated += int(torch.cuda.max_memory_allocated(idx))
                        reserved += int(torch.cuda.max_memory_reserved(idx))
            except Exception:
                pass
            return allocated, reserved

        def summary(self):
            """Function for summary.

            Returns:
                Result produced by the operation.
            """
            elapsed = max(time.perf_counter() - self.start_perf, 1e-6) if self.start_perf else 0
            end_time = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
            cpu_pct = 0.0
            try:
                cpu_now = self.process.cpu_times()
                cpu_start = self.start_cpu
                cpu_seconds = (
                    (cpu_now.user - cpu_start.user)
                    + (cpu_now.system - cpu_start.system)
                )
                cpu_pct = 100 * cpu_seconds / max(elapsed, 1e-6)
            except Exception:
                cpu_seconds = 0.0
            io_now = self._io_counters()
            read_bytes = write_bytes = 0
            if io_now is not None and self.start_io is not None:
                read_bytes = max(0, int(io_now.read_bytes - self.start_io.read_bytes))
                write_bytes = max(0, int(io_now.write_bytes - self.start_io.write_bytes))
            net_now = self._net_counters()
            net_sent = net_recv = 0
            if net_now is not None and self.start_net is not None:
                net_sent = max(0, int(net_now.bytes_sent - self.start_net.bytes_sent))
                net_recv = max(0, int(net_now.bytes_recv - self.start_net.bytes_recv))
            cuda_allocated, cuda_reserved = self._cuda_peaks()
            self.peak_cuda_allocated = max(self.peak_cuda_allocated, cuda_allocated)
            self.peak_cuda_reserved = max(self.peak_cuda_reserved, cuda_reserved)
            self.peak_rss = max(self.peak_rss, self._rss())
            return {
                "completed_at": end_time,
                "elapsed": elapsed,
                "cpu_seconds": cpu_seconds,
                "cpu_percent": cpu_pct,
                "peak_ram": self.peak_rss,
                "disk_read": read_bytes,
                "disk_write": write_bytes,
                "gpu_allocated": self.peak_cuda_allocated,
                "gpu_reserved": self.peak_cuda_reserved,
                "net_sent": net_sent,
                "net_recv": net_recv,
            }

    task_state = {
        "name": None,
        "start": None,
        "detail": None,
        "last_ui_update": 0,
        "last_log_progress": -1,
    }
    completed_actions = set()

    def _invalidate_completed_actions(*prefixes):
        """Remove completed stage markers affected by a changed GUI input."""
        prefixes = tuple(str(prefix) for prefix in prefixes)
        stale = {
            action
            for action in completed_actions
            if any(action == prefix or action.startswith(f"{prefix}:") for prefix in prefixes)
        }
        completed_actions.difference_update(stale)
        try:
            refresh_action_buttons()
        except NameError:
            pass
    active_task = {
        "cancel_event": None,
        "cleanup_paths": [],
        "monitor": None,
        "cancel_requested": False,
    }

    def _current_cancel_event():
        """Function for current cancel event.

        Returns:
            Result produced by the operation.
        """
        return active_task.get("cancel_event")

    def _raise_if_cancelled():
        """Function for raise if cancelled."""
        check_cancelled(_current_cancel_event())

    def _register_cancel_cleanup_path(path):
        """Function for register cancel cleanup path.

        Args:
            path: Input value for this operation.
        """
        if not path:
            return
        active_task.setdefault("cleanup_paths", []).append(os.path.abspath(path))

    def _remove_cancelled_task_paths():
        """Function for remove cancelled task paths.

        Returns:
            Result produced by the operation.
        """
        removed = 0
        for path in reversed(active_task.get("cleanup_paths", [])):
            try:
                if os.path.isdir(path):
                    shutil.rmtree(path)
                    removed += 1
                    print(f"Removed cancelled task folder: {path}")
                elif os.path.exists(path):
                    os.remove(path)
                    removed += 1
                    print(f"Removed cancelled task file: {path}")
            except Exception as exc:
                print(f"Could not remove cancelled task path {path}: {exc}")
        active_task["cleanup_paths"] = []
        return removed

    def _format_task_summary(metrics, status):
        """Function for format task summary.

        Args:
            metrics: Input value for this operation.
            status: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if not metrics:
            return ""
        return (
            f"  Status : {status.upper()}\n"
            f"  Time   : {format_duration(metrics['elapsed'])}\n"
            f"  CPU/RAM: {metrics['cpu_percent']:.1f}% / {_format_bytes(metrics['peak_ram'])}\n"
            f"  Disk   : read {_format_bytes(metrics['disk_read'])}, write {_format_bytes(metrics['disk_write'])}\n"
            f"  GPU    : {_format_bytes(metrics['gpu_allocated'])} allocated\n"
        )

    def _refresh_task_heartbeat():
        """Keep elapsed time moving even while a worker emits no explicit progress."""
        if not task_state.get("name") or not task_state.get("start"):
            return
        elapsed = format_duration(time.time() - task_state["start"])
        detail = task_state.get("detail") or "working"
        try:
            terminal_metrics_var.set(f"elapsed {elapsed} | {detail}")
        except NameError:
            return
        root.after(1000, _refresh_task_heartbeat)

    def request_cancel_current_task():
        """Function for request cancel current task."""
        cancel_event = active_task.get("cancel_event")
        if cancel_event is None or cancel_event.is_set():
            return
        active_task["cancel_requested"] = True
        cancel_event.set()
        status_var.set("Cancelling current task...")
        try:
            terminal_status_var.set("Task: cancelling...")
            btn_cancel_terminal.configure(state=tk.DISABLED, text="Cancelling...")
        except NameError:
            pass
        print("\n[!] Cancel requested. Waiting for the current safe checkpoint, then cleaning partial files.")

    def flash_taskbar():
        """Function for flash taskbar."""
        try:
            FLASHW_ALL = 3
            class FLASHWINFO(ctypes.Structure):
                """Container for FLASHWINFO."""
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
        """Function for update progress.

        Args:
            percent: Input value for this operation.
            message: Input value for this operation.
            detail: Input value for this operation.
        """
        if threading.current_thread() is not threading.main_thread():
            try:
                root.after(0, lambda: update_progress(percent, message, detail))
            except Exception:
                pass
            return

        if percent is None:
            progress_bar.configure(mode="indeterminate")
            progress_bar.start(12)
        else:
            percent = max(0, min(float(percent), 100))
            progress_bar.stop()
            progress_bar.configure(mode="determinate", value=percent)

        if detail is not None:
            task_state["detail"] = detail

        if message is None and task_state["name"]:
            message = task_state["name"]
        suffix = ""
        if message is not None:
            suffix = f" - {task_state['detail']}" if task_state.get("detail") else ""
            status_var.set(f"{message}{suffix}")
            try:
                terminal_status_var.set(f"Task: {message}{suffix}")
            except NameError:
                pass

        if percent is not None and task_state.get("start") and percent > 0 and percent < 100:
            elapsed = time.time() - task_state["start"]
            eta = elapsed * (100 - percent) / percent
            speed = percent / max(elapsed, 1e-6)
            metric_text = f"{percent:.1f}% | elapsed {int(elapsed)}s | ETA {int(eta)}s | {speed:.2f}%/s"
            status_var.set(f"{message}{suffix} - {metric_text}")
            try:
                terminal_metrics_var.set(metric_text)
            except NameError:
                pass
            log_bucket = int(percent // 5)
            if log_bucket != task_state.get("last_log_progress"):
                task_state["last_log_progress"] = log_bucket
                print(f"[progress] {message}{suffix}: {metric_text}")
        elif percent is not None:
            try:
                terminal_metrics_var.set(f"{percent:.1f}%")
            except NameError:
                pass

        now = time.time()
        if threading.current_thread() is threading.main_thread() and now - task_state.get("last_ui_update", 0) > 0.1:
            task_state["last_ui_update"] = now
            root.update_idletasks()

    def show_terminal_half():
        """Reveal the log pane and allocate the lower half of the window to it."""
        try:
            if str(frame_log) not in right_pane.panes():
                right_pane.add(frame_log, minsize=220, stretch="always")
            terminal_visible[0] = True
            root.update_idletasks()
            total_h = right_pane.winfo_height() or root.winfo_height()
            top_h = max(260, int(total_h * 0.56))
            set_pane_sash(right_pane, 0, top_h)
        except Exception:
            pass

    def begin_task(task_name, cancel_event, monitor):
        """Function for begin task.

        Args:
            task_name: Input value for this operation.
            cancel_event: Input value for this operation.
            monitor: Input value for this operation.
        """
        task_state["name"] = task_name
        task_state["start"] = time.time()
        task_state["detail"] = None
        task_state["last_log_progress"] = -1
        active_task["cancel_event"] = cancel_event
        active_task["monitor"] = monitor
        active_task["cancel_requested"] = False
        show_terminal_half()
        update_progress(None, f"Running: {task_name}")
        for btn in all_buttons:
            btn.configure(state=tk.DISABLED)
        try:
            btn_cancel_terminal.configure(state=tk.NORMAL, text="Cancel")
        except NameError:
            pass
        print(f"\n[{time.strftime('%H:%M:%S')}] {task_name.upper()}\n  Starting...\n")
        root.after(1000, _refresh_task_heartbeat)

    def end_task(success=False, cancelled=False, metrics=None):
        """Function for end task.

        Args:
            success: Input value for this operation.
            cancelled: Input value for this operation.
            metrics: Input value for this operation.
        """
        completed_name = task_state.get("name")
        if completed_name:
            progress_bar.stop()
            progress_bar.configure(mode="determinate", value=0 if cancelled else 100)
            status_label = "cancelled" if cancelled else ("finished" if success else "failed")
            print(_format_task_summary(metrics, status_label))
            print(f"[{time.strftime('%H:%M:%S')}] {task_state['name']} {status_label}.\n")
            flash_taskbar()
            if success:
                action_map = {
                    "Neural cache creation": "neural",
                    "Stimulus video downsampling": "downsample",
                    "Gabor asset preparation": "gabor_all",
                    "Prepare Coarse RF wavelets": "wavelet:coarse_rf",
                    "Prepare Run Model wavelets": "wavelet:model",
                    "Prepare Run Full Model wavelets": "wavelet:full_model",
                    "Coarse receptive-field analysis": "rf",
                }
                action = action_map.get(completed_name)
                if action:
                    if action == "downsample":
                        action = f"{action}:{_selected_analysis_scale()}"
                        completed_actions.add(action)
                    elif action == "gabor_all":
                        completed_actions.update({"gabor:coarse", "gabor:full"})
                    else:
                        completed_actions.add(action)
        task_state["name"] = None
        task_state["start"] = None
        task_state["detail"] = None
        task_state["last_log_progress"] = -1
        status_var.set("Ready")
        active_task["cancel_event"] = None
        active_task["monitor"] = None
        active_task["cancel_requested"] = False
        try:
            terminal_status_var.set("Task: idle")
            terminal_metrics_var.set("")
            btn_cancel_terminal.configure(state=tk.DISABLED, text="Cancel")
        except NameError:
            pass
        for btn in all_buttons:
            btn.configure(state=tk.NORMAL)
        try:
            refresh_action_buttons(advance_from=completed_name if success else None)
        except NameError:
            pass

    def run_in_thread(func, task_name=None):
        """Function for run in thread.

        Args:
            func: Input value for this operation.
            task_name: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        label = task_name or func.__name__.replace("_", " ").title()

        def wrapper(*args, **kwargs):
            """Function for wrapper.

            Args:
                args: Input value for this operation.
                kwargs: Input value for this operation.
            """
            cancel_event = threading.Event()
            monitor = TaskResourceMonitor(label)
            active_task["cancel_event"] = cancel_event
            active_task["cleanup_paths"] = []
            active_task["monitor"] = monitor
            active_task["cancel_requested"] = False
            monitor.start()
            root.after(0, lambda: begin_task(label, cancel_event, monitor))

            def thread_target():
                """Function for thread target."""
                success = False
                cancelled = False
                metrics = None
                _start_recovery_checkpoint(label)
                try:
                    result = func(*args, **kwargs)
                    success = result is not False
                except OperationCancelled:
                    cancelled = True
                    print(f"\n[CANCELLED] {label}")
                    removed = _remove_cancelled_task_paths()
                    print(f"Cancelled task cleanup removed {removed} partial path(s).")
                except Exception as exc:
                    # Print a clear header for the error
                    print(f"\n[FAILED] {label}\n  See traceback below.\n")
                    # This will dump the full "most recent call last" traceback 
                    # straight to your GUI terminal!
                    traceback.print_exc() 
                finally:
                    metrics = monitor.stop()
                    _finish_recovery_checkpoint(success, cancelled=cancelled)
                    root.after(0, lambda: end_task(success=success, cancelled=cancelled, metrics=metrics))

            threading.Thread(target=thread_target, daemon=True).start()

        return wrapper
    
    temp_directories = []
    current_wavelet_dir = [None]
    analysis_state = {}
    individual_neuron_renderer = {"draw": None}
    embedded_canvases = []
    figure_export_records = []
    active_recovery_dir = {"path": None}

    def _field_value(entries, key, default=""):
        """Function for field value.

        Args:
            entries: Input value for this operation.
            key: Input value for this operation.
            default: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        entry = entries.get(key) if isinstance(entries, dict) else None
        if entry is None:
            return default
        try:
            return entry.get()
        except Exception:
            return default

    def _safe_var_value(name, default=""):
        """Return a Tk variable value if it has already been created."""
        try:
            if name == "gabor_format_var":
                return gabor_format_var.get()
            if name == "wavelet_format_var":
                return wavelet_format_var.get()
            if name == "downsample_format_var":
                return downsample_format_var.get()
            if name == "neural_cache_format_var":
                return neural_cache_format_var.get()
        except Exception:
            return default
        return default

    def _project_root_value():
        """Return the active project root folder."""
        value = _field_value(param_entries, "Project Root", "").strip()
        if value.lower() in ("", "none", "null"):
            value = param_defaults.get("Project Root", DEFAULT_COMMON_PARAMS.get("Project Root", "your_experiment"))
        return value

    def _project_layout():
        """Return the strict folder layout for the active project root."""
        layout = WavenProjectLayout.from_root(_project_root_value())
        layout.ensure()
        return layout

    def _layout_field_folder(field_name):
        """Return the folder conventionally associated with a GUI path field."""
        return _project_layout().dir_for_field(field_name)

    def _set_entry_value(entry, value):
        """Replace one Tk entry value."""
        entry.delete(0, tk.END)
        entry.insert(0, str(value))

    def _apply_project_layout_defaults(force=False):
        """Populate folder fields with the strict project layout."""
        try:
            layout = _project_layout()
        except Exception as exc:
            print(f"Could not apply project layout defaults: {exc}")
            return
        field_map = {
            "Dir": layout.raw_data_dir,
            "Movie Path": layout.stimulus_movie_dir,
            "Path Directory": layout.coarse_wavelet_dir,
            "Library Path": layout.fine_gabor_dir,
            "Full Model Wavelet Path": layout.full_wavelet_dir,
            "Full Model Save Path": layout.model_dir,
            "Plot Cache Path": layout.plots_dir,
            "Recovery Cache Directory": layout.recovery_dir,
        }
        for key, path in field_map.items():
            if key in param_entries:
                current = param_entries[key].get().strip()
                if force or current.lower() in ("", "none", "null") or Path(current).suffix:
                    _set_entry_value(param_entries[key], path)
        gabor_field_map = {
            "Save Path": layout.fine_gabor_dir,
            "Coarse Library Path": layout.coarse_gabor_dir,
            "Fine Library Path": layout.fine_gabor_dir,
        }
        for key, path in gabor_field_map.items():
            if key in gabor_entries:
                current = gabor_entries[key].get().strip()
                if force or current.lower() in ("", "none", "null") or Path(current).suffix:
                    _set_entry_value(gabor_entries[key], path)
        if (
            "Spks Path" in param_entries
            and (
                force
                or param_entries["Spks Path"].get().strip().lower() in ("", "none", "null")
                or Path(param_entries["Spks Path"].get().strip()).suffix
            )
        ):
            _set_entry_value(param_entries["Spks Path"], layout.neural_cache_dir)

    def _folder_from_entry(entries, key, default_folder=None):
        """Return a GUI folder value, falling back to a layout folder."""
        entry = entries.get(key)
        value = entry.get().strip() if entry is not None else ""
        if value.lower() in ("", "none", "null"):
            return str(default_folder or _layout_field_folder(key))
        path = Path(value)
        if path.suffix and not path.is_dir():
            return str(path.parent)
        return str(path)

    def _find_movie_path():
        """Resolve the selected stimulus movie from its folder field."""
        folder = _folder_from_entry(param_entries, "Movie Path", _project_layout().stimulus_movie_dir)
        movie = find_stimulus_movie(folder)
        return str(movie)

    def _gabor_folder():
        """Return the conventional Gabor cache folder."""
        return _folder_from_entry(gabor_entries, "Save Path", _project_layout().fine_gabor_dir)

    def _gabor_folder_for_kind(kind):
        """Return the folder for a coarse or full Gabor library."""
        layout = _project_layout()
        if kind == "coarse":
            return _folder_from_entry(gabor_entries, "Coarse Library Path", layout.coarse_gabor_dir)
        return _folder_from_entry(gabor_entries, "Fine Library Path", layout.fine_gabor_dir)

    def _find_gabor_library(kind, output_format=None):
        """Find the Gabor library artifact for ``kind`` inside its folder."""
        folder = Path(_gabor_folder_for_kind(kind))
        output_format = output_format or _safe_var_value("gabor_format_var", "npy")
        conventional = conventional_gabor_path(folder, kind, output_format)
        if conventional.exists():
            return str(conventional)
        try:
            return str(find_single_artifact(folder, (".npy", ".zarr"), f"{kind} Gabor library"))
        except FileNotFoundError:
            return str(conventional)

    def _wavelet_folder(scale="coarse"):
        """Return the conventional wavelet cache folder for ``scale``."""
        key = "Full Model Wavelet Path" if scale == "full" else "Path Directory"
        default = _project_layout().full_wavelet_dir if scale == "full" else _project_layout().coarse_wavelet_dir
        return _folder_from_entry(param_entries, key, default)

    def _plot_cache_path():
        """Function for plot cache path.

        Returns:
            Result produced by the operation.
        """
        value = _field_value(param_entries, "Plot Cache Path", "").strip()
        if value.lower() in ("", "none", "null"):
            value = str(_project_layout().plots_dir)
        path = Path(value)
        if path.suffix:
            return str(path)
        return str(path / "plot_cache.pkl.gz")

    def _recovery_root():
        """Function for recovery root.

        Returns:
            Result produced by the operation.
        """
        value = _field_value(param_entries, "Recovery Cache Directory", "").strip()
        if value.lower() in ("", "none", "null"):
            return str(_project_layout().recovery_dir)
        path = Path(value)
        return str(path.parent if path.suffix else path)

    def _load_plot_cache():
        """Function for load plot cache.

        Returns:
            Result produced by the operation.
        """
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
        """Function for save plot cache.

        Args:
            cache: Input value for this operation.
        """
        path = _plot_cache_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with gzip.open(tmp_path, "wb") as handle:
            pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)
        print(f"Updated plot cache: {path}")

    def _cache_fingerprint(extra=None):
        """Function for cache fingerprint.

        Args:
            extra: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        fields = {
            "workflow": workflow,
            "wavelet_backend": _selected_wavelet_backend(),
            "gabor_format": _safe_var_value("gabor_format_var", "npy"),
            "neural_cache_format": _selected_neural_cache_format(),
            "downsample_percent": _selected_downsample_percent(),
            "downsample_format": _selected_downsample_format(),
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
        """Function for cache key.

        Args:
            kind: Input value for this operation.
            neuron_id: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if neuron_id is None:
            return kind
        return f"{kind}:neuron:{int(neuron_id)}"

    def _figure_records(items):
        """Function for figure records.

        Args:
            items: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
                    "caption": getattr(fig, "_waven_caption", ""),
                }
            )
        return records

    def _json_safe(value):
        """Function for json safe.

        Args:
            value: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
        """Function for cache safe payload.

        Args:
            payload: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        try:
            pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
            return payload
        except Exception as exc:
            return {
                "cache_note": f"Original payload was not pickleable for plot cache: {exc}",
                "summary": _json_safe(payload),
            }

    def _extract_figure_data(fig):
        """Function for extract figure data.

        Args:
            fig: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
        """Function for add array exports.

        Args:
            prefix: Input value for this operation.
            value: Input value for this operation.
            arrays: Input value for this operation.
            metadata: Input value for this operation.
        """
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
        """Function for tab name for parent.

        Args:
            parent_container: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        try:
            if parent_container is frame_plot_all:
                return "All neurons"
            if parent_container is frame_plot_individual:
                return "Individual neuron"
        except NameError:
            pass
        return "Plots"

    def _figure_title(fig, fallback="Plot"):
        """Function for figure title.

        Args:
            fig: Input value for this operation.
            fallback: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        for ax in fig.axes:
            if ax.get_title():
                return ax.get_title()
        return fallback

    def _set_figure_export_payload(fig, payload):
        """Function for set figure export payload.

        Args:
            fig: Input value for this operation.
            payload: Input value for this operation.
        """
        setattr(fig, "_waven_export_payload", payload)

    def _discrete_groups_from_column(values, max_groups=16):
        """Function for discrete groups from column.

        Args:
            values: Input value for this operation.
            max_groups: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        values = np.asarray(values)
        if values.ndim != 1:
            return None
        finite = np.isfinite(values)
        if not np.any(finite):
            return None
        finite_values = values[finite]
        rounded = np.rint(finite_values)
        if not np.allclose(finite_values, rounded, atol=1e-6):
            return None
        unique = np.unique(rounded.astype(int))
        if unique.size < 2 or unique.size > max_groups:
            return None
        groups = np.full(values.shape[0], -1, dtype=int)
        groups[finite] = rounded.astype(int)
        return groups, [str(value) for value in unique], unique

    def _spatial_quantile_groups(neuron_pos, n_bins=4):
        """Function for spatial quantile groups.

        Args:
            neuron_pos: Input value for this operation.
            n_bins: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        positions = np.asarray(neuron_pos, dtype=float)
        if positions.ndim != 2 or positions.shape[0] == 0:
            return np.zeros(0, dtype=int), [], "none"
        axis = 0
        values = positions[:, axis]
        if np.nanmax(values) == np.nanmin(values) and positions.shape[1] > 1:
            axis = 1
            values = positions[:, axis]
        quantiles = np.unique(np.nanquantile(values, np.linspace(0, 1, n_bins + 1)))
        if quantiles.size <= 2:
            return np.zeros(values.shape[0], dtype=int), ["all"], f"all neurons"
        groups = np.digitize(values, quantiles[1:-1], right=True)
        labels = [f"bin {idx + 1}" for idx in range(np.max(groups) + 1)]
        return groups, labels, f"spatial x-axis bins"

    def _grouping_for_selectivity(neuron_pos, preferred="unit"):
        """Function for grouping for selectivity.

        Args:
            neuron_pos: Input value for this operation.
            preferred: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        positions = np.asarray(neuron_pos)
        if positions.ndim == 2:
            if preferred == "shank":
                candidate_columns = list(range(min(positions.shape[1], 3)))
                max_groups = 8
            else:
                candidate_columns = list(range(3, positions.shape[1])) + list(range(min(positions.shape[1], 3)))
                max_groups = 16
            for col in candidate_columns:
                result = _discrete_groups_from_column(positions[:, col], max_groups=max_groups)
                if result is not None:
                    groups, labels, unique = result
                    label_map = {value: labels[idx] for idx, value in enumerate(unique)}
                    mapped_labels = [label_map[value] for value in unique]
                    return groups, mapped_labels, f"position column {col}"
        groups, labels, source = _spatial_quantile_groups(neuron_pos)
        return groups, labels, source

    def _metric_values(selectivity, name, filter_mask=None):
        """Function for metric values.

        Args:
            selectivity: Input value for this operation.
            name: Input value for this operation.
            filter_mask: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        values = np.asarray(selectivity[name], dtype=float)
        valid = np.isfinite(values)
        if filter_mask is not None:
            valid = np.logical_and(valid, np.asarray(filter_mask, dtype=bool))
        return values[valid], valid

    def _selectivity_statistics(values, label):
        """Return a compact, display-ready descriptive table for OSI/gOSI."""
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            return f"{label} DISTRIBUTION STATISTICS\nNo finite values available."
        q1, q2, q3 = np.percentile(values, [25, 50, 75])
        centered = values - values.mean()
        std = values.std()
        skew = np.mean(centered ** 3) / std ** 3 if std else 0.0
        kurtosis = np.mean(centered ** 4) / std ** 4 - 3 if std else 0.0
        counts, edges = np.histogram(values, bins=np.linspace(0, 1, 21))
        mode = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
        n = values.size
        category = lambda mask: f"{np.count_nonzero(mask)} ({100 * np.mean(mask):.1f}%)"
        return (
            f"{label} DISTRIBUTION STATISTICS\n"
            f"Sample Size — Total units: {n}\n"
            f"Central Tendency — Mean: {values.mean():.3f}; Median: {q2:.3f}; Mode: {mode:.3f}\n"
            f"Spread — Std Dev: {std:.3f}; Variance: {values.var():.3f}; Range: {np.ptp(values):.3f}\n"
            f"Distribution — Min: {values.min():.3f}; Q1 (25%): {q1:.3f}; Q2 (50%): {q2:.3f}; "
            f"Q3 (75%): {q3:.3f}; Max: {values.max():.3f}\n"
            f"Shape — Skewness: {skew:.3f}; Kurtosis: {kurtosis:.3f}\n"
            f"Selectivity Categories — High (>0.5): {category(values > .5)}; "
            f"Medium (0.3–0.5): {category((values >= .3) & (values <= .5))}; "
            f"Low (<0.3): {category(values < .3)}"
        )

    def _plot_kde(ax, values, color, label):
        """Draw a lightweight Gaussian KDE without a SciPy dependency."""
        if values.size < 2 or np.allclose(values, values[0]):
            return
        x = np.linspace(0, 1, 240)
        bandwidth = max(0.03, 1.06 * values.std(ddof=1) * values.size ** (-1 / 5))
        density = np.exp(-0.5 * ((x[:, None] - values[None, :]) / bandwidth) ** 2).mean(axis=1)
        density /= bandwidth * np.sqrt(2 * np.pi)
        # Scale density to histogram counts for a shared y axis.
        ax.plot(x, density * values.size * 0.05, color=color, linewidth=2, label=label)

    def _plot_selectivity_population(selectivity, filter_mask, neuron_pos=None, show_shanks=True):
        """Function for plot selectivity population.

        Args:
            selectivity: Input value for this operation.
            filter_mask: Input value for this operation.
            neuron_pos: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        bins = np.linspace(0, 1, 21)
        # Give each distribution its own statistics panel.  Anchoring two long
        # text blocks below a one-row figure caused them to overlap/squish the
        # OSI and gOSI plots in the GUI and in exports.
        fig, axes = plt.subplots(
            2, 2, figsize=(13, 8.8), constrained_layout=True,
            gridspec_kw={"height_ratios": [3.2, 1.5]},
        )
        plot_axes = axes[0]
        stats_axes = axes[1]
        specs = [("osi", "OSI"), ("gosi", "gOSI")]
        shank_groups = None
        shank_labels = []
        shank_source = "not available"
        if show_shanks and neuron_pos is not None:
            shank_groups, shank_labels, shank_source = _grouping_for_selectivity(neuron_pos, preferred="shank")
        colors = ["#059669", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#BE185D", "#4B5563", "#65A30D"]
        for ax, stats_ax, (key, label) in zip(plot_axes, stats_axes, specs):
            values, valid = _metric_values(selectivity, key)
            filtered, _ = _metric_values(selectivity, key, filter_mask=filter_mask)
            ax.hist(values, bins=bins, color="#2563EB", alpha=0.72, label=f"All neurons (n={values.size})")
            _plot_kde(ax, values, "#DC2626", "KDE")
            if values.size:
                ax.axvline(values.mean(), color="#059669", linewidth=2, label=f"Mean ({values.mean():.3f})")
                ax.axvline(np.median(values), color="#7C3AED", linewidth=2, linestyle="--", label=f"Median ({np.median(values):.3f})")
            if filtered.size:
                ax.hist(filtered, bins=bins, histtype="step", color="#111827", linewidth=1.6, label=f"Quality mask (n={filtered.size})")
            if shank_groups is not None and len(shank_labels) <= 8:
                metric_values = np.asarray(selectivity[key], dtype=float)
                finite_filter = np.logical_and(np.asarray(filter_mask, dtype=bool), np.isfinite(metric_values))
                for group_index, group_id in enumerate([g for g in np.unique(shank_groups) if g >= 0]):
                    group_values = metric_values[np.logical_and(finite_filter, shank_groups == group_id)]
                    if group_values.size:
                        group_label = shank_labels[group_index] if group_index < len(shank_labels) else str(group_id)
                        ax.hist(
                            group_values,
                            bins=bins,
                            histtype="step",
                            color=colors[group_index % len(colors)],
                            linewidth=1.0,
                            alpha=0.85,
                            label=f"Shank {group_label}",
                        )
            ax.set_title(f"{label} distribution")
            ax.set_xlabel(label)
            ax.set_ylabel("Neuron count")
            ax.set_xlim(0, 1)
            ax.legend(fontsize=8)
            stats_ax.axis("off")
            stats_ax.text(
                0.01, 0.98, _selectivity_statistics(values, label),
                transform=stats_ax.transAxes, va="top", ha="left",
                fontsize=8.2, family="monospace",
            )
        fig.suptitle("Orientation Selectivity by Neuron")
        fig._waven_caption = f"Shank grouping source: {shank_source}." if show_shanks else "Two-photon view: all-cell distribution only."
        _set_figure_export_payload(
            fig,
            {
                "source": "Run Coarse RF Analysis",
                "selectivity": selectivity,
                "shank_groups": shank_groups,
                "shank_group_source": shank_source,
                "filter_mask": filter_mask,
            },
        )
        return fig

    def _plot_selectivity_by_unit(selectivity, neuron_pos, filter_mask, unit_ids=None):
        """Function for plot selectivity by unit.

        Args:
            selectivity: Input value for this operation.
            neuron_pos: Input value for this operation.
            filter_mask: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if unit_ids is not None and len(unit_ids) == len(selectivity["osi"]):
            unit_labels_raw = np.asarray(unit_ids, dtype=object)
            unique_labels = list(dict.fromkeys(unit_labels_raw.tolist()))
            # Acquisition unit IDs identify individual neurons, so using them as
            # population groups guarantees a misleading n=1 in every panel.
            if len(unique_labels) < len(unit_labels_raw):
                label_to_group = {label: idx for idx, label in enumerate(unique_labels)}
                unit_groups = np.asarray([label_to_group[label] for label in unit_labels_raw], dtype=int)
                unit_labels = [str(label) for label in unique_labels]
                unit_source = "repeated acquisition unit IDs"
            else:
                unit_groups, unit_labels, unit_source = _grouping_for_selectivity(neuron_pos, preferred="unit")
                unit_source = f"aggregated {unit_source} (unique acquisition IDs)"
        else:
            unit_groups, unit_labels, unit_source = _grouping_for_selectivity(neuron_pos, preferred="unit")
        shank_groups, shank_labels, shank_source = _grouping_for_selectivity(neuron_pos, preferred="shank")
        bins = np.linspace(0, 1, 16)
        finite_any = np.isfinite(np.asarray(selectivity["osi"], dtype=float)) | np.isfinite(np.asarray(selectivity["gosi"], dtype=float))
        unique_units = [group for group in np.unique(unit_groups) if group >= 0]
        if len(unique_units) == 0:
            unique_units = [0]
            unit_groups = np.zeros_like(np.asarray(selectivity["osi"], dtype=int))
            unit_labels = ["all"]
        max_units = min(len(unique_units), 10)
        unique_units = unique_units[:max_units]
        fig, axes = plt.subplots(max_units, 2, figsize=(10, max(3.5, 2.0 * max_units)), constrained_layout=True)
        if max_units == 1:
            axes = np.asarray([axes])
        colors = ["#2563EB", "#059669", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#4B5563", "#BE185D"]
        finite_filter = np.asarray(filter_mask, dtype=bool)
        if not np.any(np.logical_and(finite_filter, finite_any)):
            finite_filter = finite_any
        for row, group_id in enumerate(unique_units):
            group_mask = np.logical_and(unit_groups == group_id, finite_filter)
            unit_label = unit_labels[row] if row < len(unit_labels) else str(group_id)
            for col, (metric, label) in enumerate((("osi", "OSI"), ("gosi", "gOSI"))):
                ax = axes[row, col]
                values = np.asarray(selectivity[metric], dtype=float)
                values = values[np.logical_and(group_mask, np.isfinite(values))]
                ax.hist(values, bins=bins, color=colors[row % len(colors)], alpha=0.76)
                ax.set_xlim(0, 1)
                ax.set_title(f"Unit {unit_label} {label} (n={values.size})")
                ax.set_xlabel(label)
                ax.set_ylabel("Count")
        fig.suptitle(f"Orientation Selectivity by Unit ({unit_source})")
        fig._waven_caption = f"Shank grouping source: {shank_source}; shank labels detected: {', '.join(shank_labels) if shank_labels else 'none'}."
        _set_figure_export_payload(
            fig,
            {
                "source": "Run Coarse RF Analysis",
                "selectivity": selectivity,
                "unit_groups": unit_groups,
                "unit_group_source": unit_source,
                "shank_groups": shank_groups,
                "shank_group_source": shank_source,
                "filter_mask": filter_mask,
            },
        )
        return fig

    def _model_result_payload(model_name, result, neuron_id):
        """Function for model result payload.

        Args:
            model_name: Input value for this operation.
            result: Input value for this operation.
            neuron_id: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
        """Function for active export records.

        Args:
            tab_name: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
        """Function for export figure record.

        Args:
            record: Input value for this operation.
            base_dir: Input value for this operation.
            index: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
        try:
            array_format = export_array_format_var.get()
        except NameError:
            array_format = "npy"
        if array_format not in {"npy", "zarr", "both"}:
            array_format = "npy"
        array_dir = os.path.join(graph_dir, "arrays")
        os.makedirs(array_dir, exist_ok=True)
        array_files = {"npy": [], "zarr": []}
        for key, value in arrays.items():
            safe_key = _safe_name(key)
            array = np.asarray(value)
            if array_format in {"npy", "both"}:
                path = os.path.join(array_dir, f"{safe_key}.npy")
                np.save(path, array)
                array_files["npy"].append(os.path.relpath(path, graph_dir))
            if array_format in {"zarr", "both"}:
                try:
                    import zarr as _zarr
                except ImportError as exc:
                    raise ImportError("Zarr export requires the 'zarr' package.") from exc
                path = os.path.join(array_dir, f"{safe_key}.zarr")
                try:
                    zarr_array = array.astype(str) if array.dtype == object else array
                    _zarr.save(path, zarr_array)
                    array_files["zarr"].append(os.path.relpath(path, graph_dir))
                except Exception as exc:
                    metadata[f"{key}.zarr_export_error"] = str(exc)
                    print(f"  Skipped Zarr array '{key}': {exc}")
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
                "array_format": array_format,
                "array_files": array_files,
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
        """Function for export single graph.

        Args:
            record: Input value for this operation.
        """
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
        """Function for export displayed results.

        Args:
            tab_name: Input value for this operation.
        """
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
        """Function for export all displayed results."""
        _export_displayed_results()

    def export_all_neurons_results():
        """Function for export all neurons results."""
        _export_displayed_results("All neurons")

    def export_individual_neuron_results():
        """Function for export individual neuron results."""
        _export_displayed_results("Individual neuron")

    def export_all_individual_neurons_results():
        """Render/export one neuron per Tk event-loop turn to keep the GUI responsive."""
        draw = individual_neuron_renderer.get("draw")
        n_neurons = np.asarray(analysis_state.get("spks", np.empty((0, 0, 0)))).shape[-1]
        if draw is None or n_neurons <= 0:
            messagebox.showinfo("No Results", "Run coarse RF analysis before exporting all neurons.")
            return
        selected_dir = filedialog.askdirectory(title="Select Folder for All Individual Neurons Export")
        if not selected_dir:
            return
        export_root = os.path.join(selected_dir, f"waven_all_individual_neurons_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(export_root, exist_ok=True)
        try:
            btn_export_all_individual_neurons.configure(state=tk.DISABLED)
        except NameError:
            pass
        print(f"[EXPORT] All individual neurons | count={n_neurons}")
        export_state = {"neuron_id": 0, "failed": None}

        def export_next_neuron():
            """Export one neuron, then yield control back to Tk before continuing."""
            neuron_id = export_state["neuron_id"]
            if neuron_id >= n_neurons:
                try:
                    btn_export_all_individual_neurons.configure(state=tk.NORMAL)
                except NameError:
                    pass
                if export_state["failed"] is None:
                    print(f"[DONE] Exported {n_neurons} individual-neuron result sets\n       {export_root}")
                    messagebox.showinfo("Export Complete", f"Exported {n_neurons} individual-neuron result sets.\n\n{export_root}")
                return
            try:
                draw(neuron_id, switch_tab=False)
                neuron_dir = os.path.join(export_root, f"neuron_{neuron_id:05d}")
                os.makedirs(neuron_dir, exist_ok=True)
                for index, record in enumerate(_active_export_records("Individual neuron"), start=1):
                    _export_figure_record(record, neuron_dir, index=index)
                export_state["neuron_id"] += 1
                percent = 100.0 * export_state["neuron_id"] / n_neurons
                update_progress(percent, "Exporting individual neurons", f"Neuron {export_state['neuron_id']}/{n_neurons}")
                # A zero-delay timer lets paint/input events run between neurons.
                root.after(1, export_next_neuron)
            except Exception as exc:
                export_state["failed"] = exc
                try:
                    btn_export_all_individual_neurons.configure(state=tk.NORMAL)
                except NameError:
                    pass
                print(f"[FAILED] Individual-neuron export at neuron {neuron_id}: {exc}")
                messagebox.showerror("Export Failed", f"Stopped at neuron {neuron_id}: {exc}")

        root.after_idle(export_next_neuron)

    def _render_figure_records(records, message="Loaded plots from cache.", clear=True):
        """Function for render figure records.

        Args:
            records: Input value for this operation.
            message: Input value for this operation.
            clear: Input value for this operation.
        """
        def render():
            """Function for render."""
            _ensure_plot_imports()
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
                    if record.get("caption"):
                        fig._waven_caption = record.get("caption")
                except Exception as exc:
                    print(f"Could not restore cached figure: {exc}")
                    continue
                parent = frame_plot_all if record.get("tab") == "all" else frame_plot_individual
                embed_interactive_figure(fig, parent, record.get("title"))
            print(message)
        root.after(0, render)

    def _state_for_plot_cache(state):
        """Function for state for plot cache.

        Args:
            state: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        excluded = {"wavelets_complex"}
        cached = {}
        for key, value in state.items():
            if key in excluded:
                continue
            cached[key] = value
        return cached

    def _get_cached_entry(kind, neuron_id=None, extra=None):
        """Function for get cached entry.

        Args:
            kind: Input value for this operation.
            neuron_id: Input value for this operation.
            extra: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        cache = _load_plot_cache()
        entry = cache.get("entries", {}).get(_cache_key(kind, neuron_id))
        if not entry:
            return None
        if entry.get("fingerprint") != _cache_fingerprint(extra=extra):
            return None
        return entry

    def _put_cached_entry(kind, entry, neuron_id=None, extra=None):
        """Function for put cached entry.

        Args:
            kind: Input value for this operation.
            entry: Input value for this operation.
            neuron_id: Input value for this operation.
            extra: Input value for this operation.
        """
        cache = _load_plot_cache()
        cache.setdefault("entries", {})[_cache_key(kind, neuron_id)] = {
            **entry,
            "fingerprint": _cache_fingerprint(extra=extra),
            "saved_at": time.time(),
        }
        _save_plot_cache(cache)

    def _start_recovery_checkpoint(task_name):
        """Function for start recovery checkpoint.

        Args:
            task_name: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
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
        """Function for write recovery step.

        Args:
            step: Input value for this operation.
            data: Input value for this operation.
        """
        task_dir = active_recovery_dir.get("path")
        if not task_dir:
            return
        payload = {"step": step, "time": time.time(), **data}
        with open(os.path.join(task_dir, f"{_safe_name(step)}.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, default=str)

    def _recovery_subdir(name):
        """Function for recovery subdir.

        Args:
            name: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        task_dir = active_recovery_dir.get("path") or _recovery_root()
        path = os.path.join(task_dir, _safe_name(name))
        os.makedirs(path, exist_ok=True)
        return path

    def _finish_recovery_checkpoint(success, cancelled=False):
        """Function for finish recovery checkpoint.

        Args:
            success: Input value for this operation.
            cancelled: Input value for this operation.
        """
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
            manifest["status"] = "complete" if success else ("cancelled" if cancelled else "failed")
            manifest["finished_at"] = time.time()
            with open(manifest_path, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2)
        except Exception:
            pass
        if success or cancelled:
            try:
                shutil.rmtree(task_dir)
                reason = "successful" if success else "cancelled"
                print(f"Removed {reason} recovery checkpoint: {task_dir}")
            except Exception as exc:
                print(f"Could not remove recovery checkpoint {task_dir}: {exc}")
        else:
            print(f"Kept recovery checkpoint after failure: {task_dir}")

    def _library_output_path(kind, base_path):
        """Function for library output path.

        Args:
            kind: Input value for this operation.
            base_path: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        folder = _gabor_folder_for_kind(kind)
        return str(conventional_gabor_path(folder, kind, gabor_format_var.get()))

    def _save_library_array(library, path_save, description):
        """Function for save library array.

        Args:
            library: Input value for this operation.
            path_save: Input value for this operation.
            description: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        update_progress(70, f"Saving {description}", "Writing output file")
        os.makedirs(os.path.dirname(path_save) or ".", exist_ok=True)
        if gabor_format_var.get() == "zarr":
            try:
                import zarr as _zarr
            except ImportError as exc:
                raise ImportError(
                    "Zarr library output requires the 'zarr' package. "
                    "Install project requirements or select 'npy' as the library format."
                ) from exc
            output_path = os.path.splitext(path_save)[0] + ".zarr"
            if not os.path.exists(output_path):
                _register_cancel_cleanup_path(output_path)
            print(f"Saving {description} into Zarr container: {output_path}")
            _raise_if_cancelled()
            _zarr.save(output_path, library)
        else:
            output_path = path_save
            if not os.path.exists(output_path):
                _register_cancel_cleanup_path(output_path)
            _raise_if_cancelled()
            np.save(output_path, library)
            print(f"{description} saved to: {output_path}")
        return output_path

    def _artifact_shape(path):
        """Function for artifact shape.

        Args:
            path: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if not path or not os.path.exists(path):
            return None
        try:
            if str(path).endswith(".zarr") or os.path.isdir(path):
                import zarr as _zarr

                return tuple(_zarr.open(path, mode="r").shape)
            return tuple(np.load(path, mmap_mode="r").shape)
        except Exception as exc:
            print(f"Could not read artifact shape for {path}: {exc}")
            return None

    def _artifact_matches(path, expected_shape):
        """Function for artifact matches.

        Args:
            path: Input value for this operation.
            expected_shape: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        shape = _artifact_shape(path)
        if shape is None:
            return False
        expected_shape = tuple(int(dim) for dim in expected_shape)
        if tuple(shape) != expected_shape:
            print(f"Existing artifact has shape {shape}, expected {expected_shape}: {path}")
            return False
        return True

    def _artifact_meta_path(path):
        """Return the sidecar metadata path for an artifact."""
        return f"{path}.waven.json"

    def _read_artifact_metadata(path):
        """Read artifact sidecar metadata when it exists."""
        meta_path = _artifact_meta_path(path)
        if not path or not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:
            print(f"Could not read artifact metadata {meta_path}: {exc}")
            return None

    def _write_artifact_metadata(path, kind, expected_shape, fingerprint, params=None):
        """Write small JSON metadata used to decide whether an artifact is reusable."""
        if not path:
            return
        payload = {
            "version": 1,
            "kind": kind,
            "shape": tuple(int(dim) for dim in expected_shape),
            "fingerprint": fingerprint,
            "params": params or {},
        }
        meta_path = _artifact_meta_path(path)
        os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _artifact_ready(path, expected_shape, fingerprint=None, kind=None):
        """Return whether an artifact shape and optional metadata are current."""
        if not _artifact_matches(path, expected_shape):
            return False
        if fingerprint is None:
            return True
        metadata = _read_artifact_metadata(path)
        if metadata is None:
            print(f"Existing artifact has no parameter metadata, regenerating: {path}")
            return False
        if kind is not None and metadata.get("kind") != kind:
            print(f"Existing artifact metadata kind changed, regenerating: {path}")
            return False
        if metadata.get("fingerprint") != fingerprint:
            print(f"Existing artifact parameters changed, regenerating: {path}")
            return False
        return True

    def _library_artifact_path(path_save):
        """Function for library artifact path.

        Args:
            path_save: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        return path_save

    def _convolution_kernel_cache_output_path(kind):
        """Return where the compact convolution kernel cache should be stored."""
        folder_path = str(_project_layout().gabor_kernel_dir)
        return convolution_kernel_cache_path(folder_path, kind)

    def _ensure_convolution_kernel_cache(kind, force=False):
        """Build or reuse the compact convolution kernel cache for ``kind``."""
        _ensure_wavelet_imports("convolution kernel cache construction")
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        phase_offsets = _gabor_phase_offsets_radians()
        n_theta = int(gabor_entries["N_thetas"].get())
        kind = kind.lower()
        if kind == "fine":
            sigmas = _ordered_float_union(
                sigmas,
                parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"),
            )
            cache_frequencies = frequencies
        elif kind == "coarse":
            cache_frequencies = []
        else:
            raise ValueError(f"Unknown convolution kernel cache kind: {kind}")

        cache_path = _convolution_kernel_cache_output_path(kind)
        folder_path = os.path.dirname(cache_path) or "."
        return build_convolution_kernel_cache(
            folder_path,
            kind,
            sigmas,
            n_theta,
            phase_offsets=phase_offsets,
            frequencies=cache_frequencies,
            force=force,
            cancel_event=_current_cancel_event(),
        )

    def _selected_analysis_scale():
        """Compatibility value for legacy callers; the GUI now uses one shared grid."""
        return "coarse"

    def _selected_wavelet_backend():
        """Return the selected wavelet decomposition backend."""
        try:
            value = wavelet_backend_var.get()
        except NameError:
            return "legacy"
        return value if value in {"legacy", "convolution"} else "legacy"

    def _selected_neural_cache_format():
        """Return the selected aligned neural-cache output format."""
        try:
            value = neural_cache_format_var.get()
        except NameError:
            return "npy"
        return value if value in {"npy", "zarr"} else "npy"

    def _selected_neural_source():
        """Return whether neural cache creation should use data dir or cache path."""
        try:
            value = neural_source_var.get()
        except NameError:
            return "data_dir"
        return value if value in {"data_dir", "spks_path"} else "data_dir"

    def _selected_downsample_format():
        """Return the selected precomputed downsampled movie output format."""
        try:
            value = downsample_format_var.get()
        except NameError:
            return "npy"
        return value if value in {"npy", "zarr"} else "npy"

    def _selected_downsample_percent():
        """Return the selected spatial downsample percentage."""
        try:
            value = float(downsample_percent_var.get())
        except Exception:
            return 20.0
        return max(0.0, min(100.0, value))

    def _movie_metadata(path=None):
        """Return authoritative stimulus dimensions, frame count, and FPS."""
        return read_movie_metadata(path or _find_movie_path())

    def _stimulus_grid_dimensions(scale=None, movie_path=None):
        """Derive analysis-grid dimensions solely from movie metadata and percentage."""
        return downsampled_grid_dimensions(
            _movie_metadata(movie_path), _selected_downsample_percent()
        )

    def _gabor_phase_offsets_radians():
        """The GUI accepts degrees; skimage convolution kernels require radians."""
        return np.deg2rad(parse_literal(gabor_entries["Phases"].get(), "Phases (degrees)"))

    def _analysis_grid_dimensions(_nx=None, _ny=None, scale=None):
        """Compatibility wrapper; dimensions always come from movie metadata."""
        return _stimulus_grid_dimensions(scale)

    def _downsample_video_path(movpath, scale=None, output_format=None):
        """Return the selected downsampled-video artifact path."""
        scale = scale or _selected_analysis_scale()
        output_format = output_format or _selected_downsample_format()
        percent = int(round(_selected_downsample_percent()))
        return str(conventional_downsample_path(_wavelet_folder(scale), scale, output_format, percent))

    def _downsample_cache_candidates(movie_path, scale, preferred_format=None):
        """Return current and legacy downsample-cache candidates in priority order."""
        preferred_format = preferred_format or _selected_downsample_format()
        alternate_format = "zarr" if preferred_format == "npy" else "npy"
        percent = int(round(_selected_downsample_percent()))
        wavelet_folder = Path(_wavelet_folder(scale))
        movie = Path(movie_path)
        legacy_input_folder = _project_layout().input_dir
        candidates = [
            conventional_downsample_path(wavelet_folder, scale, preferred_format, percent),
            conventional_downsample_path(wavelet_folder, scale, alternate_format, percent),
        ]
        # Coarse and full-model products intentionally share the one metadata-
        # derived stimulus grid.  A full-model run may therefore reuse the
        # single cache prepared in Step 2 instead of creating a duplicate.
        if scale == "full":
            shared_folder = Path(_wavelet_folder("coarse"))
            candidates.extend(
                (
                    conventional_downsample_path(shared_folder, "coarse", preferred_format, percent),
                    conventional_downsample_path(shared_folder, "coarse", alternate_format, percent),
                )
            )
        for suffix in (".npy", ".zarr"):
            if scale == "coarse":
                candidates.extend(
                    (
                        movie.with_name(f"{movie.stem}_coarse_downsampled{suffix}"),
                        wavelet_folder / f"{movie.stem}_coarse_downsampled{suffix}",
                        legacy_input_folder / f"{movie.stem}_coarse_downsampled{suffix}",
                    )
                )
            else:
                candidates.extend(
                    (
                        movie.with_name(f"{movie.stem}_downsampled{suffix}"),
                        movie.with_name(f"{movie.stem}_full_downsampled{suffix}"),
                        wavelet_folder / f"{movie.stem}_downsampled{suffix}",
                        wavelet_folder / f"{movie.stem}_full_downsampled{suffix}",
                        legacy_input_folder / f"{movie.stem}_downsampled{suffix}",
                        legacy_input_folder / f"{movie.stem}_full_downsampled{suffix}",
                    )
                )
        if wavelet_folder.exists():
            candidates.extend(sorted(wavelet_folder.glob(f"stimulus_{scale}_downsampled_p*.npy")))
            candidates.extend(sorted(wavelet_folder.glob(f"stimulus_{scale}_downsampled_p*.zarr")))
        unique = []
        seen = set()
        for candidate in candidates:
            normalized = os.path.normcase(os.path.abspath(str(candidate)))
            if normalized not in seen:
                seen.add(normalized)
                unique.append(Path(candidate))
        return unique

    def _find_compatible_downsample_cache(movie_path, scale, expected_shape, preferred_format=None):
        """Find a current or legacy NPY/Zarr movie cache with the required shape."""
        for candidate in _downsample_cache_candidates(movie_path, scale, preferred_format):
            if not candidate.exists():
                continue
            if not _artifact_matches(str(candidate), expected_shape):
                continue
            metadata = _read_artifact_metadata(str(candidate))
            if metadata is not None:
                if metadata.get("kind") not in {None, "downsampled_video"}:
                    continue
                params = metadata.get("params") or {}
                if params.get("scale") not in {None, scale, "coarse"}:
                    continue
            actual_format = "zarr" if candidate.suffix.lower() == ".zarr" else "npy"
            print(
                f"  Reusing {scale} stimulus cache ({actual_format.upper()}):\n"
                f"    {candidate}"
            )
            if actual_format != _selected_downsample_format():
                root.after(0, lambda value=actual_format: downsample_format_var.set(value))
            return str(candidate), actual_format
        return None, preferred_format or _selected_downsample_format()

    def _coverage_ratios_for_values(visual_coverage, analysis_coverage):
        """Return x/y coverage ratios used by video downsampling."""
        return coverage_ratios(visual_coverage, analysis_coverage)

    def _scale_label(scale=None):
        """Return a short user-facing label for an analysis scale."""
        scale = scale or _selected_analysis_scale()
        return "Full model" if scale == "full" else "Coarse RF"

    def _selected_gabor_kind():
        """Return the library kind required by the current analysis scale."""
        return "fine" if _selected_analysis_scale() == "full" else "coarse"

    def create_selected_gabor_library():
        """Build the Gabor library required by the selected analysis scale."""
        if _selected_wavelet_backend() == "convolution":
            kind = _selected_gabor_kind()
            description = "fine full-model" if kind == "fine" else "coarse RF"
            update_progress(5, "Convolution kernel cache", f"Building {description} compact kernels")
            print(
                "Convolution backend selected: building compact Gabor kernels instead "
                "of the giant flattened legacy library."
            )
            cache_path = _ensure_convolution_kernel_cache(kind)
            print(f"Convolution kernel cache ready: {cache_path}")
            update_progress(100, "Convolution kernel cache", "Compact kernels ready")
            return
        create_gabor(_selected_gabor_kind())

    def create_gabor(kind="fine"):
        """Function for create gabor.

        Args:
            kind: Input value for this operation.
        """
        _ensure_gabor_imports("Gabor library construction")
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        full_nx, full_ny = _stimulus_grid_dimensions(movie_path=_find_movie_path())
        n_theta = int(gabor_entries["N_thetas"].get())
        offsets = _gabor_phase_offsets_radians()
        path_save = gabor_entries["Save Path"].get()
        kind = kind.lower()
        if kind not in {"coarse", "fine"}:
            raise ValueError(f"Unknown Gabor library kind: {kind}")
        if kind == "coarse":
            nx, ny = _analysis_grid_dimensions(full_nx, full_ny, "coarse")
            path_save = _library_output_path("coarse", path_save)
            description = f"coarse coupled Gabor library ({nx} x {ny})"
        else:
            nx, ny = _analysis_grid_dimensions(full_nx, full_ny, "full")
            sigmas = _ordered_float_union(
                sigmas,
                parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"),
            )
            path_save = _library_output_path("fine", path_save)
            description = f"fine independent-frequency Gabor library ({nx} x {ny})"
        xs = np.arange(nx)
        ys = np.arange(ny)
        thetas = np.array([(i * np.pi) / n_theta for i in range(n_theta)])
        sigmas = np.array(sigmas)
        offsets = np.array(offsets)
        frequencies = np.array(frequencies)

        _, expected_shape = _gabor_library_npy_bytes(
            nx,
            ny,
            n_theta,
            sigmas,
            offsets,
            frequencies if kind == "fine" else [0],
        )
        output_path = _library_artifact_path(path_save)
        library_fingerprint = _cache_fingerprint(
            {
                "artifact": "gabor_library",
                "kind": kind,
                "shape": expected_shape,
                "grid": (nx, ny),
            }
        )
        if _artifact_ready(output_path, expected_shape, library_fingerprint, kind=f"{kind}_gabor_library"):
            print(f"Resume: found completed {description}, reusing {output_path}")
            _write_recovery_step(f"{kind}_gabor_reused", path=output_path, shape=expected_shape)
            entry_key = "Coarse Library Path" if kind == "coarse" else "Fine Library Path"
            if entry_key in gabor_entries:
                _set_entry_value(gabor_entries[entry_key], _gabor_folder_for_kind(kind))
            if kind == "fine" and "Library Path" in param_entries:
                _set_entry_value(param_entries["Library Path"], _gabor_folder_for_kind(kind))
            update_progress(100, f"{description} already complete")
            return

        update_progress(5, f"Building {description}", "Estimating filter bank")
        _raise_if_cancelled()
        if kind == "fine" and frequencies.size and np.any(frequencies != 0):
            L = makeFilterLibrary2(
                xs,
                ys,
                thetas,
                sigmas,
                offsets,
                frequencies,
                cancel_event=_current_cancel_event(),
            )
        else:
            frequency = frequencies[0] if frequencies.size else 0
            L = makeFilterLibrary(
                xs,
                ys,
                thetas,
                sigmas,
                offsets,
                frequency,
                freq=False,
                cancel_event=_current_cancel_event(),
            )

        _raise_if_cancelled()
        output_path = _save_library_array(L, path_save, description)
        _write_artifact_metadata(
            output_path,
            f"{kind}_gabor_library",
            expected_shape,
            library_fingerprint,
            params={"kind": kind, "grid": (nx, ny)},
        )
        entry_key = "Coarse Library Path" if kind == "coarse" else "Fine Library Path"
        if entry_key in gabor_entries:
            _set_entry_value(gabor_entries[entry_key], _gabor_folder_for_kind(kind))
        if kind == "fine" and "Library Path" in param_entries:
            _set_entry_value(param_entries["Library Path"], _gabor_folder_for_kind(kind))
            refresh_size_estimates()
        update_progress(100, f"{description} complete")

    def create_both_gabor_libraries():
        """Prepare the coarse and full-model Gabor assets for the shared grid."""
        if _selected_wavelet_backend() == "convolution":
            _ensure_wavelet_imports("convolution kernel cache construction")
            _ensure_convolution_kernel_cache("coarse")
            _ensure_convolution_kernel_cache("fine")
            update_progress(100, "Convolution kernel caches", "Coarse and full-model kernels ready")
            return
        create_gabor("coarse")
        create_gabor("fine")

    def create_downsampled_video_cache(scale=None):
        """Create or reuse the selected downsampled stimulus movie cache."""
        _ensure_wavelet_imports("stimulus movie downsampling")
        _raise_if_cancelled()
        scale = scale or _selected_analysis_scale()
        movpath = _find_movie_path()
        target_nx, target_ny = _stimulus_grid_dimensions(scale, movpath)
        expected_frames = _movie_metadata(movpath)["frames"]
        output_format = _selected_downsample_format()
        downsample_path = _downsample_video_path(movpath, scale, output_format)
        expected_shape = (expected_frames, target_ny, target_nx)
        visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
        analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")
        reusable_path, reusable_format = _find_compatible_downsample_cache(
            movpath, scale, expected_shape, output_format
        )
        if reusable_path:
            # Mark this prerequisite immediately.  This makes the next stage
            # available even when the cache was created in an earlier session.
            completed_actions.add("downsample:coarse")
            update_progress(100, "Stimulus downsample", f"Existing {reusable_format.upper()} cache ready")
            return True

        downsample_fingerprint = _cache_fingerprint(
            {
                "artifact": "downsampled_video",
                "scale": scale,
                "output_format": output_format,
                "shape": expected_shape,
            }
        )

        update_progress(5, "Stimulus downsample", f"Preparing {scale} {output_format.upper()} cache")
        ratio_x, ratio_y = _coverage_ratios_for_values(visual_coverage, analysis_coverage)
        _register_cancel_cleanup_path(downsample_path)
        print(
            f"Downsampling video for {scale}: {target_nx} x {target_ny} "
            f"({int(round(_selected_downsample_percent()))}% slider, {output_format.upper()})"
        )
        downsample_video_binary(
            movpath,
            visual_coverage,
            analysis_coverage,
            shape=(target_ny, target_nx),
            chunk_size=video_downsample_chunk_size(),
            ratios=(ratio_x, ratio_y),
            save_path=downsample_path,
            output_format=output_format,
            cancel_event=_current_cancel_event(),
        )
        if not _artifact_matches(downsample_path, expected_shape):
            raise ValueError(f"Downsampled video cache has an unexpected shape: {downsample_path}")
        _write_artifact_metadata(
            downsample_path,
            "downsampled_video",
            expected_shape,
            downsample_fingerprint,
            params={"scale": scale, "format": output_format, "grid": (target_nx, target_ny)},
        )
        completed_actions.add("downsample:coarse")
        update_progress(100, "Stimulus downsample", "Downsampled movie ready")
        return True

    def run_wavelet(product="coarse_rf"):
        """Function for run wavelet.

        Returns:
            Result produced by the operation.
        """
        _ensure_wavelet_imports("stimulus wavelet generation")
        _raise_if_cancelled()
        product = str(product or "coarse_rf").lower()
        if product not in {"coarse_rf", "model", "full_model"}:
            raise ValueError(f"Unknown wavelet product: {product}")
        scale = "full" if product == "full_model" else "coarse"
        backend = _selected_wavelet_backend()
        if scale not in {"coarse", "full"}:
            raise ValueError(f"Unknown wavelet decomposition scale: {scale}")
        if backend not in {"legacy", "convolution"}:
            raise ValueError(f"Unknown wavelet decomposition backend: {backend}")
        try:
            movpath = _find_movie_path()
        except Exception as exc:
            print(f"Error: {exc}")
            return False

        # Internal wavelet artifacts are always Zarr: they are large, hidden
        # cache products and must stay bounded/disk-backed under heavy loads.
        output_format = "zarr"
        is_zarr_wavelet = output_format == "zarr"
        downsample_output_format = _selected_downsample_format()
        wavelet_folder = _wavelet_folder("coarse")
        os.makedirs(wavelet_folder, exist_ok=True)

        current_wavelet_dir[0] = wavelet_folder

        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        phase_offsets = _gabor_phase_offsets_radians()
        fine_library_sigmas = _ordered_float_union(
            sigmas,
            parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"),
        )
        coarse_lib_path = _find_gabor_library("coarse")
        fine_lib_path = _find_gabor_library("fine")
        if backend == "legacy" and scale == "coarse" and not coarse_lib_path:
            raise ValueError("Coarse Library Path is required. Select Coarse RF and click Build Gabor Library first.")
        if backend == "legacy" and scale == "full" and not fine_lib_path:
            raise ValueError("Fine Library Path is required. Select Full model and click Build Gabor Library first.")
        coarse_kernel_cache_path = None
        fine_kernel_cache_path = None
        if backend == "convolution":
            cache_kind = "fine" if scale == "full" else "coarse"
            update_progress(4, "Convolution kernel cache", "Checking compact kernel cache")
            cache_path = _ensure_convolution_kernel_cache(cache_kind)
            if scale == "full":
                fine_kernel_cache_path = cache_path
            else:
                coarse_kernel_cache_path = cache_path
        n_thetas = int(gabor_entries["N_thetas"].get())
        coarse_nx, coarse_ny = _stimulus_grid_dimensions("coarse", movpath)
        full_nx, full_ny = _stimulus_grid_dimensions("full", movpath)
        expected_frames = _movie_metadata(movpath)["frames"]
        full_downsample_path = _downsample_video_path(movpath, "full", downsample_output_format)
        coarse_downsample_path = _downsample_video_path(movpath, "coarse", downsample_output_format)

        visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
        analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")
        coarse_downsample_shape = (expected_frames, coarse_ny, coarse_nx)
        full_downsample_shape = (expected_frames, full_ny, full_nx)
        coarse_downsample_ready = False
        full_downsample_ready = False
        coarse_downsample_format = downsample_output_format
        full_downsample_format = downsample_output_format
        if scale == "coarse":
            found_path, coarse_downsample_format = _find_compatible_downsample_cache(
                movpath, "coarse", coarse_downsample_shape, downsample_output_format
            )
            if found_path:
                coarse_downsample_path = found_path
                coarse_downsample_ready = True
        else:
            found_path, full_downsample_format = _find_compatible_downsample_cache(
                movpath, "full", full_downsample_shape, downsample_output_format
            )
            if found_path:
                full_downsample_path = found_path
                full_downsample_ready = True
        coarse_downsample_fingerprint = _cache_fingerprint(
            {
                "artifact": "downsampled_video",
                "scale": "coarse",
                "output_format": coarse_downsample_format,
                "shape": coarse_downsample_shape,
            }
        )
        full_downsample_fingerprint = _cache_fingerprint(
            {
                "artifact": "downsampled_video",
                "scale": "full",
                "output_format": full_downsample_format,
                "shape": full_downsample_shape,
            }
        )

        def _coverage_ratios():
            """Function for coverage ratios.

            Returns:
                Result produced by the operation.
            """
            return _coverage_ratios_for_values(visual_coverage, analysis_coverage)

        def _load_downsampled_movie(path, *, streaming=False):
            """Load a binary movie safely, retaining disk backing for convolution."""
            from ..storage.array_store import load_array
            from ..storage.binary_movie import SignedBinaryMovie

            arr = load_array(path, mmap_mode="r")
            if streaming:
                # Convolution consumes frame chunks.  This adapter converts only
                # the requested chunk, avoiding a full cache-sized allocation.
                return SignedBinaryMovie(arr)
            from ..runtime.performance import has_enough_ram
            required_bytes = int(np.prod(arr.shape, dtype=np.int64))
            if not has_enough_ram(required_bytes, safety_margin=1.50):
                raise MemoryError(
                    "Legacy wavelet decomposition needs the signed stimulus movie in RAM "
                    f"({required_bytes / 1024**3:.2f} GiB). Choose the convolution backend "
                    "for disk-streamed execution or reduce the downsampling percentage."
                )
            return np.asarray(arr, dtype=np.int8) * 2 - 1

        coarse_power_path = os.path.join(wavelet_folder, "coarse_rf_power.zarr")
        model_real_path = os.path.join(wavelet_folder, "coarse_model_real.zarr")
        model_imag_path = os.path.join(wavelet_folder, "coarse_model_imag.zarr")
        temp_real_path = os.path.join(wavelet_folder, ".coarse-rf-phase-real.zarr")
        temp_imag_path = os.path.join(wavelet_folder, ".coarse-rf-phase-imag.zarr")
        real_phase_path = model_real_path if product == "model" else temp_real_path
        imag_phase_path = model_imag_path if product == "model" else temp_imag_path
        coarse_phase_shape = (
            expected_frames,
            coarse_nx,
            coarse_ny,
            n_thetas,
            len(sigmas),
        )
        coarse_power_shape = coarse_phase_shape
        coarse_phase_fingerprint = _cache_fingerprint(
            {
                "artifact": "coarse_wavelet_phase",
                "backend": backend,
                "shape": coarse_phase_shape,
                "downsample": coarse_downsample_fingerprint,
            }
        )
        coarse_power_fingerprint = _cache_fingerprint(
            {
                "artifact": "coarse_rf_power",
                "backend": backend,
                "shape": coarse_power_shape,
                "phase": coarse_phase_fingerprint,
            }
        )

        def _build_coarse_power_zarr(real_path, imag_path, power_path):
            """Stream phase magnitudes into the RF-only Zarr product."""
            from ..storage.array_store import load_array
            try:
                import zarr
                from numcodecs import Blosc
            except ImportError as exc:
                raise ImportError("Coarse RF power cache requires zarr and numcodecs.") from exc
            real = load_array(real_path, mmap_mode="r")
            imag = load_array(imag_path, mmap_mode="r")
            if tuple(real.shape) != coarse_phase_shape or tuple(imag.shape) != coarse_phase_shape:
                raise ValueError(
                    "Coarse phase dimensions do not match the movie-derived analysis grid: "
                    f"expected {coarse_phase_shape}, got {real.shape} and {imag.shape}."
                )
            chunks = (min(expected_frames, 128), min(coarse_nx, 16), min(coarse_ny, 16), n_thetas, len(sigmas))
            if os.path.exists(power_path):
                shutil.rmtree(power_path)
            zarr_kwargs = dict(
                mode="w", shape=coarse_power_shape, chunks=chunks, dtype=np.float32,
                compressor=Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE),
            )
            try:
                power = zarr.open(power_path, **zarr_kwargs)
            except TypeError:
                compressor = zarr_kwargs.pop("compressor")
                zarr_kwargs["compressors"] = [compressor]
                power = zarr.open(power_path, **zarr_kwargs)
            for t0 in range(0, expected_frames, chunks[0]):
                _raise_if_cancelled()
                t1 = min(expected_frames, t0 + chunks[0])
                for x0 in range(0, coarse_nx, chunks[1]):
                    x1 = min(coarse_nx, x0 + chunks[1])
                    for y0 in range(0, coarse_ny, chunks[2]):
                        y1 = min(coarse_ny, y0 + chunks[2])
                        real_chunk = np.asarray(real[t0:t1, x0:x1, y0:y1, :, :], dtype=np.float32)
                        imag_chunk = np.asarray(imag[t0:t1, x0:x1, y0:y1, :, :], dtype=np.float32)
                        power[t0:t1, x0:x1, y0:y1, :, :] = real_chunk * real_chunk + imag_chunk * imag_chunk
            del power, real, imag
            gc.collect()
        if scale == "coarse":
            current_wavelet_dir[0] = wavelet_folder
            requested_path = coarse_power_path if product == "coarse_rf" else model_real_path
            requested_shape = coarse_power_shape if product == "coarse_rf" else coarse_phase_shape
            requested_fingerprint = (
                coarse_power_fingerprint if product == "coarse_rf"
                else _cache_fingerprint({"phase": 0, "base": coarse_phase_fingerprint})
            )
            requested_kind = "coarse_rf_power" if product == "coarse_rf" else "coarse_model_phase"
            coarse_cache_ready = _artifact_ready(
                requested_path, requested_shape, requested_fingerprint, kind=requested_kind
            )
            if product == "model":
                coarse_cache_ready = coarse_cache_ready and _artifact_ready(
                    model_imag_path,
                    coarse_phase_shape,
                    _cache_fingerprint({"phase": 1, "base": coarse_phase_fingerprint}),
                    kind="coarse_model_phase",
                )

            if coarse_cache_ready:
                update_progress(80, "Coarse wavelet decomposition", f"Reusing {product.replace('_', ' ')} cache")
                print(f"Resume: found completed {product} wavelet product, reusing {requested_path}")
                _write_recovery_step("coarse_product_reused", path=requested_path, shape=requested_shape)
            else:
                update_progress(8, "Coarse wavelet decomposition", "Preparing coarse stimulus movie")
                print("Step 1/4: Preparing coarse stimulus movie...")
                if not os.path.exists(coarse_downsample_path):
                    _register_cancel_cleanup_path(coarse_downsample_path)
                if coarse_downsample_ready:
                    _write_recovery_step("coarse_downsampling_reused", path=coarse_downsample_path)
                else:
                    raise FileNotFoundError(
                        "The stimulus downsample stage is incomplete. Run 'Prepare Stimulus Cache' "
                        "in Stage 2 before wavelet decomposition."
                    )
                if not _artifact_matches(coarse_downsample_path, coarse_downsample_shape):
                    raise FileNotFoundError(
                        "Coarse downsampled stimulus was not created. "
                        f"Check that Movie Path points to a readable video: {movpath}"
                    )
                _raise_if_cancelled()
                videodata = _load_downsampled_movie(
                    coarse_downsample_path, streaming=(backend == "convolution")
                )

                if product == "coarse_rf" and backend == "convolution":
                    update_progress(25, "Coarse wavelet decomposition", "Writing direct RF power cache")
                    print("Step 2/2: Writing direct coarse RF power (no temporary phase caches)...")
                    _register_cancel_cleanup_path(coarse_power_path)
                    waveletPowerDecompositionConv(
                        videodata,
                        sigmas,
                        wavelet_folder,
                        n_orientations=n_thetas,
                        phase_offsets=phase_offsets,
                        kernel_cache_path=coarse_kernel_cache_path,
                        output_stem=os.path.splitext(os.path.basename(coarse_power_path))[0],
                        cancel_event=_current_cancel_event(),
                    )
                    _raise_if_cancelled()
                    if not _artifact_matches(coarse_power_path, coarse_power_shape):
                        raise ValueError(f"Direct coarse RF power cache has an unexpected shape: {coarse_power_path}")
                    _write_artifact_metadata(
                        coarse_power_path, "coarse_rf_power", coarse_power_shape, coarse_power_fingerprint
                    )
                    _write_recovery_step("coarse_rf_power_complete", path=coarse_power_path)
                    print(f"Direct coarse RF power cache is ready: {coarse_power_path}")
                    update_progress(100, "Coarse wavelet decomposition", "Coarse RF power cache ready")
                    return True

                update_progress(25, "Coarse wavelet decomposition", "Preparing coarse real phase")
                print("Step 2/4: Preparing coarse real phase wavelets...")
                real_phase_fingerprint = _cache_fingerprint({"phase": 0, "base": coarse_phase_fingerprint})
                if _artifact_ready(real_phase_path, coarse_phase_shape, real_phase_fingerprint, kind="coarse_model_phase" if product == "model" else "coarse_rf_temporary_phase"):
                    print(f"Resume: found completed coarse real phase, reusing {real_phase_path}")
                    _write_recovery_step("coarse_phase_real_reused", path=real_phase_path)
                else:
                    _register_cancel_cleanup_path(real_phase_path)
                    if backend == "convolution":
                        waveletDecompositionConv(
                            videodata,
                            0,
                            sigmas,
                            wavelet_folder,
                            n_orientations=n_thetas,
                            phase_offsets=phase_offsets,
                            kernel_cache_path=coarse_kernel_cache_path,
                            output_format="zarr",
                            output_stem=os.path.splitext(os.path.basename(real_phase_path))[0],
                            cancel_event=_current_cancel_event(),
                        )
                    else:
                        waveletDecomposition(
                            videodata,
                            0,
                            sigmas,
                            wavelet_folder,
                            coarse_lib_path,
                            output_format="zarr",
                            output_stem=os.path.splitext(os.path.basename(real_phase_path))[0],
                            cancel_event=_current_cancel_event(),
                        )
                    _write_artifact_metadata(real_phase_path, "coarse_model_phase" if product == "model" else "coarse_rf_temporary_phase", coarse_phase_shape, real_phase_fingerprint)
                    _write_recovery_step("coarse_phase_real_complete", path=real_phase_path)

                update_progress(45, "Coarse wavelet decomposition", "Preparing coarse imaginary phase")
                print("Step 3/4: Preparing coarse imaginary phase wavelets...")
                imag_phase_fingerprint = _cache_fingerprint({"phase": 1, "base": coarse_phase_fingerprint})
                if _artifact_ready(imag_phase_path, coarse_phase_shape, imag_phase_fingerprint, kind="coarse_model_phase" if product == "model" else "coarse_rf_temporary_phase"):
                    print(f"Resume: found completed coarse imaginary phase, reusing {imag_phase_path}")
                    _write_recovery_step("coarse_phase_imaginary_reused", path=imag_phase_path)
                else:
                    _register_cancel_cleanup_path(imag_phase_path)
                    if backend == "convolution":
                        waveletDecompositionConv(
                            videodata,
                            1,
                            sigmas,
                            wavelet_folder,
                            n_orientations=n_thetas,
                            phase_offsets=phase_offsets,
                            kernel_cache_path=coarse_kernel_cache_path,
                            output_format="zarr",
                            output_stem=os.path.splitext(os.path.basename(imag_phase_path))[0],
                            cancel_event=_current_cancel_event(),
                        )
                    else:
                        waveletDecomposition(
                            videodata,
                            1,
                            sigmas,
                            wavelet_folder,
                            coarse_lib_path,
                            output_format="zarr",
                            output_stem=os.path.splitext(os.path.basename(imag_phase_path))[0],
                            cancel_event=_current_cancel_event(),
                        )
                    _write_artifact_metadata(imag_phase_path, "coarse_model_phase" if product == "model" else "coarse_rf_temporary_phase", coarse_phase_shape, imag_phase_fingerprint)
                    _write_recovery_step("coarse_phase_imaginary_complete", path=imag_phase_path)

                update_progress(68, "Coarse wavelet decomposition", "Building the requested cache product")
                print(f"Step 4/4: Building the {product.replace('_', ' ')} cache product...")
                if product == "coarse_rf":
                    _register_cancel_cleanup_path(coarse_power_path)
                    _build_coarse_power_zarr(real_phase_path, imag_phase_path, coarse_power_path)
                elif not _artifact_ready(model_imag_path, coarse_phase_shape, imag_phase_fingerprint, kind="coarse_model_phase"):
                    raise ValueError("Run Model requires both named coarse model phase caches.")
                _raise_if_cancelled()
                if product == "coarse_rf":
                    if not _artifact_matches(coarse_power_path, coarse_power_shape):
                        raise ValueError(f"Coarse RF power cache was written with an unexpected shape: {coarse_power_path}")
                    _write_artifact_metadata(coarse_power_path, "coarse_rf_power", coarse_power_shape, coarse_power_fingerprint)
                    _write_recovery_step("coarse_rf_power_complete", path=coarse_power_path)
                    for intermediate_path in (real_phase_path, imag_phase_path):
                        try:
                            if os.path.isdir(intermediate_path):
                                shutil.rmtree(intermediate_path)
                                metadata_path = _artifact_meta_path(intermediate_path)
                                if os.path.exists(metadata_path):
                                    os.remove(metadata_path)
                                print(f"Removed RF-only temporary phase cache: {intermediate_path}")
                        except Exception as exc:
                            print(f"Could not remove temporary phase cache {intermediate_path}: {exc}")
                else:
                    _write_recovery_step("coarse_model_cache_complete", real=model_real_path, imag=model_imag_path)

            print(
                f"Coarse RF grid derived from movie metadata: {coarse_nx} x {coarse_ny} "
                f"({_selected_downsample_percent():.0f}% of "
                f"{_movie_metadata(movpath)['width']} x {_movie_metadata(movpath)['height']})"
            )
            print(f"Coarse {product.replace('_', ' ')} wavelet files are ready: {wavelet_folder}")
            update_progress(100, "Coarse wavelet decomposition", f"{product.replace('_', ' ').title()} cache ready")
            return True

        if scale == "full":
            update_progress(5, "Full wavelet decomposition", "Preparing full-model outputs")

        full_output_target = _wavelet_folder("full")
        if is_zarr_wavelet:
            full_output = full_output_target
            os.makedirs(full_output_target, exist_ok=True)
            print(f"Writing full-model wavelets directly to Zarr in: {full_output}")
        else:
            full_output = full_output_target
            os.makedirs(full_output, exist_ok=True)

        sigmas_full = parse_literal(
            param_entries["Sigmas Full Model"].get(),
            "Sigmas Full Model",
        )
        update_progress(15, "Full wavelet decomposition", "Preparing full-resolution stimulus movie")
        print("Step 1/3: Preparing full-resolution stimulus movie...")
        if full_downsample_ready:
            _write_recovery_step("full_downsampling_reused", path=full_downsample_path)
        else:
            raise FileNotFoundError(
                "The stimulus downsample stage is incomplete. Run 'Prepare Stimulus Cache' "
                "in Stage 2 before wavelet decomposition."
            )
        if not _artifact_matches(full_downsample_path, full_downsample_shape):
            raise FileNotFoundError(
                "Full-resolution downsampled stimulus was not created. "
                f"Check that Movie Path points to a readable video: {movpath}"
            )
        _raise_if_cancelled()
        videodata = _load_downsampled_movie(
            full_downsample_path, streaming=(backend == "convolution")
        )
        full_model_shape = (
            videodata.shape[0],
            full_nx,
            full_ny,
            n_thetas,
            len(sigmas_full),
            max(1, len(frequencies)),
        )
        zarr_chunks = (
            min(expected_frames, 128),
            min(full_nx, 16),
            min(full_ny, 16),
            n_thetas,
            len(sigmas_full),
            max(1, len(frequencies)),
        )
        full_phase_fingerprint = _cache_fingerprint(
            {
                "artifact": "full_wavelet_phase",
                "backend": backend,
                "shape": full_model_shape,
                "downsample": full_downsample_fingerprint,
                "output_format": output_format,
            }
        )
        for phase in (0, 1):
            suffix = "_r" if phase == 0 else "_i"
            target_ext = ".zarr" if is_zarr_wavelet else ".npy"
            target = os.path.join(full_output, f"dwt_videodata2{suffix}{target_ext}")
            phase_fingerprint = _cache_fingerprint({"phase": phase, "base": full_phase_fingerprint})
            if _artifact_ready(target, full_model_shape, phase_fingerprint, kind="full_wavelet_phase"):
                print(f"Resume: found completed full-model wavelets, reusing {target}")
                _write_recovery_step(f"full_model_phase_{phase}_reused", path=target, shape=full_model_shape)
                continue
            update_progress(45 + phase * 25, "Full wavelet decomposition", f"Writing full-model phase {phase}")
            _register_cancel_cleanup_path(target)
            if backend == "convolution":
                waveletDecompositionFullConv(
                    videodata,
                    phase,
                    sigmas_full,
                    frequencies,
                    full_output,
                    n_orientations=n_thetas,
                    phase_offsets=phase_offsets,
                    output_format=output_format,
                    zarr_chunks=zarr_chunks if is_zarr_wavelet else None,
                    kernel_cache_path=fine_kernel_cache_path,
                    cancel_event=_current_cancel_event(),
                )
            else:
                waveletDecompositionFull(
                    videodata,
                    phase,
                    sigmas_full,
                    frequencies,
                    full_output,
                    fine_lib_path,
                    library_sigmas=fine_library_sigmas,
                    output_format=output_format,
                    zarr_chunks=zarr_chunks if is_zarr_wavelet else None,
                    cancel_event=_current_cancel_event(),
                )
            if not _artifact_matches(target, full_model_shape):
                raise ValueError(f"Full-model wavelets were written with an unexpected shape: {target}")
            _write_artifact_metadata(target, "full_wavelet_phase", full_model_shape, phase_fingerprint)
            _write_recovery_step(f"full_model_phase_{phase}_complete", path=target)

        if is_zarr_wavelet:
            _write_recovery_step("full_model_zarr_complete", path=full_output_target)
            print(
                f"Full-model wavelet files are ready. Zarr output: {full_output_target}"
            )
        else:
            print(f"Full-model wavelet files are ready. NPY output: {full_output}")
        update_progress(100, "Full wavelet decomposition", "Full-model wavelets ready")
        return True
    
    def embed_interactive_figure(fig, parent_container, title=None):
        """Embed a matplotlib figure with navigation toolbar in ``parent_container``."""
        _ensure_plot_imports()
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
        caption = getattr(fig, "_waven_caption", "")
        caption_label = ctk.CTkLabel(
            section,
            text=caption,
            text_color="#6B7280",
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        caption_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(2, 8))
        embedded_canvases.append(canvas)
        record = {
            "canvas": canvas,
            "figure": fig,
            "title": graph_title,
            "tab": _tab_name_for_parent(parent_container),
            "section": section,
            "caption_widget": caption_label,
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

    def refresh_figure_caption(fig):
        """Function for refresh figure caption.

        Args:
            fig: Input value for this operation.
        """
        caption = getattr(fig, "_waven_caption", "")
        for record in figure_export_records:
            if record.get("figure") is fig and record.get("caption_widget") is not None:
                record["caption_widget"].configure(text=caption)
                break

    def clear_plot_tab(parent):
        """Function for clear plot tab.

        Args:
            parent: Input value for this operation.
        """
        tab_name = _tab_name_for_parent(parent)
        for widget in parent.winfo_children():
            widget.destroy()
        figure_export_records[:] = [
            record for record in figure_export_records
            if record.get("tab") != tab_name
        ]

    def switch_to_individual_tab(flash=True):
        """Function for switch to individual tab.

        Args:
            flash: Input value for this operation.
        """
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
        """Function for embed captured figures.

        Args:
            figures: Input value for this operation.
            parent: Input value for this operation.
            title_prefix: Input value for this operation.
        """
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
        """Function for capture new figures.

        Args:
            callback: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        _ensure_plot_imports()
        before = set(plt.get_fignums())
        result = callback()
        after = set(plt.get_fignums())
        figures = [plt.figure(num) for num in sorted(after - before)]
        return result, figures

    def gui_trailing_sep(path):
        """Function for gui trailing sep.

        Args:
            path: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        path = str(path)
        if path.endswith(("/", "\\")):
            return path
        return path + os.sep

    def _neural_alignment_context():
        """Return parsed inputs needed to load or create aligned neural caches."""
        data_dirs = _parse_data_dir(param_entries["Dir"].get())
        data_dirs = (resolve_folder_reference(data_dirs[0], "raw_data"),) + tuple(data_dirs[1:])
        exp_info = parse_literal(param_entries["Experiment Info"].get(), "Experiment Info")
        block_end = int(param_entries["Block End"].get())
        nb_frames = _movie_metadata()["frames"]
        pathdata = Path(data_dirs[0]) / exp_info[0] / exp_info[1] / str(exp_info[2])
        pathsuite2p = pathdata / "suite2p"
        neural_cache_dir = Path(_folder_from_entry(param_entries, "Spks Path", _project_layout().neural_cache_dir))
        if workflow == WORKFLOW_2P:
            n_planes = int(param_entries["Number of Planes"].get())
            resolution = float(param_entries["Resolution"].get())
            sampling_rate = None
        else:
            n_planes = None
            resolution = None
            sampling_rate = float(param_entries["Sampling Rate (samples / sec)"].get())
        return {
            "data_dirs": data_dirs,
            "experiment_info": exp_info,
            "pathdata": pathdata,
            "suite2p_dir": pathsuite2p,
            "neural_cache_dir": neural_cache_dir,
            "block_end": block_end,
            "n_planes": n_planes,
            "nb_frames": nb_frames,
            "resolution": resolution,
            "sampling_rate": sampling_rate,
        }

    def create_neural_cache():
        """Create or validate the aligned ``pos`` and ``spikes`` cache pair."""
        if "downsample:coarse" not in completed_actions:
            raise RuntimeError(
                "Prepare Stimulus & Metadata first. Neural alignment uses the selected movie's "
                "metadata-derived frame count and duration."
            )
        context = _neural_alignment_context()
        spks_text = param_entries["Spks Path"].get().strip()
        if _selected_neural_source() == "spks_path":
            if spks_text.lower() in ("", "none", "null"):
                raise ValueError("Select a Spks Path, or switch neural source to Data Dir.")
            spks_folder = Path(_folder_from_entry(param_entries, "Spks Path", _project_layout().neural_cache_dir))
            update_progress(10, "Neural cache", "Validating existing aligned cache")
            spks, neuron_pos, loaded_spks_path, loaded_pos_path = load_neural_cache_pair(
                spks_folder,
                mmap_mode=None,
            )
            print(f"Validated aligned spikes cache: {loaded_spks_path} {tuple(spks.shape)}")
            print(f"Validated neuron position cache: {loaded_pos_path} {tuple(np.asarray(neuron_pos).shape)}")
            _set_entry_value(param_entries["Spks Path"], spks_folder)
            update_progress(100, "Neural cache", "Existing cache ready")
            return True

        if spks_text.lower() not in ("", "none", "null"):
            print("Neural source is Data Dir; Spks Path will be ignored unless you select Spks Path mode.")

        cache_pair = find_neural_cache_pair(context["neural_cache_dir"])
        if cache_pair is not None:
            update_progress(20, "Neural cache", "Loading existing aligned cache")
            spks, neuron_pos, loaded_spks_path, loaded_pos_path = load_neural_cache_pair(context["neural_cache_dir"])
            print(f"Resume: found aligned spikes cache: {loaded_spks_path} {tuple(spks.shape)}")
            print(f"Resume: found neuron position cache: {loaded_pos_path} {tuple(np.asarray(neuron_pos).shape)}")
            _set_entry_value(param_entries["Spks Path"], context["neural_cache_dir"])
            update_progress(100, "Neural cache", "Existing cache ready")
            return True

        update_progress(10, "Neural cache", "Aligning neural data")
        from .. import time_alignment as ta

        aligned = ta.load_aligned_spikes(
            workflow,
            experiment_info=context["experiment_info"],
            data_dir=Path(context["data_dirs"][0]),
            data_dir_strings=context["data_dirs"],
            suite2p_dir=context["suite2p_dir"],
            block_end=context["block_end"],
            n_planes=context["n_planes"],
            nb_frames=context["nb_frames"],
            resolution=context["resolution"],
            sampling_rate=context["sampling_rate"],
            stimulus_duration=_movie_metadata()["duration"],
            threshold=1.25,
            method="frame2ttl",
            save_dir=context["neural_cache_dir"],
            output_format=_selected_neural_cache_format(),
        )
        cache_pair = find_neural_cache_pair(context["neural_cache_dir"])
        if cache_pair is None:
            raise FileNotFoundError(
                f"Alignment finished but no spikes/pos cache pair was found in {context['neural_cache_dir']}."
            )
        saved_spks_path, saved_pos_path = cache_pair
        print(f"Created aligned spikes cache: {saved_spks_path} {tuple(aligned.spikes.shape)}")
        print(f"Created neuron position cache: {saved_pos_path} {tuple(np.asarray(aligned.neuron_pos).shape)}")
        _set_entry_value(param_entries["Spks Path"], context["neural_cache_dir"])
        update_progress(100, "Neural cache", "Cache created")
        return True

    def plot_data():
        """Function for plot data.

        Returns:
            Result produced by the operation.
        """
        _ensure_rf_imports("coarse RF analysis")
        rf_extra = {
            "selected_neuron": _field_value(param_entries, "Neuron ID", ""),
        }
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
            data_dirs = (resolve_folder_reference(data_dirs[0], "raw_data"),) + tuple(data_dirs[1:])
            exp_info = parse_literal(param_entries["Experiment Info"].get(), "Experiment Info")
            sigmas = np.array(parse_literal(gabor_entries["Sigmas"].get(), "Sigmas"))
            frequencies = np.array(parse_literal(gabor_entries["Frequencies"].get(), "Frequencies"))
            nf = len(frequencies)
            visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
            analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")
            block_end = int(param_entries["Block End"].get())
            movie_metadata = _movie_metadata()
            nx, ny = movie_metadata["width"], movie_metadata["height"]
            coarse_nx, coarse_ny = _stimulus_grid_dimensions("coarse")
            n_orientations = int(gabor_entries["N_thetas"].get())
            ns = len(sigmas)
            spks_path = param_entries["Spks Path"].get()
            neural_cache_dir = Path(_folder_from_entry(param_entries, "Spks Path", _project_layout().neural_cache_dir))
            nb_frames = movie_metadata["frames"]
            movpath = _find_movie_path()
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

        unit_ids = None
        if _selected_neural_source() == "data_dir":
            cache_pair = find_neural_cache_pair(neural_cache_dir)
            if cache_pair is not None:
                update_progress(10, "Coarse receptive-field analysis", "Loading aligned neural cache")
                spks, neuron_pos, saved_spks_path, saved_pos_path = load_neural_cache_pair(neural_cache_dir)
                unit_ids = load_unit_ids(neural_cache_dir, np.asarray(neuron_pos).shape[0])
                print(f"Loaded aligned spikes from: {saved_spks_path}")
                print(f"Loaded neuron positions from: {saved_pos_path}")
            else:
                update_progress(10, "Coarse receptive-field analysis", "Aligning neural data")
                from .. import time_alignment as ta

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
                        stimulus_duration=_movie_metadata()["duration"],
                        threshold=1.25,
                        method='frame2ttl',
                        save_dir=neural_cache_dir,
                        output_format=_selected_neural_cache_format(),
                    )
                except NotImplementedError as exc:
                    print(exc)
                    return False
                spks = aligned.spikes
                neuron_pos = aligned.neuron_pos
                unit_ids = aligned.unit_ids
                cache_pair = find_neural_cache_pair(neural_cache_dir)
                saved_spks_path = cache_pair[0] if cache_pair else None
                if saved_spks_path is not None:
                    print(f"Saved aligned neural cache as {_selected_neural_cache_format().upper()}: {saved_spks_path}")
            if saved_spks_path is not None and saved_spks_path.exists():
                _set_entry_value(param_entries["Spks Path"], neural_cache_dir)
            if workflow == WORKFLOW_2P:
                neuron_pos = np.asarray(neuron_pos)
                neuron_pos[:, 1] = abs(neuron_pos[:, 1] - np.max(neuron_pos[:, 1]))
        else:
            try:
                update_progress(10, "Coarse receptive-field analysis", "Loading pre-aligned spikes")
                spks_folder = Path(_folder_from_entry(param_entries, "Spks Path", _project_layout().neural_cache_dir))
                spks, neuron_pos, loaded_spks_path, loaded_pos_path = load_neural_cache_pair(
                    spks_folder,
                    mmap_mode=None,
                )
                unit_ids = load_unit_ids(spks_folder, np.asarray(neuron_pos).shape[0])
                print(f"Loaded aligned spikes from: {loaded_spks_path}")
                print(f"Loaded neuron positions from: {loaded_pos_path}")
            except Exception as e:
                print(f"File not found: {e}")
                return False

        neuron_pos = np.asarray(neuron_pos)
        print("Loading neural data and coarse wavelets...")
        update_progress(25, "Coarse receptive-field analysis", "Loading coarse wavelets")
        _write_recovery_step("loading_neural_data")
        respcorr = repetability_trial3(spks, neuron_pos, plotting=False)
        skewness = np.array(compute_skewness_neurons(spks, plotting=False))
        filter_mask = np.logical_and(respcorr >= 0.2, skewness <= 20)

        point_alphas = np.where(filter_mask, 1.0, 0.05)

        parent_dir = current_wavelet_dir[0] or _wavelet_folder("coarse")
        try:
            # Coarse RF owns only its magnitude/power product.  It stays
            # disk-backed and correlation reads bounded feature blocks.
            from ..storage.array_store import load_array
            power_path = os.path.join(parent_dir, "coarse_rf_power.zarr")
            w_c_downsampled = load_array(power_path, mmap_mode="r")
        except Exception as e:
            print(f"Coarse RF power cache loading failed: {e}")
            return False
            
        n_frames = min(nb_frames, w_c_downsampled.shape[0], spks.shape[1])

        if w_c_downsampled.ndim == 6:
            rf_nf = w_c_downsampled.shape[5]
            rf_frequencies = frequencies[:rf_nf]
        else:
            rf_nf = 1
            rf_frequencies = frequencies[:1]

        # Pass the disk-backed wavelet tensor directly.  PearsonCorrelationPinkNoise
        # streams spatial-feature blocks; flattening this Zarr selection here would
        # allocate the entire coarse cache (several GiB for long recordings).
        rfs_gabor = PearsonCorrelationPinkNoise(w_c_downsampled,
                                                np.mean(spks[:, :n_frames], axis=0),
                                                neuron_pos, coarse_nx, coarse_ny, ns, rf_nf, analysis_coverage, screen_ratio, sigmas_deg, rf_frequencies,
                                                n_orientations=n_orientations,
                                                plotting=False)
        update_progress(75, "Coarse receptive-field analysis", "Computing firing-rate OSI/gOSI")
        # ``spks`` is the aligned neural cache.  For ephys it is the
        # frame-bin firing rate produced from spike counts / bin duration in
        # sta.py-style alignment; it is never replaced by RF correlations.
        # The wavelet energy only weights frames into orientation bins at each
        # neuron's already-established preferred RF features.
        orientation_selectivity = firing_rate_orientation_tuning(
            spks[:, :n_frames, :],
            w_c_downsampled,
            rfs_gabor,
        )
        _write_recovery_step("coarse_rf_complete", wavelet_dir=parent_dir)
        analysis_state.clear()
        analysis_state.update(
            spks=spks,
            neuron_pos=neuron_pos,
            unit_ids=unit_ids,
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
            orientation_selectivity=orientation_selectivity,
            nb_frames=nb_frames,
            wavelet_dir=parent_dir,
        )

        def render_gui_plots():
            """Function for render gui plots."""
            update_progress(90, "Coarse receptive-field analysis", "Rendering plots")
            clear_plot_tab(frame_plot_all)
            clear_plot_tab(frame_plot_individual)
            embedded_canvases.clear()
            plt.close('all')

            pos_dim = neuron_pos.shape[1] if getattr(neuron_pos, "ndim", 0) == 2 else 0
            has_z = pos_dim >= 3
            fig1 = plt.figure(figsize=(6, 5), constrained_layout=True)
            ax1 = fig1.add_subplot(111, projection='3d') if has_z else fig1.add_subplot(111)
            if has_z:
                ax1.scatter(
                    neuron_pos[:, 0], neuron_pos[:, 1], neuron_pos[:, 2],
                    c='k', alpha=0.3, label="Neurons", picker=True, rasterized=True,
                )
                ax1.set_zlabel("Z (um)")
            else:
                ax1.scatter(neuron_pos[:, 0], neuron_pos[:, 1], c='k', alpha=0.3, label="Neurons", picker=True, rasterized=True)
            ax1.set_title("Neuron Positions (µm)")
            ax1.set_xlabel("X (µm)")
            ax1.set_ylabel("Y (µm)")

            ax1.set_title("Neuron Positions")
            ax1.set_xlabel("X (um)")
            ax1.set_ylabel("Y (um)")
            if has_z:
                ax1.set_zlabel("Z (um)")

            subplot_kwargs = {"projection": "3d"} if has_z else {}
            fig10, axes10 = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True, subplot_kw=subplot_kwargs)
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
                title = ["Azimuth (deg)", "Elevation (deg)", "Orientation (deg)", "Size (deg)"][idx]
                if has_z:
                    scatter = ax10[idx].scatter(
                        neuron_pos[:, 0], neuron_pos[:, 1], neuron_pos[:, 2], s=5, c=values,
                        cmap=cmap, alpha=point_alphas, rasterized=True, picker=True,
                    )
                    ax10[idx].set_zlabel("Z (um)")
                else:
                    scatter = ax10[idx].scatter(
                        neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=values,
                        cmap=cmap, alpha=point_alphas, rasterized=True, picker=True,
                    )
                fig10.colorbar(scatter, ax=ax10[idx], fraction=0.046)
                ax10[idx].set_title(title)
                ax10[idx].set_xlabel("X (µm)")
                ax10[idx].set_ylabel("Y (µm)")

            for axis in ax10:
                axis.set_xlabel("X (um)")
                axis.set_ylabel("Y (um)")
                if has_z:
                    axis.set_zlabel("Z (um)")

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

            def correlation_tuning_ci(feature_matrix, trial_responses):
                """Return 95% CIs from trial-wise feature/response correlations."""
                features = np.asarray(feature_matrix, dtype=float)
                responses = np.asarray(trial_responses, dtype=float)
                if features.ndim != 2 or responses.ndim != 2 or features.shape[0] != responses.shape[1]:
                    return None, 0
                feature_centered = features - np.nanmean(features, axis=0, keepdims=True)
                feature_norm = np.sqrt(np.nansum(feature_centered ** 2, axis=0))
                trial_curves = np.full((responses.shape[0], features.shape[1]), np.nan, dtype=float)
                for trial_index, response in enumerate(responses):
                    valid = np.isfinite(response)
                    if valid.sum() < 2:
                        continue
                    centered_response = response[valid] - np.mean(response[valid])
                    response_norm = np.sqrt(np.sum(centered_response ** 2))
                    if response_norm == 0:
                        continue
                    centered_features = feature_centered[valid]
                    norms = np.sqrt(np.sum(centered_features ** 2, axis=0))
                    good = norms > 0
                    trial_curves[trial_index, good] = (
                        centered_response @ centered_features[:, good]
                    ) / (response_norm * norms[good])
                counts = np.sum(np.isfinite(trial_curves), axis=0)
                if not np.any(counts >= 2):
                    return None, int(np.max(counts))
                ci = np.full(features.shape[1], np.nan, dtype=float)
                valid_columns = counts >= 2
                ci[valid_columns] = (
                    1.96
                    * np.nanstd(trial_curves[:, valid_columns], axis=0, ddof=1)
                    / np.sqrt(counts[valid_columns])
                )
                return ci, int(np.nanmin(counts[valid_columns]))

            def draw_individual_neuron(neuron_id, switch_tab=True):
                """Function for draw individual neuron.

                Args:
                    neuron_id: Input value for this operation.
                """
                try:
                    entry_neuron.delete(0, tk.END)
                    entry_neuron.insert(0, str(neuron_id))

                    trial_spikes = spks[:, :, neuron_id]
                    spike_train = np.mean(trial_spikes, axis=0)
                    ax2.clear()
                    ax2.plot(
                        np.arange(spike_train.shape[0]),
                        spike_train,
                        c="k",
                        label=f"Neuron {neuron_id} trial-averaged activity",
                    )
                    ax2.set_title("Trial-averaged Spike Train")
                    ax2.set_xlabel("Frame index")
                    ax2.set_ylabel("Activity (a.u.)")
                    ax2.legend()
                    _set_figure_export_payload(
                        fig2,
                        {
                            "source": "Inspect Single Neuron",
                            "neuron_id": neuron_id,
                            "spike_train": spike_train,
                            "trial_spikes": trial_spikes,
                        },
                    )
                    refresh_figure_caption(fig2)
                    canvas2.draw()

                    rf2d, x_tuning, y_tuning, ori_tun, s_tuning, f_tuning = PlotTuningCurve(rfs_gabor, neuron_id, analysis_coverage, sigmas_deg, screen_ratio, frequencies, show=False)
                    best_indices = np.asarray(rfs_gabor[1], dtype=int)[:, neuron_id]
                    best_x, best_y, best_orientation, best_sigma = best_indices[:4]
                    if w_c_downsampled.ndim == 6:
                        best_frequency = best_indices[4]
                        orientation_features = w_c_downsampled[:n_frames, best_x, best_y, :, best_sigma, best_frequency]
                        size_features = w_c_downsampled[:n_frames, best_x, best_y, best_orientation, :, best_frequency]
                    else:
                        orientation_features = w_c_downsampled[:n_frames, best_x, best_y, :, best_sigma]
                        size_features = w_c_downsampled[:n_frames, best_x, best_y, best_orientation, :]
                    ori_ci, ori_ci_trials = correlation_tuning_ci(orientation_features, trial_spikes[:, :n_frames])
                    size_ci, size_ci_trials = correlation_tuning_ci(size_features, trial_spikes[:, :n_frames])
                    rate_ori_tun = np.asarray(
                        analysis_state["orientation_selectivity"]["orientation_tuning"][neuron_id],
                        dtype=float,
                    )
                    neuron_osi = float(analysis_state["orientation_selectivity"]["osi"][neuron_id])
                    neuron_gosi = float(analysis_state["orientation_selectivity"]["gosi"][neuron_id])
                    for extra_ax in [axis for axis in list(fig3.axes) if axis not in ax3]:
                        extra_ax.remove()
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
                    ax3[1].set_xlabel("Elevation (deg)")
                    ax3[1].set_ylabel("Correlation (a.u.)")
                    ax3[2].plot(y_tuning, c='k')
                    ax3[2].set_title('Azimuth (deg)')
                    ax3[2].set_xticks([0, rf2d.shape[1]], [xM, xm])
                    ax3[2].set_xlabel("Azimuth (deg)")
                    ax3[2].set_ylabel("Correlation (a.u.)")
                    ax3[3].errorbar(
                        np.arange(len(ori_tun)), ori_tun, yerr=None if ori_ci is None else np.append(ori_ci, ori_ci[0]),
                        fmt='o-', c='#2563EB', ecolor='#60A5FA', capsize=3,
                        label=f'Correlation (95% CI; n={ori_ci_trials})' if ori_ci is not None else 'Correlation',
                    )
                    ax3[3].set_title(f'Orientation (firing-rate OSI {neuron_osi:.3f}, gOSI {neuron_gosi:.3f})')
                    n_ori = rfs_gabor[0].shape[3]
                    ax3[3].set_xticks(
                        [0, max(1, n_ori // 2), max(2, n_ori - 1)],
                        [0, 90, 180],
                    )
                    ax3[3].set_xlabel("Orientation (deg)")
                    ax3[3].set_ylabel("Correlation (a.u.)")
                    ax3[3].legend(fontsize=8)
                    ax3[4].errorbar(
                        np.arange(len(s_tuning)), s_tuning, yerr=size_ci,
                        fmt='o-', c='#7C3AED', ecolor='#C4B5FD', capsize=3,
                        label=f'Correlation (95% CI; n={size_ci_trials})' if size_ci is not None else 'Correlation',
                    )
                    ax3[4].set_title('Size (deg)')
                    ax3[4].set_xticks([0, len(sigmas) - 1], [sigmas_deg[0], sigmas_deg[-1]])
                    ax3[4].set_xlabel("Size (deg)")
                    ax3[4].set_ylabel("Correlation (a.u.)")
                    ax3[4].legend(fontsize=8)
                    has_frequency_axis = w_c_downsampled.ndim == 6 and rf_nf > 1 and len(f_tuning) > 1
                    if has_frequency_axis:
                        ax3[5].set_visible(True)
                        ax3[5].plot(f_tuning, 'o-', c='k')
                        ax3[5].set_title('Spatial Frequency')
                        ax3[5].set_xticks(range(len(rf_frequencies)), [round(f, 3) for f in rf_frequencies])
                        ax3[5].set_xlabel("Spatial frequency (cycles/deg)")
                        ax3[5].set_ylabel("Correlation (a.u.)")
                    else:
                        ax3[5].set_visible(False)
                    _set_figure_export_payload(
                        fig3,
                        {
                            "source": "Inspect Single Neuron",
                            "neuron_id": neuron_id,
                            "rf2d": rf2d,
                            "x_tuning": x_tuning,
                            "y_tuning": y_tuning,
                            "orientation_correlation_tuning": ori_tun,
                            "orientation_correlation_ci_95": ori_ci,
                            "orientation_firing_rate_tuning": rate_ori_tun,
                            "osi": neuron_osi,
                            "gosi": neuron_gosi,
                            "osi_source": "firing_rate",
                            "size_tuning": s_tuning,
                            "size_correlation_ci_95": size_ci,
                            "frequency_tuning": f_tuning if has_frequency_axis else None,
                            "frequency_tuning_available": has_frequency_axis,
                            "best_params": np.asarray(rfs_gabor[1])[:, neuron_id],
                            "retinotopy": np.asarray(rfs_gabor[2])[:, neuron_id],
                        },
                    )
                    canvas3.draw()
                    if switch_tab:
                        switch_to_individual_tab(flash=True)
                except Exception as e:
                    print(f"Error drawing selected neuron: {e}")

            def onpick(event):
                """Function for onpick.

                Args:
                    event: Input value for this operation.
                """
                try:
                    draw_individual_neuron(int(event.ind[0]))
                except Exception as e:
                    print(f"Error drawing pick event: {e}")

            fig1.canvas.mpl_connect('pick_event', onpick)
            fig10.canvas.mpl_connect('pick_event', onpick)
            individual_neuron_renderer["draw"] = draw_individual_neuron

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
            fig_osi_population = _plot_selectivity_population(
                orientation_selectivity,
                filter_mask,
                neuron_pos=neuron_pos,
                show_shanks=workflow == WORKFLOW_EPHYS,
            )
            fig_osi_units = None
            if workflow == WORKFLOW_EPHYS:
                fig_osi_units = _plot_selectivity_by_unit(
                    orientation_selectivity,
                    neuron_pos,
                    filter_mask,
                    unit_ids=unit_ids,
                )
            embed_interactive_figure(fig1, frame_plot_all, title="Neuron Layout")
            embed_interactive_figure(fig10, frame_plot_all, title="Population Retinotopy Maps")
            embed_interactive_figure(
                fig_osi_population,
                frame_plot_all,
                title="OSI and gOSI by Neuron/Shank" if workflow == WORKFLOW_EPHYS else "OSI and gOSI All Cells Distribution",
            )
            if fig_osi_units is not None:
                embed_interactive_figure(fig_osi_units, frame_plot_all, title="OSI and gOSI by Unit")
            canvas2 = embed_interactive_figure(fig2, frame_plot_individual, title="Spike Train")
            canvas3 = embed_interactive_figure(fig3, frame_plot_individual, title="Selected Neuron Tuning")

            def click_RF():
                """Function for click RF."""
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
            figure_records = [
                ("all", "Neuron Layout", fig1),
                ("all", "Population Retinotopy Maps", fig10),
                (
                    "all",
                    "OSI and gOSI by Neuron/Shank" if workflow == WORKFLOW_EPHYS else "OSI and gOSI All Cells Distribution",
                    fig_osi_population,
                ),
                ("individual", "Spike Train", fig2),
                ("individual", "Selected Neuron Tuning", fig3),
            ]
            if fig_osi_units is not None:
                figure_records.insert(3, ("all", "OSI and gOSI by Unit", fig_osi_units))
            _put_cached_entry(
                "coarse_rf",
                {
                    "analysis_state": _state_for_plot_cache(analysis_state),
                    "figures": _figure_records(figure_records),
                },
                extra=rf_extra,
            )
            print("Plots rendered successfully.")
            update_progress(100, "Coarse receptive-field analysis", "Plots ready")

        root.after(0, render_gui_plots)

    def _selected_neuron_id():
        """Function for selected neuron id.

        Returns:
            Result produced by the operation.
        """
        neuron_id = int(param_entries["Neuron ID"].get())
        if "spks" in analysis_state:
            neuron_count = int(analysis_state["spks"].shape[2])
            if neuron_count <= 0:
                raise ValueError("The loaded neural cache contains no neurons.")
            if not (0 <= neuron_id < neuron_count):
                corrected = min(max(neuron_id, 0), neuron_count - 1)
                print(
                    f"Neuron ID {neuron_id} is outside the loaded range 0-{neuron_count - 1}; "
                    f"using {corrected}."
                )
                # Model actions run in a worker.  Schedule the widget update
                # on Tk's main thread instead of touching it here.
                def update_neuron_entry():
                    param_entries["Neuron ID"].delete(0, tk.END)
                    param_entries["Neuron ID"].insert(0, str(corrected))
                root.after(0, update_neuron_entry)
                neuron_id = corrected
        return neuron_id

    def _parse_bool_entry(value, label):
        """Function for parse bool entry.

        Args:
            value: Input value for this operation.
            label: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(f"{label} must be True or False.")

    def _parse_trial_indices_entry(value, n_trials, label, train_indices=None):
        """Function for parse trial indices entry.

        Args:
            value: Input value for this operation.
            n_trials: Input value for this operation.
            label: Input value for this operation.
            train_indices: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        text = str(value).strip()
        if text.lower() in {"", "auto"}:
            if train_indices is None:
                defaults = [0, 2]
                indices = [idx for idx in defaults if idx < n_trials]
                return indices or [0]
            train_set = set(int(idx) for idx in train_indices)
            indices = [idx for idx in range(n_trials) if idx not in train_set]
            if not indices:
                raise ValueError(
                    "Automatic test trial selection found no held-out trials. "
                    "Use fewer training trials or enter explicit Test Trial Indices."
                )
            return indices
        parsed = parse_literal(text, label)
        if isinstance(parsed, int):
            parsed = [parsed]
        if not isinstance(parsed, (list, tuple)):
            raise ValueError(f"{label} must be a list of trial indices or 'auto'.")
        indices = [int(idx) for idx in parsed]
        if not indices:
            raise ValueError(f"{label} cannot be empty.")
        invalid = [idx for idx in indices if idx < 0 or idx >= n_trials]
        if invalid:
            raise ValueError(
                f"{label} contains out-of-range trial index/indices {invalid}; "
                f"available trials are 0 through {n_trials - 1}."
            )
        return indices

    def _model_split_settings(state):
        """Function for model split settings.

        Args:
            state: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        n_trials = int(state["spks"].shape[0])
        if n_trials < 2:
            raise ValueError("Model cross-validation requires at least two trials.")
        train_idx = _parse_trial_indices_entry(
            param_entries["Train Trial Indices"].get(),
            n_trials,
            "Train Trial Indices",
        )
        test_idx = _parse_trial_indices_entry(
            param_entries["Test Trial Indices"].get(),
            n_trials,
            "Test Trial Indices",
            train_indices=train_idx,
        )
        overlap = sorted(set(train_idx) & set(test_idx))
        if overlap:
            raise ValueError(f"Train and test trial indices overlap: {overlap}")
        return {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "lastmin": _parse_bool_entry(
                param_entries["Use Last Minute Holdout"].get(),
                "Use Last Minute Holdout",
            ),
        }

    def _require_rf_state():
        """Function for require rf state.

        Returns:
            Result produced by the operation.
        """
        if not analysis_state:
            raise RuntimeError("Run Coarse RF Analysis before model plotting.")
        return analysis_state

    def _append_model_figures(figures, title_prefix):
        """Function for append model figures.

        Args:
            figures: Input value for this operation.
            title_prefix: Input value for this operation.
        """
        def render():
            """Function for render."""
            switch_to_individual_tab(flash=True)
            embed_captured_figures(figures, frame_plot_individual, title_prefix)
        root.after(0, render)

    def _model_summary_figure(model_name, payload):
        """Create a backend-neutral model summary for safe main-thread embedding."""
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7.2, 4.2), constrained_layout=True)
        axis = figure.add_subplot(111)
        metrics = np.asarray(payload.get("metrics", []), dtype=float)
        if metrics.size:
            values = metrics.reshape(metrics.shape[0], -1)[0]
            labels = ["Metric 1", "Metric 2", "Metric 3"][:values.size]
            axis.bar(np.arange(values.size), values, color=["#2563EB", "#059669", "#7C3AED"][:values.size])
            axis.set_xticks(np.arange(values.size), labels)
            axis.set_ylabel("Model metric")
            axis.axhline(0, color="#64748B", linewidth=0.8)
        else:
            axis.text(0.5, 0.5, "Model completed; no scalar metrics were returned.", ha="center", va="center")
            axis.set_axis_off()
        axis.set_title(f"{model_name} summary — neuron {payload.get('neuron_id')}")
        figure._waven_caption = "Numerical fitting ran in a worker; this summary is rendered safely in the GUI thread."
        return figure

    def plot_run_model_outputs():
        """Function for plot run model outputs.

        Returns:
            Result produced by the operation.
        """
        _ensure_model_imports("run_Model plot capture")
        state = _require_rf_state()
        neuron_id = _selected_neuron_id()
        split_settings = _model_split_settings(state)
        cache_extra = {
            "model": "run_Model",
            "neuron": neuron_id,
            **split_settings,
        }
        cached = _get_cached_entry("run_model", neuron_id=neuron_id, extra=cache_extra)
        if cached:
            _render_figure_records(
                cached.get("figures"),
                message=f"Loaded run_Model plots for neuron {neuron_id} from cache.",
                clear=False,
            )
            return
        wavelet_dir = state["wavelet_dir"]
        from ..storage.array_store import load_array
        w_r = load_array(os.path.join(wavelet_dir, "coarse_model_real.zarr"), mmap_mode="r")
        w_i = load_array(os.path.join(wavelet_dir, "coarse_model_imag.zarr"), mmap_mode="r")
        expected_coarse_features = (
            int(state["coarse_nx"]),
            int(state["coarse_ny"]),
            int(state["n_orientations"]),
            len(state["sigmas"]),
        )
        if w_r.ndim != 5 or w_i.ndim != 5 or w_r.shape != w_i.shape or tuple(w_r.shape[1:]) != expected_coarse_features:
            raise ValueError(
                "The coarse model cache does not match the RF-analysis grid. "
                f"Expected (time, {expected_coarse_features[0]}, {expected_coarse_features[1]}, "
                f"{expected_coarse_features[2]}, {expected_coarse_features[3]}), got "
                f"{w_r.shape} and {w_i.shape}. Re-run coarse wavelet decomposition and RF analysis."
            )
        raw_best_params = np.array(state["rfs_gabor"][1])
        smoothed_best_params = smooth_best_positions(
            raw_best_params,
            state["neuron_pos"],
        )
        for name, params in (("raw", raw_best_params), ("smoothed", smoothed_best_params)):
            coords = np.asarray(params[:2, neuron_id], dtype=float)
            limits = np.asarray(expected_coarse_features[:2], dtype=float) - 1
            if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords > limits):
                raise ValueError(
                    f"{name.title()} RF coordinates {tuple(coords)} are outside the coarse wavelet grid "
                    f"{tuple(expected_coarse_features[:2])}. Re-run coarse RF analysis for the current cache."
                )
        movie_metadata = _movie_metadata()
        dt1 = min(movie_metadata["frames"], state["spks"].shape[1], w_r.shape[0], w_i.shape[0])
        if dt1 < 2:
            raise ValueError("Run Model needs at least two frames shared by spikes and coarse wavelets.")
        frames_per_minute = int(round(movie_metadata["fps"] * 60))

        def call_model():
            """Function for call model.

            Returns:
                Result produced by the operation.
            """
            return run_Model(
                smoothed_best_params[:, [neuron_id]],
                raw_best_params[:, [neuron_id]],
                state["spks"][:, :, [neuron_id]],
                w_i,
                w_r,
                dt1=dt1,
                n_min=5,
                double_wavelet_model=False,
                train_idx=split_settings["train_idx"],
                test_idx=split_settings["test_idx"],
                lastmin=split_settings["lastmin"],
                # Model fitting runs in a worker thread.  Matplotlib/Tk figures
                # must only be constructed on Tk's main thread; the numerical
                # result is cached and rendered by the GUI afterwards.
                plotting=False,
                frames_per_minute=frames_per_minute,
            )

        # ``call_model`` deliberately runs without plotting in this worker.
        # Do not touch pyplot here; Tk/Matplotlib state belongs to the main UI
        # thread.  A backend-neutral summary figure is created below instead.
        result, figures = call_model(), []
        del w_r, w_i
        gc.collect()
        model_payload = _model_result_payload("run_Model", result, neuron_id)
        if not figures:
            figures = [_model_summary_figure("Run Model", model_payload)]
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
        """Function for plot run full model outputs.

        Returns:
            Result produced by the operation.
        """
        _ensure_model_imports("run_Full_Model plot capture")
        state = _require_rf_state()
        neuron_id = _selected_neuron_id()
        split_settings = _model_split_settings(state)
        cache_extra = {
            "model": "run_Full_Model",
            "neuron": neuron_id,
            **split_settings,
        }
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
        for name, params in (("raw", raw_best_params), ("smoothed", smoothed_best_params)):
            coords = np.asarray(params[:2, neuron_id], dtype=float)
            limits = np.asarray((state["coarse_nx"], state["coarse_ny"]), dtype=float) - 1
            if not np.all(np.isfinite(coords)) or np.any(coords < 0) or np.any(coords > limits):
                raise ValueError(
                    f"{name.title()} RF coordinates {tuple(coords)} are outside the coarse RF grid "
                    f"{tuple(limits.astype(int) + 1)}. Re-run coarse RF analysis for the current cache."
                )
        sigmas_full = np.array(parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"))
        frequencies = np.array(parse_literal(gabor_entries["Frequencies"].get(), "Frequencies"))
        wavelet_path = _wavelet_folder("full")
        save_path = _folder_from_entry(param_entries, "Full Model Save Path", _project_layout().model_dir)
        os.makedirs(save_path, exist_ok=True)
        movie_metadata = _movie_metadata()
        frames_per_minute = int(round(movie_metadata["fps"] * 60))

        def call_full_model():
            """Function for call full model.

            Returns:
                Result produced by the operation.
            """
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
                tt=[0, min(movie_metadata["frames"], state["nb_frames"], state["spks"].shape[1])],
                memmapping=True,
                train_idx=split_settings["train_idx"],
                test_idx=split_settings["test_idx"],
                double_wavelet_model=False,
                lastmin=split_settings["lastmin"],
                # As above, never create Tk/Matplotlib figures from the worker
                # thread that performs Full Model refinement.
                plotting=False,
                frames_per_minute=frames_per_minute,
                coarse_shape=(state["coarse_nx"], state["coarse_ny"]),
                hz=movie_metadata["fps"],
            )

        # See Run Model above: avoid pyplot inspection from the worker thread.
        result, figures = call_full_model(), []
        model_payload = _model_result_payload("run_Full_Model", result, neuron_id)
        if not figures:
            figures = [_model_summary_figure("Run Full Model", model_payload)]
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

    def plot_selected_model_outputs():
        """Run the model plotter that matches the selected analysis scale."""
        if _selected_analysis_scale() == "full":
            return plot_run_full_model_outputs()
        return plot_run_model_outputs()

    def click_save():
        """Function for click save."""
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
        """Function for export plots."""
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
        """Function for save app state."""
        analysis_values = {key: entry.get() for key, entry in param_entries.items() if key != "Neuron ID"}
        modality_keys = (
            {"Resolution", "Number of Planes"}
            if workflow == WORKFLOW_2P
            else {"Sampling Rate (samples / sec)"}
        )
        modality_values = {
            key: value for key, value in analysis_values.items() if key in modality_keys
        }
        common_values = {
            key: value for key, value in analysis_values.items() if key not in modality_keys
        }
        state = {
            "workflow": workflow,
            "gui": {
                "wavelet_backend": _selected_wavelet_backend(),
                "downsample_percent": _selected_downsample_percent(),
                "gabor_format": gabor_format_var.get(),
                "downsample_format": _selected_downsample_format(),
                "neural_source": _selected_neural_source(),
                "neural_cache_format": _selected_neural_cache_format(),
            },
            "gabor_param": {key: entry.get() for key, entry in gabor_entries.items()},
            "common": common_values,
            "two_photon": modality_values if workflow == WORKFLOW_2P else {},
            "ephys": modality_values if workflow == WORKFLOW_EPHYS else {},
        }
        path = filedialog.asksaveasfilename(
            title="Save Current GUI Inputs and Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            initialfile="pipeline_config.json",
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
        """Function for load app state."""
        path = filedialog.askopenfilename(
            title="Load GUI Inputs and Parameters",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            loaded_workflow = state.get("workflow")
            if loaded_workflow in (WORKFLOW_2P, WORKFLOW_EPHYS) and loaded_workflow != workflow:
                workflow_var.set(loaded_workflow)
                set_workflow_from_panel(loaded_workflow)
            effective_workflow = loaded_workflow if loaded_workflow in (WORKFLOW_2P, WORKFLOW_EPHYS) else workflow
            gui_state = state.get("gui") or state.get("save_options") or {}
            loaded_backend = gui_state.get("wavelet_backend", state.get("wavelet_backend"))
            if loaded_backend in {"legacy", "convolution"}:
                wavelet_backend_var.set(loaded_backend)
                set_wavelet_backend_from_panel(loaded_backend)
            loaded_percent = gui_state.get("downsample_percent", state.get("downsample_percent"))
            if loaded_percent is not None:
                downsample_percent_var.set(float(loaded_percent))
            loaded_gabor = state.get("gabor_param") or state.get("gabor") or {}
            for key, value in loaded_gabor.items():
                if key in gabor_entries:
                    if key == "Phases":
                        phases = np.asarray(parse_literal(value, "Phases"), dtype=float)
                        # Saved configurations before this change stored radians.
                        if phases.size and np.nanmax(np.abs(phases)) <= 2 * np.pi + 1e-6:
                            value = repr(np.rad2deg(phases).round(10).tolist())
                    gabor_entries[key].delete(0, tk.END)
                    gabor_entries[key].insert(0, str(value))
            analysis_values = state.get("analysis") or state.get("param_defaults") or {}
            if "common" in state:
                analysis_values = {
                    **state.get("common", {}),
                    **(
                        state.get("two_photon", {})
                        if effective_workflow == WORKFLOW_2P
                        else state.get("ephys", {})
                    ),
                }
            render_parameter_fields(preserve_values=True, loaded_values=analysis_values)
            for key, value in analysis_values.items():
                if key in param_entries:
                    param_entries[key].delete(0, tk.END)
                    param_entries[key].insert(0, str(value))
            save_options = gui_state
            gabor_format_var.set(save_options.get("gabor_format", gabor_format_var.get()))
            downsample_format = save_options.get("downsample_format")
            if downsample_format in {"npy", "zarr"}:
                downsample_format_var.set(downsample_format)
            neural_source = save_options.get("neural_source")
            if neural_source in {"data_dir", "spks_path"}:
                neural_source_var.set(neural_source)
            neural_cache_format = save_options.get("neural_cache_format")
            if neural_cache_format in {"npy", "zarr"}:
                neural_cache_format_var.set(neural_cache_format)
            _apply_project_layout_defaults(force=True)
            try:
                _refresh_neural_source_controls()
                _refresh_downsample_controls()
                _sync_completed_actions_from_artifacts()
                refresh_action_buttons()
            except NameError:
                pass
            refresh_size_estimates()
            print(f"Loaded GUI state from: {path}")
        except Exception as exc:
            messagebox.showerror("Load Failed", f"Could not load GUI state: {exc}")
            print(f"Failed to load GUI state: {exc}")


    def cleanup_temporary_directories():
        """Function for cleanup temporary directories."""
        for temp_dir in list(temp_directories):
            if os.path.isdir(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Removed temporary folder: {temp_dir}")
                except Exception as exc:
                    print(f"Could not remove temporary folder {temp_dir}: {exc}")
        temp_directories.clear()

    def quit_app():
        """Function for quit app."""
        keep_awake.stop()
        cleanup_temporary_directories()
        root.quit()
        root.destroy()

    def browse_path(entry_widget, kind, field_name=None):
        """Open a file or directory picker appropriate for the field type."""
        if kind == "file":
            path = filedialog.askopenfilename(title="Select file")
        elif kind == "savefile":
            path = filedialog.asksaveasfilename(title="Select output file")
        else:
            path = filedialog.askdirectory(title="Select directory")
        if path:
            selected = Path(path)
            if field_name == "Project Root":
                _set_entry_value(entry_widget, selected)
                _apply_project_layout_defaults(force=True)
            elif field_name in BROWSE_KIND and kind == "dir":
                try:
                    conventional = _layout_field_folder(field_name)
                    selected_abs = selected.resolve()
                    conventional_abs = conventional.resolve()
                    if selected_abs != conventional_abs:
                        if field_name in {"Dir", "Movie Path", "Spks Path"}:
                            ref_kind = {
                                "Dir": "raw_data",
                                "Movie Path": "stimulus_movie",
                                "Spks Path": "neural_cache",
                            }[field_name]
                            write_reference(conventional, selected, ref_kind)
                            print(f"Stored {field_name} reference: {conventional} -> {selected}")
                        else:
                            print(
                                f"Strict layout keeps generated {field_name} artifacts in {conventional}. "
                                "Change Project Root to relocate the complete project tree."
                            )
                    _set_entry_value(entry_widget, conventional)
                except Exception as exc:
                    print(f"Could not store layout reference for {field_name}: {exc}")
                    _set_entry_value(entry_widget, selected)
            else:
                _set_entry_value(entry_widget, selected)
            refresh_size_estimates()

    def _check_movie_metadata_against_gui(movie_path):
        """Compare lightweight movie metadata with GUI fields and offer updates."""
        try:
            import cv2

            cap = cv2.VideoCapture(str(movie_path))
            try:
                if not cap.isOpened():
                    return
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
            finally:
                cap.release()
        except Exception:
            return

        if min(frames, width, height) > 0 and fps > 0:
            print(
                "Stimulus metadata is authoritative: "
                f"{width} x {height} px, {frames} frames, {fps:.6g} fps."
            )

    def _movie_frame_count(path, fallback):
        """Return actual movie frame count when the selected movie can be opened."""
        try:
            if not path or not os.path.exists(path):
                return fallback
            import cv2

            cap = cv2.VideoCapture(path)
            try:
                if not cap.isOpened():
                    return fallback
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                return frames if frames > 0 else fallback
            finally:
                cap.release()
        except Exception:
            return fallback

    def _gabor_library_npy_bytes(nx, ny, n_theta, sigmas, offsets, frequencies):
        """Exact byte count for the in-memory/NPY Gabor library tensor."""
        flat_size = nx * ny
        dtype_size = np.dtype(np.float16).itemsize
        n_sigmas = max(1, len(sigmas))
        n_offsets = max(1, len(offsets))
        n_frequencies = max(1, len(frequencies))
        if len(frequencies) and np.any(np.asarray(frequencies, dtype=float) != 0):
            shape = (
                nx,
                ny,
                n_theta,
                n_sigmas,
                n_frequencies,
                n_offsets,
                flat_size,
            )
        else:
            shape = (nx, ny, n_theta, n_sigmas, n_offsets, flat_size)
        return int(np.prod(shape, dtype=np.int64) * dtype_size), shape

    def estimate_gabor_library_size():
        """Function for estimate gabor library size."""
        try:
            nx, ny = _stimulus_grid_dimensions(movie_path=_find_movie_path())
            n_theta = int(gabor_entries["N_thetas"].get())
            sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
            fine_sigmas = _ordered_float_union(
                sigmas,
                parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"),
            )
            offsets = _gabor_phase_offsets_radians()
            frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
            coarse_nx, coarse_ny = _analysis_grid_dimensions(nx, ny, "coarse")
            full_nx, full_ny = _analysis_grid_dimensions(nx, ny, "full")
            coarse_bytes, coarse_shape = _gabor_library_npy_bytes(
                coarse_nx,
                coarse_ny,
                n_theta,
                sigmas,
                offsets,
                [0],
            )
            fine_bytes, fine_shape = _gabor_library_npy_bytes(
                full_nx,
                full_ny,
                n_theta,
                fine_sigmas,
                offsets,
                frequencies,
            )
            if gabor_format_var.get() == "zarr":
                gabor_size_label.configure(
                    text=(
                        "Legacy Gabor Zarr sizes are compression-dependent; NPY equivalents: "
                        f"coarse {_format_bytes(coarse_bytes)} {coarse_shape}; "
                        f"full {_format_bytes(fine_bytes)} {fine_shape}."
                    )
                )
            else:
                gabor_size_label.configure(
                    text=(
                        f"Legacy Gabor NPY sizes: coarse {_format_bytes(coarse_bytes)} {coarse_shape}; "
                        f"full {_format_bytes(fine_bytes)} {fine_shape}"
                    )
                )
        except Exception:
            gabor_size_label.configure(text="Gabor disk size: enter valid dimensions to calculate")

    def estimate_wavelet_size():
        """Function for estimate wavelet size."""
        try:
            movie_entry = param_entries.get("Movie Path")
            movie_path = ""
            if movie_entry is not None:
                try:
                    movie_path = str(find_stimulus_movie(movie_entry.get().strip()))
                except Exception:
                    movie_path = movie_entry.get().strip()
            metadata = _movie_metadata(movie_path)
            n_frames = metadata["frames"]
            nx, ny = _stimulus_grid_dimensions(movie_path=movie_path)
            n_thetas = int(gabor_entries["N_thetas"].get())
            sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
            sigmas_full = parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model")
            frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
            n_sigmas = len(sigmas)
            n_sigmas_full = len(sigmas_full)
            n_frequencies = max(1, len(frequencies))
            coarse_nx, coarse_ny = _analysis_grid_dimensions(nx, ny, "coarse")
            full_nx, full_ny = _analysis_grid_dimensions(nx, ny, "full")
            bytes_per_float = np.dtype(np.float32).itemsize
            coarse_phase_bytes = n_frames * coarse_nx * coarse_ny * n_thetas * n_sigmas * bytes_per_float
            coarse_rf_bytes = coarse_phase_bytes
            coarse_model_bytes = 2 * coarse_phase_bytes
            full_model_raw_bytes = 2 * n_frames * full_nx * full_ny * n_thetas * n_sigmas_full * n_frequencies * bytes_per_float
            backend = "convolution" if _selected_wavelet_backend() == "convolution" else "legacy"
            wavelet_size_label.configure(
                text=(
                    f"{backend.title()} internal Zarr products (uncompressed equivalents): "
                    f"Coarse RF power {_format_bytes(coarse_rf_bytes)}; "
                    f"Run Model real + imaginary {_format_bytes(coarse_model_bytes)}; "
                    f"Run Full Model real + imaginary {_format_bytes(full_model_raw_bytes)}. "
                    "Zarr compression is data-dependent."
                )
            )
        except Exception:
            wavelet_size_label.configure(text="Wavelet disk size: enter valid dimensions to calculate")

    def refresh_size_estimates():
        """Function for refresh size estimates."""
        estimate_gabor_library_size()
        estimate_wavelet_size()

    GABOR_ESTIMATE_KEYS = {
        "N_thetas",
        "Sigmas",
        "Sigmas Full Model",
        "Phases",
        "Frequencies",
        "Save Path",
    }
    WAVELET_ESTIMATE_KEYS = {"Movie Path", "N_thetas", "Sigmas", "Sigmas Full Model", "Frequencies", "Full Model Wavelet Path"}

    def _on_config_entry_changed(key):
        """Invalidate only the completed stages that consume ``key``."""
        if key in {
            "Dir", "Spks Path", "Experiment Info", "Block End", "Resolution",
            "Number of Planes", "Sampling Rate (samples / sec)",
        }:
            _invalidate_completed_actions("neural", "rf")
        if key in {
            "Movie Path", "Visual Coverage", "Analysis Coverage",
        }:
            _invalidate_completed_actions("downsample", "wavelet", "rf")
        if key in {
            "N_thetas", "Sigmas", "Frequencies", "Phases",
            "Coarse Library Path", "Fine Library Path",
        }:
            _invalidate_completed_actions("gabor", "wavelet", "rf")
        if key == "Sigmas Full Model":
            completed_actions.difference_update({"gabor:full", "wavelet:full", "rf"})
        if key == "Path Directory":
            completed_actions.difference_update({"wavelet:coarse", "rf"})
        if key == "Full Model Wavelet Path":
            completed_actions.discard("wavelet:full")
        try:
            refresh_action_buttons()
        except NameError:
            pass

    def add_config_row(parent, key, default, entries_dict, row, bg, labels_map):
        """Render one labeled configuration row with an optional typed browse button."""
        label_text = labels_map.get(key, key)
        label_widget = ctk.CTkLabel(
            parent, text=label_text, text_color=text_color, font=ctk.CTkFont(size=12)
        )
        label_widget.grid(row=row, column=0, sticky="w", pady=4)

        entry_wrap = ctk.CTkFrame(parent, fg_color="transparent")
        entry_wrap.grid(row=row, column=1, pady=3, padx=(10, 0), sticky="ew")
        entry_wrap.columnconfigure(0, weight=1)

        hint = INPUT_HINTS.get(key, "")
        entry = ctk.CTkEntry(
            entry_wrap,
            height=30,
            corner_radius=6,
            border_width=1,
            placeholder_text=hint,
        )
        entry.insert(0, default)
        entry.grid(row=0, column=0, sticky="ew")
        ToolTip(entry)
        entries_dict[key] = entry

        entry.bind("<KeyRelease>", lambda event, name=key: _on_config_entry_changed(name))
        if key in GABOR_ESTIMATE_KEYS or key in WAVELET_ESTIMATE_KEYS:
            entry.bind("<KeyRelease>", lambda event: refresh_size_estimates(), add="+")

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
                command=lambda e=entry, k=browse_kind, name=key: browse_path(e, k, name),
            ).grid(row=0, column=1, padx=(5, 0))
        elif hint:
            ctk.CTkLabel(
                entry_wrap,
                text=hint,
                text_color=muted_text,
                font=ctk.CTkFont(size=10),
                anchor="w",
            ).grid(row=1, column=0, sticky="w", pady=(1, 0))
        return label_widget, entry_wrap

    # --- Root Window Setup & Theming ---
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    keep_awake = KeepAwake("waven analysis GUI is open")
    keep_awake.start()
    workflow_label = workflow_display_name(workflow)
    root.title(f"Neuron Analysis Toolkit — {workflow_label}")

    def on_closing():
        """Function for on closing."""
        if messagebox.askokcancel("Quit", "Are you sure you want to close the application? Unsaved temporary data will be removed."):
            keep_awake.stop()
            cleanup_temporary_directories()
            root.quit()
            root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.geometry("1600x1000")
    root.resizable(True, True)

    try:
        icon_base64 = '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'
        if icon_base64:
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

    def tune_scrollable_frame(scrollable_frame, increment=36):
        """Make mouse-wheel scrolling feel steadier on CustomTkinter scroll frames."""
        for attr in ("_parent_canvas", "_canvas"):
            canvas = getattr(scrollable_frame, attr, None)
            if canvas is not None:
                try:
                    canvas.configure(yscrollincrement=increment, highlightthickness=0)
                except Exception:
                    pass

    def set_pane_sash(pane, index, position):
        """Function for set pane sash.

        Args:
            pane: Input value for this operation.
            index: Input value for this operation.
            position: Input value for this operation.
        """
        try:
            pane.sashpos(index, position)
            return
        except AttributeError:
            pass
        try:
            if str(pane.cget("orient")) == str(tk.VERTICAL):
                pane.sash_place(index, 1, int(position))
            else:
                pane.sash_place(index, int(position), 1)
        except Exception:
            pass

    # --- View Menu ---
    menubar = tk.Menu(root)
    view_menu = tk.Menu(menubar, tearoff=0)

    terminal_visible = [True]

    def toggle_terminal():
        """Function for toggle terminal."""
        try:
            if terminal_visible[0] and str(frame_log) in right_pane.panes():
                right_pane.forget(frame_log)
                terminal_visible[0] = False
            else:
                right_pane.add(frame_log, minsize=220, stretch="always")
                terminal_visible[0] = True
                show_terminal_half()
        except Exception as exc:
            print(f"Could not toggle terminal pane: {exc}")

    left_panel_visible = [True]

    def toggle_left_panel():
        """Function for toggle left panel."""
        try:
            if str(left_frame) in paned_h.panes():
                paned_h.forget(left_frame)
                left_panel_visible[0] = False
            else:
                paned_h.add(left_frame, minsize=340, stretch="never", before=right_pane)
                left_panel_visible[0] = True
                root.after_idle(lambda: set_pane_sash(paned_h, 0, 420))
        except Exception as exc:
            print(f"Could not toggle left pane: {exc}")

    view_menu.add_command(label="Toggle Left Panel", command=toggle_left_panel)
    view_menu.add_command(label="Toggle Terminal", command=toggle_terminal)
    menubar.add_cascade(label="View", menu=view_menu)
    root.config(menu=menubar)

    content_root = ctk.CTkFrame(root, fg_color=bg_color, corner_radius=0)
    content_root.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 0))

    status_frame = ctk.CTkFrame(content_root, fg_color="transparent")
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(6, 0))
    status_var = tk.StringVar(value="Ready")
    progress_bar = ttk.Progressbar(status_frame, mode="determinate", length=260, maximum=100)
    progress_bar.pack(side=tk.RIGHT, padx=(8, 0))

    # Split layout using paned windows so left and terminal are resizable
    paned_h = tk.PanedWindow(
        content_root,
        orient=tk.HORIZONTAL,
        opaqueresize=False,
        sashwidth=8,
        sashrelief=tk.FLAT,
        bg=bg_color,
        bd=0,
        showhandle=False,
    )
    paned_h.pack(fill=tk.BOTH, expand=True)

    # Left pane (resizable horizontally)
    left_frame = ttk.Frame(paned_h, style="TFrame")
    paned_h.add(left_frame, minsize=340, stretch="never")

    # Right pane is a vertical paned window so the terminal is resizable vertically
    right_pane = tk.PanedWindow(
        paned_h,
        orient=tk.VERTICAL,
        opaqueresize=False,
        sashwidth=8,
        sashrelief=tk.FLAT,
        bg=bg_color,
        bd=0,
        showhandle=False,
    )
    paned_h.add(right_pane, minsize=700, stretch="always")

    # Container for left content (scrollable inside)
    container_left = ctk.CTkFrame(left_frame, fg_color=frame_color, corner_radius=8)
    container_left.pack(fill=tk.BOTH, expand=True)

    # Right-top visualization area
    container_right_top = ctk.CTkFrame(right_pane, fg_color=frame_color, corner_radius=8)
    right_pane.add(container_right_top, minsize=260, stretch="always")

    # Right-bottom terminal
    frame_log = ctk.CTkFrame(right_pane, fg_color="#FFFFFF", corner_radius=8)
    right_pane.add(frame_log, minsize=220, stretch="always")
    terminal_status_var = tk.StringVar(value="Task: idle")
    terminal_metrics_var = tk.StringVar(value="")

    terminal_toolbar = ctk.CTkFrame(frame_log, fg_color="transparent")
    terminal_toolbar.pack(fill=tk.X, padx=10, pady=(8, 4))
    ctk.CTkLabel(
        terminal_toolbar,
        text="Terminal",
        text_color=primary_btn,
        font=ctk.CTkFont(size=13, weight="bold"),
    ).pack(side=tk.LEFT)
    terminal_status_label = ctk.CTkLabel(
        terminal_toolbar,
        textvariable=terminal_status_var,
        text_color=text_color,
        font=ctk.CTkFont(size=12),
    )
    terminal_status_label.pack(side=tk.LEFT, padx=(14, 0))
    terminal_metrics_label = ctk.CTkLabel(
        terminal_toolbar,
        textvariable=terminal_metrics_var,
        text_color=muted_text,
        font=ctk.CTkFont(size=12),
    )
    terminal_metrics_label.pack(side=tk.LEFT, padx=(14, 0))
    btn_cancel_terminal = ctk.CTkButton(
        terminal_toolbar,
        text="Cancel",
        width=76,
        height=26,
        corner_radius=6,
        fg_color=danger_btn,
        hover_color="#A4262C",
        state=tk.DISABLED,
        command=request_cancel_current_task,
    )
    btn_cancel_terminal.pack(side=tk.RIGHT, padx=(8, 0))
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
        wrap="word", borderwidth=0, highlightthickness=0,
    )
    text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=(0, 10))
    log_scroll.config(command=text_log.yview)
    text_log.configure(blockcursor=False)

    def _terminal_mousewheel(event):
        """Function for terminal mousewheel.

        Args:
            event: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        try:
            step = -int(event.delta / 120) if getattr(event, "delta", 0) else 0
            text_log.yview_scroll(step * 3, "units")
            return "break"
        except Exception:
            return None

    text_log.bind("<MouseWheel>", _terminal_mousewheel)
    text_log.bind("<Button-4>", lambda event: (text_log.yview_scroll(-3, "units"), "break")[1])
    text_log.bind("<Button-5>", lambda event: (text_log.yview_scroll(3, "units"), "break")[1])
    sys.stdout = RedirectText(text_log)
    sys.stderr = RedirectText(text_log)

    # Set sensible initial sash positions after layout
    root.update_idletasks()
    try:
        set_pane_sash(paned_h, 0, 420)
        pane_height = right_pane.winfo_height() or root.winfo_height()
        set_pane_sash(right_pane, 0, int(pane_height * 0.56))
    except Exception:
        pass
    root.after(150, show_terminal_half)
    root.after(500, show_terminal_half)

    # Left panel scrollable content
    frame_left = ctk.CTkScrollableFrame(
        container_left,
        fg_color=frame_color,
        corner_radius=8,
        scrollbar_button_color="#CBD5E1",
        scrollbar_button_hover_color="#94A3B8",
    )
    frame_left.pack(fill=tk.BOTH, expand=True)
    tune_scrollable_frame(frame_left, increment=32)

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
    tune_scrollable_frame(frame_plot_all, increment=36)
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
    tune_scrollable_frame(frame_plot_individual, increment=36)

    # --- Session configuration ---
    frame_session = ttk.LabelFrame(frame_left, text="Session Configuration", padding=15)
    frame_session.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)

    workflow_var = tk.StringVar(value=workflow)

    def set_workflow_from_panel(value):
        """Function for set workflow from panel.

        Args:
            value: Input value for this operation.
        """
        nonlocal workflow, workflow_label
        workflow = value
        workflow_label = workflow_display_name(workflow)
        root.title(f"Neuron Analysis Toolkit — {workflow_label}")
        try:
            frame_params.configure(text=f"Experiment Configuration ({workflow_label})")
            render_parameter_fields(preserve_values=True)
            refresh_size_estimates()
        except NameError:
            pass

    workflow_frame = ctk.CTkFrame(frame_session, fg_color="transparent")
    workflow_frame.pack(fill=tk.X, pady=(0, 8))
    ctk.CTkLabel(
        workflow_frame,
        text="Neural data type",
        text_color=text_color,
        font=ctk.CTkFont(size=12, weight="bold"),
    ).pack(anchor="w", pady=(0, 4))
    workflow_segment = ctk.CTkSegmentedButton(
        workflow_frame,
        values=[WORKFLOW_2P, WORKFLOW_EPHYS],
        variable=workflow_var,
        command=set_workflow_from_panel,
        height=28,
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
    workflow_segment.pack(fill=tk.X)
    workflow_segment.set(workflow)

    initial_backend = gui_options.get("wavelet_backend", "legacy")
    initial_neural_source = gui_options.get("neural_source", "data_dir")
    wavelet_backend_var = tk.StringVar(value=initial_backend if initial_backend in {"legacy", "convolution"} else "legacy")
    initial_downsample_format = gui_options.get("downsample_format", "npy")
    downsample_format_var = tk.StringVar(value=initial_downsample_format if initial_downsample_format in {"npy", "zarr"} else "npy")
    downsample_percent_var = tk.DoubleVar(value=float(gui_options.get("downsample_percent", 20.0)))
    neural_source_var = tk.StringVar(value=initial_neural_source if initial_neural_source in {"data_dir", "spks_path"} else "data_dir")
    neural_source_display_var = tk.StringVar(
        value="Continue / existing cache" if neural_source_var.get() == "spks_path" else "Fresh / raw data"
    )
    initial_neural_format = gui_options.get("neural_cache_format", "npy")
    neural_cache_format_var = tk.StringVar(value=initial_neural_format if initial_neural_format in {"npy", "zarr"} else "npy")

    def refresh_scale_controls():
        """Refresh shared-grid controls for the selected backend."""
        backend = _selected_wavelet_backend()
        try:
            if backend == "convolution":
                btn_submit_gabor.configure(text="Prepare Convolution Kernels (Coarse RF + Full Model)")
            else:
                btn_submit_gabor.configure(text="Prepare Gabor Assets (Coarse RF + Full Model)")
            # Legacy materializes a flattened library; convolution writes a
            # compact kernel cache, so library storage estimates do not apply.
            if backend == "legacy":
                format_frame.grid()
                gabor_size_label.grid()
            else:
                format_frame.grid_remove()
                gabor_size_label.grid_remove()
            wavelet_format_segment.configure(state="normal")
            _refresh_scale_field_visibility()
            metadata = _movie_metadata()
            grid_width, grid_height = _stimulus_grid_dimensions()
            gabor_dimensions_label.configure(
                text=(f"Movie metadata: {metadata['width']} × {metadata['height']} px at "
                      f"{metadata['fps']:.3g} fps  →  analysis grid {grid_width} × {grid_height} px")
            )
            refresh_action_buttons()
            refresh_size_estimates()
        except NameError:
            pass

    def set_wavelet_backend_from_panel(value):
        """Update the active legacy/convolution wavelet backend."""
        if value not in {"legacy", "convolution"}:
            return
        wavelet_backend_var.set(value)
        _invalidate_completed_actions("gabor", "wavelet", "rf")
        refresh_scale_controls()

    ctk.CTkLabel(
        frame_session,
        text="One metadata-derived analysis grid is shared by Coarse RF, Run Model, and Run Full Model.",
        text_color=muted_text,
        wraplength=330,
        justify="left",
    ).pack(fill=tk.X, pady=(8, 2))

    backend_frame = ctk.CTkFrame(frame_session, fg_color="transparent")
    backend_frame.pack(fill=tk.X, pady=(8, 8))
    ctk.CTkLabel(
        backend_frame,
        text="Wavelet backend",
        text_color=text_color,
        font=ctk.CTkFont(size=12, weight="bold"),
    ).pack(anchor="w", pady=(0, 4))
    wavelet_backend_segment = ctk.CTkSegmentedButton(
        backend_frame,
        values=["legacy", "convolution"],
        variable=wavelet_backend_var,
        command=set_wavelet_backend_from_panel,
        height=28,
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
    wavelet_backend_segment.pack(fill=tk.X)
    wavelet_backend_segment.set(wavelet_backend_var.get())

    btn_load_state = ctk.CTkButton(
        frame_session,
        text="Load GUI Inputs / Parameters",
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=load_app_state,
    )
    btn_load_state.pack(fill=tk.X, pady=3)
    btn_save_state = ctk.CTkButton(
        frame_session,
        text="Save Current GUI Inputs / Parameters",
        fg_color="#4B5563",
        hover_color="#374151",
        command=save_app_state,
    )
    btn_save_state.pack(fill=tk.X, pady=3)

    stage_tabs = ctk.CTkTabview(
        frame_left,
        height=740,
        corner_radius=8,
        fg_color=frame_color,
        segmented_button_selected_color=primary_btn,
        segmented_button_selected_hover_color="#1D4ED8",
    )
    stage_tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    stage_downsample = stage_tabs.add("1 Stimulus & Metadata")
    stage_setup = stage_tabs.add("2 Session Setup")
    stage_gabor = stage_tabs.add("3 Gabor")
    stage_wavelet = stage_tabs.add("4 Wavelet Products")
    stage_analysis = stage_tabs.add("5 Analysis")
    stage_export = stage_tabs.add("6 Export")

    # --- Gabor filter bank ---
    frame_gabor = ttk.LabelFrame(stage_gabor, text="Gabor Filter Bank", padding=15)
    frame_gabor.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_gabor.columnconfigure(1, weight=1)

    gabor_entries = {}
    gabor_row_widgets = {}
    editable_gabor_params = [(key, value) for key, value in gabor_param.items() if key not in {"NX", "NY"}]
    for i, (label, default) in enumerate(editable_gabor_params):
        gabor_row_widgets[label] = add_config_row(
            frame_gabor, label, default, gabor_entries, i, frame_color, GABOR_LABELS
        )

    gabor_dimensions_label = ctk.CTkLabel(
        frame_gabor,
        text="Stimulus grid: select a movie in Step 1",
        text_color=muted_text,
    )
    gabor_dimensions_label.grid(row=len(editable_gabor_params), column=0, columnspan=2, sticky="w", pady=(6, 0))

    initial_gabor_format = gui_options.get("gabor_format", "npy")
    gabor_format_var = tk.StringVar(value=initial_gabor_format if initial_gabor_format in {"npy", "zarr"} else "npy")
    btn_submit_gabor = ctk.CTkButton(
        frame_gabor,
        text="Prepare Gabor Assets (Coarse RF + Full Model)",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(create_both_gabor_libraries, "Gabor asset preparation"),
    )
    btn_submit_gabor.grid(row=len(editable_gabor_params)+1, column=0, columnspan=2, pady=(15, 0), sticky="ew")

    format_frame = ctk.CTkFrame(frame_gabor, fg_color="transparent")
    format_frame.grid(row=len(editable_gabor_params)+2, column=0, columnspan=2, pady=(10, 0), sticky="w")
    ctk.CTkLabel(format_frame, text="Library format:", text_color=muted_text).pack(side=tk.LEFT)

    def _set_gabor_format(val):
        """Function for set gabor format.

        Args:
            val: Input value for this operation.
        """
        gabor_format_var.set(val)
        _invalidate_completed_actions("gabor", "wavelet", "rf")
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
        text="Gabor disk size: enter valid dimensions to calculate",
        font=main_font,
        background=frame_color,
        foreground=text_color,
    )
    gabor_size_label.grid(row=len(editable_gabor_params)+3, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # --- Stimulus wavelet pipeline ---
    frame_processing = ttk.LabelFrame(stage_wavelet, text="Stimulus Wavelet Pipeline", padding=15)
    frame_processing.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    initial_wavelet_format = gui_options.get("wavelet_format", "zarr")
    wavelet_format_var = tk.StringVar(value=initial_wavelet_format if initial_wavelet_format in {"npy", "zarr"} else "zarr")
    btn_submit_wavelet = ctk.CTkButton(
        frame_processing,
        text="Prepare Coarse RF Power Cache",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(lambda: run_wavelet("coarse_rf"), "Prepare Coarse RF wavelets"),
    )
    btn_submit_wavelet.pack(fill=tk.X, pady=3)

    btn_prepare_model_wavelets = ctk.CTkButton(
        frame_processing,
        text="Prepare Run Model Phase Caches",
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=run_in_thread(lambda: run_wavelet("model"), "Prepare Run Model wavelets"),
    )
    btn_prepare_model_wavelets.pack(fill=tk.X, pady=3)

    btn_prepare_full_model_wavelets = ctk.CTkButton(
        frame_processing,
        text="Prepare Run Full Model Phase Caches",
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=run_in_thread(lambda: run_wavelet("full_model"), "Prepare Run Full Model wavelets"),
    )
    btn_prepare_full_model_wavelets.pack(fill=tk.X, pady=3)

    format_frame_wavelet = ctk.CTkFrame(frame_processing, fg_color="transparent")
    format_frame_wavelet.pack(anchor="w", pady=(10, 0))
    ctk.CTkLabel(format_frame_wavelet, text="Wavelet storage format:", text_color=muted_text).pack(side=tk.LEFT)

    def _set_wavelet_format(val):
        """Function for set wavelet format.

        Args:
            val: Input value for this operation.
        """
        wavelet_format_var.set(val)
        completed_actions.discard("wavelet:full")
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
        text="Wavelet disk size: enter valid dimensions to calculate",
        font=main_font,
        background=frame_color,
        foreground=text_color,
    )
    wavelet_size_label.pack(anchor="w", pady=(6, 0))

    ttk.Separator(frame_processing, orient="horizontal").pack(fill=tk.X, pady=8)

    # --- Experiment configuration ---
    frame_params = ttk.LabelFrame(
        stage_setup,
        text=f"Experiment Configuration ({workflow_label})",
        padding=15,
    )
    frame_params.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_params.columnconfigure(1, weight=1)

    param_entries = {}
    PARAMETER_GROUPS = {
        "Data Input": [
            "Project Root",
            "Experiment Info",
        ],
        "Acquisition & Timing": [
            "Number of Planes",
            "Sampling Rate (samples / sec)",
            "Train Trial Indices",
            "Test Trial Indices",
            "Use Last Minute Holdout",
            "Block End",
        ],
        "Spatial & Wavelet": [
            "Resolution",
            "Visual Coverage",
            "Analysis Coverage",
        ],
    }

    def render_parameter_fields(preserve_values=False, loaded_values=None):
        """Function for render parameter fields.

        Args:
            preserve_values: Input value for this operation.
            loaded_values: Input value for this operation.
        """
        external_param_keys = {
            "Neuron ID", "Dir", "Spks Path", "Movie Path", "Path Directory",
            "Full Model Wavelet Path", "Full Model Save Path", "Plot Cache Path",
            "Recovery Cache Directory",
        }
        existing_values = {}
        if preserve_values:
            existing_values = {
                key: entry.get()
                for key, entry in param_entries.items()
                if key not in {"Neuron ID", "Dir", "Spks Path"}
            }
        if loaded_values:
            existing_values.update({str(key): str(value) for key, value in loaded_values.items()})

        for widget in frame_params.winfo_children():
            widget.destroy()
        for key in list(param_entries):
            if key not in external_param_keys:
                param_entries.pop(key, None)

        workflow_param_keys, filtered_param_defaults = workflow_defaults(workflow)
        for section_title, section_keys in PARAMETER_GROUPS.items():
            section_frame = ttk.LabelFrame(frame_params, text=section_title, padding=(10, 8))
            section_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
            section_frame.columnconfigure(1, weight=1)
            section_row = 0
            for key in section_keys:
                if key not in workflow_param_keys:
                    continue
                default = existing_values.get(key, filtered_param_defaults.get(key, ""))
                add_config_row(section_frame, key, default, param_entries, section_row, frame_color, ANALYSIS_LABELS)
                section_row += 1

    render_parameter_fields()
    recovery_frame = ttk.LabelFrame(frame_session, text="Recovery & Resume", padding=(10, 8))
    recovery_frame.pack(fill=tk.X, pady=(8, 0))
    recovery_frame.columnconfigure(1, weight=1)
    add_config_row(
        recovery_frame, "Recovery Cache Directory",
        param_defaults.get("Recovery Cache Directory", ""),
        param_entries, 0, frame_color, ANALYSIS_LABELS,
    )
    wavelet_paths_frame = ttk.LabelFrame(frame_processing, text="Wavelet Cache Folders", padding=(10, 8))
    wavelet_paths_frame.pack(fill=tk.X, pady=(8, 0))
    wavelet_paths_frame.columnconfigure(1, weight=1)
    coarse_wavelet_path_row = add_config_row(
        wavelet_paths_frame, "Path Directory", param_defaults.get("Path Directory", ""),
        param_entries, 0, frame_color, ANALYSIS_LABELS,
    )
    full_wavelet_path_row = add_config_row(
        wavelet_paths_frame, "Full Model Wavelet Path", param_defaults.get("Full Model Wavelet Path", ""),
        param_entries, 1, frame_color, ANALYSIS_LABELS,
    )
    full_sigmas_row = add_config_row(
        wavelet_paths_frame, "Sigmas Full Model", param_defaults.get("Sigmas Full Model", ""),
        param_entries, 2, frame_color, ANALYSIS_LABELS,
    )
    refresh_size_estimates()

    # --- Neural spike/position cache ---
    frame_neural_cache = ttk.LabelFrame(stage_setup, text="Neural Spike/Position Cache", padding=15)
    frame_neural_cache.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_neural_cache.columnconfigure(1, weight=1)
    _, neural_section_defaults = workflow_defaults(workflow)

    neural_source_frame = ctk.CTkFrame(frame_neural_cache, fg_color="transparent")
    neural_source_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
    ctk.CTkLabel(
        neural_source_frame,
        text="Neural cache source",
        text_color=muted_text,
    ).pack(side=tk.LEFT)

    neural_row_widgets = {}
    neural_source_labels = {
        "Fresh / raw data": "data_dir",
        "Continue / existing cache": "spks_path",
    }

    def _refresh_neural_source_controls(value=None):
        """Show only the neural input required by the selected starting point."""
        if value in neural_source_labels:
            neural_source_var.set(neural_source_labels[value])
            _invalidate_completed_actions("neural", "rf")
        source = _selected_neural_source()
        try:
            neural_source_display_var.set(
                "Fresh / raw data" if source == "data_dir" else "Continue / existing cache"
            )
            for key, row_widgets in neural_row_widgets.items():
                visible = (key == "Dir" and source == "data_dir") or (
                    key == "Spks Path" and source == "spks_path"
                )
                for widget in row_widgets:
                    widget.grid() if visible else widget.grid_remove()
            if source == "data_dir":
                neural_cache_format_frame.grid()
                btn_create_neural_cache.configure(text="Create pos/spikes Cache")
            else:
                neural_cache_format_frame.grid_remove()
                btn_create_neural_cache.configure(text="Validate Existing Neural Cache")
        except Exception:
            pass

    neural_source_segment = ctk.CTkSegmentedButton(
        neural_source_frame,
        values=list(neural_source_labels),
        variable=neural_source_display_var,
        command=_refresh_neural_source_controls,
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
    neural_source_segment.pack(side=tk.LEFT, padx=(10, 0))
    neural_source_segment.set(neural_source_display_var.get())

    neural_row_widgets["Dir"] = add_config_row(
        frame_neural_cache,
        "Dir",
        param_defaults.get("Dir", neural_section_defaults.get("Dir", "")),
        param_entries,
        1,
        frame_color,
        ANALYSIS_LABELS,
    )

    neural_row_widgets["Spks Path"] = add_config_row(
        frame_neural_cache,
        "Spks Path",
        param_defaults.get("Spks Path", neural_section_defaults.get("Spks Path", "None")),
        param_entries,
        2,
        frame_color,
        ANALYSIS_LABELS,
    )

    neural_cache_format_frame = ctk.CTkFrame(frame_neural_cache, fg_color="transparent")
    neural_cache_format_frame.grid(row=3, column=0, columnspan=2, sticky="w", pady=(10, 0))
    ctk.CTkLabel(
        neural_cache_format_frame,
        text="Create as:",
        text_color=muted_text,
    ).pack(side=tk.LEFT)

    neural_cache_format_segment = ctk.CTkSegmentedButton(
        neural_cache_format_frame,
        values=["npy", "zarr"],
        variable=neural_cache_format_var,
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
    neural_cache_format_segment.pack(side=tk.LEFT, padx=(10, 0))
    neural_cache_format_segment.set(neural_cache_format_var.get())

    btn_create_neural_cache = ctk.CTkButton(
        frame_neural_cache,
        text="Create pos/spikes Cache (.npy or .zarr)",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(create_neural_cache, "Neural cache creation"),
    )
    btn_create_neural_cache.grid(row=4, column=0, columnspan=2, pady=(12, 0), sticky="ew")
    _refresh_neural_source_controls()

    # --- Stimulus downsample cache ---
    frame_downsample = ttk.LabelFrame(stage_downsample, text="Stimulus Downsample Cache", padding=15)
    frame_downsample.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_downsample.columnconfigure(1, weight=1)

    add_config_row(
        frame_downsample, "Movie Path", param_defaults.get("Movie Path", ""),
        param_entries, 0, frame_color, ANALYSIS_LABELS,
    )

    downsample_percent_label = ctk.CTkLabel(frame_downsample, text="", text_color=muted_text)

    def _refresh_downsample_controls(_value=None):
        """Refresh downsample-control state and labels."""
        percent = _selected_downsample_percent()
        if _value is not None:
            completed_actions.difference_update({
                "downsample:coarse", "wavelet:coarse_rf", "wavelet:model", "wavelet:full_model", "rf",
            })
        try:
            downsample_slider.configure(state="normal")
            downsample_format_segment.configure(state="normal")
            btn_downsample_video.configure(
                text="Prepare Stimulus Cache"
            )
            downsample_percent_label.configure(text=f"Downsampling percentage: {percent:.0f}%")
            refresh_size_estimates()
        except Exception:
            pass

    downsample_percent_label.grid(row=1, column=0, sticky="w", pady=(8, 2))
    downsample_slider = ctk.CTkSlider(
        frame_downsample,
        from_=0,
        to=100,
        number_of_steps=100,
        variable=downsample_percent_var,
        command=_refresh_downsample_controls,
        height=18,
    )
    downsample_slider.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=(8, 2))

    downsample_format_frame = ctk.CTkFrame(frame_downsample, fg_color="transparent")
    downsample_format_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=(10, 0))
    ctk.CTkLabel(
        downsample_format_frame,
        text="Cache array format:",
        text_color=text_color,
        font=ctk.CTkFont(size=12, weight="bold"),
    ).pack(side=tk.LEFT)
    downsample_format_segment = ctk.CTkSegmentedButton(
        downsample_format_frame,
        values=["npy", "zarr"],
        variable=downsample_format_var,
        command=lambda _value=None: refresh_size_estimates(),
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
    downsample_format_segment.pack(side=tk.LEFT, padx=(10, 0))
    downsample_format_segment.set(downsample_format_var.get())

    btn_downsample_video = ctk.CTkButton(
        frame_downsample,
        text="Prepare Downsampled Video Cache",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(create_downsampled_video_cache, "Stimulus video downsampling"),
    )
    btn_downsample_video.grid(row=3, column=0, columnspan=2, pady=(12, 0), sticky="ew")
    _refresh_downsample_controls()

    # --- Neural & RF analysis ---
    frame_analysis = ttk.LabelFrame(stage_analysis, text="Neural & RF Analysis", padding=15)
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
        text="Run Model (Coarse RF)",
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=run_in_thread(plot_run_model_outputs, "Run Model"),
    )
    btn_run_model_plots.pack(fill=tk.X, pady=(8, 0))

    btn_run_full_model_plots = ctk.CTkButton(
        frame_analysis,
        text="Run Full Model",
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=run_in_thread(plot_run_full_model_outputs, "Run Full Model"),
    )
    btn_run_full_model_plots.pack(fill=tk.X, pady=(8, 0))

    # --- Export ---
    frame_export = ttk.LabelFrame(stage_export, text="Export", padding=15)
    frame_export.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    output_paths_frame = ttk.LabelFrame(frame_export, text="Output Folders", padding=(10, 8))
    output_paths_frame.pack(fill=tk.X, pady=(0, 8))
    output_paths_frame.columnconfigure(1, weight=1)
    output_path_rows = {}
    for row, (key, default) in enumerate((
        ("Full Model Save Path", param_defaults.get("Full Model Save Path", "")),
        ("Plot Cache Path", param_defaults.get("Plot Cache Path", "")),
    )):
        output_path_rows[key] = add_config_row(
            output_paths_frame, key, default, param_entries, row, frame_color, ANALYSIS_LABELS
        )

    export_array_format_var = tk.StringVar(value="npy")
    export_format_wrap = ctk.CTkFrame(frame_export, fg_color="transparent")
    export_format_wrap.pack(fill=tk.X, pady=(0, 8))
    ctk.CTkLabel(export_format_wrap, text="Array format:", text_color=muted_text).pack(side=tk.LEFT)
    export_format_segment = ctk.CTkSegmentedButton(
        export_format_wrap,
        values=["npy", "zarr", "both"],
        variable=export_array_format_var,
        height=26,
        corner_radius=6,
        selected_color=primary_btn,
        selected_hover_color="#1D4ED8",
    )
    export_format_segment.pack(side=tk.LEFT, padx=(10, 0))
    export_format_segment.set("npy")

    ctk.CTkLabel(frame_export, text="All-neuron results", text_color=text_color,
                 font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(4, 2))
    btn_export_all_results = ctk.CTkButton(
        frame_export,
        text="Export Every Displayed Graph",
        fg_color="#2563EB",
        hover_color="#1D4ED8",
        command=export_all_displayed_results,
    )
    btn_export_all_results.pack(fill=tk.X, pady=3)
    btn_export_all_neurons = ctk.CTkButton(
        frame_export,
        text="Export All-Neuron Summary Graphs",
        fg_color="#0891B2",
        hover_color="#0E7490",
        command=export_all_neurons_results,
    )
    btn_export_all_neurons.pack(fill=tk.X, pady=3)
    ctk.CTkLabel(frame_export, text="Individual-neuron results", text_color=text_color,
                 font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", pady=(10, 2))
    btn_export_individual_neuron = ctk.CTkButton(
        frame_export,
        text="Export Current Neuron Graphs",
        fg_color="#7C3AED",
        hover_color="#6D28D9",
        command=export_individual_neuron_results,
    )
    btn_export_individual_neuron.pack(fill=tk.X, pady=3)
    btn_export_all_individual_neurons = ctk.CTkButton(
        frame_export,
        text="Export Graphs for Every Neuron",
        fg_color="#059669",
        hover_color="#047857",
        command=export_all_individual_neurons_results,
    )
    btn_export_all_individual_neurons.pack(fill=tk.X, pady=3)

    # --- Global Controls ---
    frame_controls = ttk.Frame(frame_left, style="TFrame")
    frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    all_buttons = [
        btn_create_neural_cache,
        btn_downsample_video,
        btn_submit_gabor,
        btn_submit_wavelet,
        btn_prepare_model_wavelets,
        btn_prepare_full_model_wavelets,
        btn_submit_plot,
        btn_runRF,
        btn_run_model_plots,
        btn_run_full_model_plots,
        btn_export_all_results,
        btn_export_all_neurons,
        btn_export_individual_neuron,
        btn_export_all_individual_neurons,
        btn_save_state,
        btn_load_state,
    ]

    def _sync_completed_actions_from_artifacts():
        """Unlock stages whose validated artifacts already exist on disk."""
        try:
            movpath = _find_movie_path()
            nx, ny = _stimulus_grid_dimensions("coarse", movpath)
            expected_movie_shape = (_movie_metadata(movpath)["frames"], ny, nx)
            cache_path, _cache_format = _find_compatible_downsample_cache(
                movpath, "coarse", expected_movie_shape, _selected_downsample_format()
            )
            if cache_path:
                completed_actions.add("downsample:coarse")
        except Exception:
            # The movie may not be selected yet; that is the normal initial state.
            pass

        try:
            neural_dir = _folder_from_entry(
                param_entries, "Spks Path", _project_layout().neural_cache_dir
            )
            cache_pair = find_neural_cache_pair(neural_dir)
            if cache_pair is not None:
                spikes, _positions, spikes_path, _positions_path = load_neural_cache_pair(
                    neural_dir, mmap_mode="r"
                )
                if spikes.ndim == 3:
                    movpath = _find_movie_path()
                    expected_frames = _movie_metadata(movpath)["frames"]
                    if spikes.shape[1] == expected_frames:
                        completed_actions.add("neural")
                    else:
                        print(
                            "Existing neural cache was not used for prerequisites: "
                            f"{spikes_path} has {spikes.shape[1]} frames; movie has {expected_frames}."
                        )
        except Exception:
            pass

        try:
            if _selected_wavelet_backend() == "legacy":
                if _find_gabor_library("coarse"):
                    completed_actions.add("gabor:coarse")
                if _find_gabor_library("fine"):
                    completed_actions.add("gabor:full")
            else:
                kernel_dir = _project_layout().gabor_kernel_dir
                if os.path.exists(os.path.join(kernel_dir, "gabor_kernels_coarse_conv.npz")):
                    completed_actions.add("gabor:coarse")
                if os.path.exists(os.path.join(kernel_dir, "gabor_kernels_fine_conv.npz")):
                    completed_actions.add("gabor:full")
        except Exception:
            pass

        try:
            movpath = _find_movie_path()
            frames = _movie_metadata(movpath)["frames"]
            nx, ny = _stimulus_grid_dimensions("coarse", movpath)
            n_orientations = int(gabor_entries["N_thetas"].get())
            n_sigmas = len(parse_literal(gabor_entries["Sigmas"].get(), "Sigmas"))
            coarse_shape = (frames, nx, ny, n_orientations, n_sigmas)
            wavelet_dir = _wavelet_folder("coarse")
            if _artifact_matches(os.path.join(wavelet_dir, "coarse_rf_power.zarr"), coarse_shape):
                completed_actions.add("wavelet:coarse_rf")
            if (
                _artifact_matches(os.path.join(wavelet_dir, "coarse_model_real.zarr"), coarse_shape)
                and _artifact_matches(os.path.join(wavelet_dir, "coarse_model_imag.zarr"), coarse_shape)
            ):
                completed_actions.add("wavelet:model")
            full_sigmas = parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model")
            frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
            full_shape = (frames, nx, ny, n_orientations, len(full_sigmas), max(1, len(frequencies)))
            full_dir = _wavelet_folder("full")
            if (
                _artifact_matches(os.path.join(full_dir, "dwt_videodata2_r.zarr"), full_shape)
                and _artifact_matches(os.path.join(full_dir, "dwt_videodata2_i.zarr"), full_shape)
            ):
                completed_actions.add("wavelet:full_model")
        except Exception:
            pass

    def refresh_action_buttons(advance_from=None):
        """Apply stage prerequisites and advance to the next chronological tab."""
        prerequisites = {
            btn_create_neural_cache: {"downsample:coarse"},
            btn_submit_wavelet: {"downsample:coarse", "gabor:coarse"},
            btn_prepare_model_wavelets: {"downsample:coarse", "gabor:coarse"},
            btn_prepare_full_model_wavelets: {"downsample:coarse", "gabor:full"},
            btn_submit_plot: {"neural", "wavelet:coarse_rf"},
            btn_runRF: {"rf"},
            btn_run_model_plots: {"rf", "wavelet:model"},
            btn_run_full_model_plots: {"rf", "wavelet:full_model"},
            btn_export_all_results: {"rf"},
            btn_export_all_neurons: {"rf"},
            btn_export_individual_neuron: {"rf"},
            btn_export_all_individual_neurons: {"rf"},
        }
        for button, required in prerequisites.items():
            button.configure(state=tk.NORMAL if required.issubset(completed_actions) else tk.DISABLED)
        next_tabs = {
            "Stimulus video downsampling": "2 Session Setup",
            "Neural cache creation": "3 Gabor",
            "Gabor asset preparation": "4 Wavelet Products",
            "Prepare Coarse RF wavelets": "5 Analysis",
            "Coarse receptive-field analysis": "6 Export",
        }
        if advance_from in next_tabs:
            stage_tabs.set(next_tabs[advance_from])

    def _set_grid_row_visible(row_widgets, visible):
        """Show or hide one label/entry pair without discarding its value."""
        for widget in row_widgets:
            widget.grid() if visible else widget.grid_remove()

    def _refresh_scale_field_visibility():
        """Expose the parameters used by the shared-grid workflow."""
        for key, row_widgets in gabor_row_widgets.items():
            if key == "Save Path":
                visible = False
            else:
                visible = True
            _set_grid_row_visible(row_widgets, visible)
        _set_grid_row_visible(coarse_wavelet_path_row, True)
        _set_grid_row_visible(full_wavelet_path_row, True)
        _set_grid_row_visible(full_sigmas_row, True)
        _set_grid_row_visible(output_path_rows["Full Model Save Path"], True)
        # The selector must remain visible whenever decomposition is available.
        format_frame_wavelet.pack(anchor="w", pady=(10, 0), before=wavelet_size_label)

    refresh_scale_controls()
    _sync_completed_actions_from_artifacts()
    refresh_action_buttons()

    for section in (frame_session, stage_tabs, frame_controls):
        section.pack_forget()

    frame_session.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    stage_tabs.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    try:
        root.mainloop()
    finally:
        keep_awake.stop()
