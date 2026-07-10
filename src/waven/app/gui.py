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
from ..storage.neural_cache import find_neural_cache_pair, load_neural_cache_pair
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
    global waveletDecompositionConv, waveletDecompositionFullConv
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
    global repetability_trial3, orientation_selectivity_from_tuning, selectivity_for_rfs
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
        orientation_selectivity_from_tuning as _orientation_selectivity_from_tuning,
        selectivity_for_rfs as _selectivity_for_rfs,
    )

    compute_skewness_neurons = _compute_skewness_neurons
    PearsonCorrelationPinkNoise = _PearsonCorrelationPinkNoise
    PlotTuningCurve = _PlotTuningCurve
    repetability_trial3 = _repetability_trial3
    orientation_selectivity_from_tuning = _orientation_selectivity_from_tuning
    selectivity_for_rfs = _selectivity_for_rfs
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


def run(param_defaults, gabor_param, workflow=None):
    """Function for run.

    Args:
        param_defaults: Input value for this operation.
        gabor_param: Input value for this operation.
        workflow: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if workflow not in (WORKFLOW_2P, WORKFLOW_EPHYS):
        workflow = WORKFLOW_2P
    gabor_param = _normalise_gabor_params(gabor_param)

    GABOR_LABELS = {
        "N_thetas": "Orientation Count",
        "Sigmas": "Filter Sizes (px)",
        "Frequencies": "Spatial Frequencies (cyc/px)",
        "Phases": "Phases (degrees)",
        "NX": "Stimulus Width (px)",
        "NY": "Stimulus Height (px)",
        "Save Path": "Legacy/Default Library Output",
        "Coarse Library Path": "Coarse Library Output",
        "Fine Library Path": "Fine Library Output",
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
        "Coarse Library Path": "Coarse Gabor Library Path",
        "Fine Library Path": "Fine Gabor Library Path",
        "Spks Path": "Pre-aligned Spikes (.npy/.zarr, optional)",
        "Full Model Wavelet Path": "Full-Model Wavelet Store",
        "Full Model Save Path": "Full-Model Results Directory",
        "Plot Cache Path": "Plot Cache File",
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
        merged = dict(DEFAULT_COMMON_PARAMS)
        if value == WORKFLOW_2P:
            merged.update(DEFAULT_TWO_PHOTON_PARAMS)
        else:
            merged.update(DEFAULT_EPHYS_PARAMS)
        merged.update(param_defaults)
        return keys, {key: merged.get(key, "") for key in keys}

    FIELD_LABELS = {**GABOR_LABELS, **ANALYSIS_LABELS}

    BROWSE_KIND = {
        "Save Path": "savefile",
        "Movie Path": "file",
        "Library Path": "file",
        "Coarse Library Path": "file",
        "Fine Library Path": "file",
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

        def write(self, string):
            """Function for write.

            Args:
                string: Input value for this operation.
            """
            if not string:
                return
            with self._lock:
                self._buffer.append(string)
                should_schedule = not self._flush_pending
                self._flush_pending = True
            if should_schedule:
                try:
                    self.widget.after(self.flush_ms, self._flush)
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
            try:
                self.widget.after(0, self._flush)
            except Exception:
                pass

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
            f"Task {status} at {metrics['completed_at']} | "
            f"elapsed {format_duration(metrics['elapsed'])} | "
            f"CPU {metrics['cpu_percent']:.1f}% ({metrics['cpu_seconds']:.1f}s) | "
            f"peak RAM {_format_bytes(metrics['peak_ram'])} | "
            f"disk read {_format_bytes(metrics['disk_read'])}, write {_format_bytes(metrics['disk_write'])} | "
            f"GPU peak allocated {_format_bytes(metrics['gpu_allocated'])}, reserved {_format_bytes(metrics['gpu_reserved'])} | "
            f"network sent {_format_bytes(metrics['net_sent'])}, received {_format_bytes(metrics['net_recv'])}"
        )

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
        print(f"\n--- {task_name} ---")

    def end_task(success=False, cancelled=False, metrics=None):
        """Function for end task.

        Args:
            success: Input value for this operation.
            cancelled: Input value for this operation.
            metrics: Input value for this operation.
        """
        if task_state["name"]:
            progress_bar.stop()
            progress_bar.configure(mode="determinate", value=0 if cancelled else 100)
            status_label = "cancelled" if cancelled else ("finished" if success else "failed")
            print(_format_task_summary(metrics, status_label))
            print(f"--- {status_label.title()}: {task_state['name']} ---\n")
            flash_taskbar()
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
                    print(f"\n[!] TASK CANCELLED: {label}")
                    removed = _remove_cancelled_task_paths()
                    print(f"Cancelled task cleanup removed {removed} partial path(s).")
                except Exception as exc:
                    # Print a clear header for the error
                    print(f"\n[!] AN ERROR OCCURRED IN TASK: {label}")
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
    embedded_canvases = []
    figure_export_records = []
    active_recovery_dir = {"path": None}

    def _sem_enabled():
        """Function for sem enabled.

        Returns:
            Result produced by the operation.
        """
        try:
            return bool(show_sem_var.get())
        except Exception:
            return False

    def _sem_over_trials(trials):
        """Function for sem over trials.

        Args:
            trials: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        trials = np.asarray(trials, dtype=float)
        if trials.ndim < 2 or trials.shape[0] <= 1:
            return None
        return np.nanstd(trials, axis=0, ddof=1) / np.sqrt(trials.shape[0])

    def _plot_trace_with_optional_sem(ax, trials, x=None, color='k', label=None):
        """Function for plot trace with optional sem.

        Args:
            ax: Input value for this operation.
            trials: Input value for this operation.
            x: Input value for this operation.
            color: Input value for this operation.
            label: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        trials = np.asarray(trials, dtype=float)
        if trials.ndim == 1:
            mean = trials
            n_trials = 1
        else:
            mean = np.nanmean(trials, axis=0)
            n_trials = trials.shape[0]
        if x is None:
            x = np.arange(mean.shape[0])
        ax.plot(x, mean, c=color, label=label)
        sem = _sem_over_trials(trials)
        if _sem_enabled() and sem is not None:
            every = max(1, mean.shape[0] // 100)
            ax.errorbar(
                x,
                mean,
                yerr=sem,
                fmt='none',
                ecolor=color,
                elinewidth=0.6,
                capsize=1,
                alpha=0.55,
                errorevery=every,
            )
            return n_trials
        return None

    def _set_sem_caption(fig, n_trials):
        """Function for set sem caption.

        Args:
            fig: Input value for this operation.
            n_trials: Input value for this operation.
        """
        if n_trials and n_trials > 1:
            fig._waven_caption = f"Error bars represent SEM over {n_trials} trials."
        elif hasattr(fig, "_waven_caption"):
            delattr(fig, "_waven_caption")

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

    def _plot_cache_path():
        """Function for plot cache path.

        Returns:
            Result produced by the operation.
        """
        value = _field_value(param_entries, "Plot Cache Path", "").strip()
        if value.lower() in ("", "none", "null"):
            save_dir = _field_value(param_entries, "Full Model Save Path", "").strip()
            if save_dir.lower() in ("", "none", "null"):
                movie_path = _field_value(param_entries, "Movie Path", "").strip()
                save_dir = os.path.dirname(movie_path) or "."
            value = os.path.join(save_dir, "plot_cache.pkl.gz")
        return value

    def _recovery_root():
        """Function for recovery root.

        Returns:
            Result produced by the operation.
        """
        value = _field_value(param_entries, "Recovery Cache Directory", "").strip()
        if value.lower() in ("", "none", "null"):
            save_dir = _field_value(param_entries, "Full Model Save Path", "").strip()
            if save_dir.lower() in ("", "none", "null"):
                movie_path = _field_value(param_entries, "Movie Path", "").strip()
                save_dir = os.path.dirname(movie_path) or "."
            value = os.path.join(save_dir, "recovery_cache")
        return value

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
            "analysis_scale": _selected_analysis_scale(),
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

    def _plot_selectivity_population(selectivity, filter_mask, neuron_pos=None):
        """Function for plot selectivity population.

        Args:
            selectivity: Input value for this operation.
            filter_mask: Input value for this operation.
            neuron_pos: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        bins = np.linspace(0, 1, 21)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), constrained_layout=True)
        specs = [("osi", "OSI"), ("gosi", "gOSI")]
        shank_groups = None
        shank_labels = []
        shank_source = "not available"
        if neuron_pos is not None:
            shank_groups, shank_labels, shank_source = _grouping_for_selectivity(neuron_pos, preferred="shank")
        colors = ["#059669", "#D97706", "#7C3AED", "#DC2626", "#0891B2", "#BE185D", "#4B5563", "#65A30D"]
        for ax, (key, label) in zip(axes, specs):
            values, valid = _metric_values(selectivity, key)
            filtered, _ = _metric_values(selectivity, key, filter_mask=filter_mask)
            ax.hist(values, bins=bins, color="#2563EB", alpha=0.72, label=f"All neurons (n={values.size})")
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
        fig.suptitle("Orientation Selectivity by Neuron")
        fig._waven_caption = f"Shank grouping source: {shank_source}."
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

    def _plot_selectivity_by_unit(selectivity, neuron_pos, filter_mask):
        """Function for plot selectivity by unit.

        Args:
            selectivity: Input value for this operation.
            neuron_pos: Input value for this operation.
            filter_mask: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        unit_groups, unit_labels, unit_source = _grouping_for_selectivity(neuron_pos, preferred="unit")
        shank_groups, shank_labels, shank_source = _grouping_for_selectivity(neuron_pos, preferred="shank")
        bins = np.linspace(0, 1, 16)
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
        path = gabor_entries.get(f"{kind.title()} Library Path")
        if path is not None and path.get().strip():
            return path.get().strip()
        root, ext = os.path.splitext(base_path)
        return f"{root}_{kind}{ext or '.npy'}"

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

    def _library_artifact_path(path_save):
        """Function for library artifact path.

        Args:
            path_save: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        if gabor_format_var.get() == "zarr":
            return os.path.splitext(path_save)[0] + ".zarr"
        return path_save

    def _convolution_kernel_cache_output_path(kind):
        """Return where the compact convolution kernel cache should be stored."""
        base_path = gabor_entries["Save Path"].get().strip()
        library_path = _library_output_path(kind, base_path)
        folder_path = os.path.dirname(library_path or base_path) or "."
        return convolution_kernel_cache_path(folder_path, kind)

    def _ensure_convolution_kernel_cache(kind, force=False):
        """Build or reuse the compact convolution kernel cache for ``kind``."""
        _ensure_wavelet_imports("convolution kernel cache construction")
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        phase_offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
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
        """Return the currently selected analysis scale."""
        try:
            value = analysis_scale_var.get()
        except NameError:
            return "coarse"
        return value if value in {"coarse", "full"} else "coarse"

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
        full_nx = int(gabor_entries["NX"].get())
        full_ny = int(gabor_entries["NY"].get())
        n_theta = int(gabor_entries["N_thetas"].get())
        offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
        path_save = gabor_entries["Save Path"].get()
        kind = kind.lower()
        if kind not in {"coarse", "fine"}:
            raise ValueError(f"Unknown Gabor library kind: {kind}")
        if kind == "coarse":
            nx, ny = coarse_grid_dimensions(full_nx, full_ny)
            path_save = _library_output_path("coarse", path_save)
            description = f"coarse coupled Gabor library ({nx} x {ny})"
        else:
            nx, ny = full_nx, full_ny
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
        if _artifact_matches(output_path, expected_shape):
            print(f"Resume: found completed {description}, reusing {output_path}")
            _write_recovery_step(f"{kind}_gabor_reused", path=output_path, shape=expected_shape)
            entry_key = "Coarse Library Path" if kind == "coarse" else "Fine Library Path"
            if entry_key in gabor_entries:
                gabor_entries[entry_key].delete(0, tk.END)
                gabor_entries[entry_key].insert(0, output_path)
            if kind == "fine" and "Library Path" in param_entries:
                param_entries["Library Path"].delete(0, tk.END)
                param_entries["Library Path"].insert(0, output_path)
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
        entry_key = "Coarse Library Path" if kind == "coarse" else "Fine Library Path"
        if entry_key in gabor_entries:
            gabor_entries[entry_key].delete(0, tk.END)
            gabor_entries[entry_key].insert(0, output_path)
        if kind == "fine" and "Library Path" in param_entries:
            param_entries["Library Path"].delete(0, tk.END)
            param_entries["Library Path"].insert(0, output_path)
            refresh_size_estimates()
        update_progress(100, f"{description} complete")

    def create_both_gabor_libraries():
        """Function for create both gabor libraries."""
        create_gabor("coarse")
        create_gabor("fine")

    def run_wavelet(scale=None):
        """Function for run wavelet.

        Returns:
            Result produced by the operation.
        """
        _ensure_wavelet_imports("stimulus wavelet generation")
        _raise_if_cancelled()
        scale = scale or _selected_analysis_scale()
        backend = _selected_wavelet_backend()
        if scale not in {"coarse", "full"}:
            raise ValueError(f"Unknown wavelet decomposition scale: {scale}")
        if backend not in {"legacy", "convolution"}:
            raise ValueError(f"Unknown wavelet decomposition backend: {backend}")
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
        phase_offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
        fine_library_sigmas = _ordered_float_union(
            sigmas,
            parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"),
        )
        legacy_lib_path = param_entries["Library Path"].get()
        coarse_lib_path = gabor_entries.get("Coarse Library Path")
        fine_lib_path = gabor_entries.get("Fine Library Path")
        coarse_lib_path = coarse_lib_path.get().strip() if coarse_lib_path else ""
        fine_lib_path = fine_lib_path.get().strip() if fine_lib_path else ""
        if not coarse_lib_path:
            coarse_lib_path = legacy_lib_path
        if not fine_lib_path:
            fine_lib_path = legacy_lib_path
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
        nx = int(param_entries["NX"].get())
        ny = int(param_entries["NY"].get())
        n_thetas = int(gabor_entries["N_thetas"].get())
        coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
        fallback_frames = int(param_entries["Number of Frames"].get())
        expected_frames = _movie_frame_count(movpath, fallback_frames)
        full_downsample_path = movpath[:-4] + '_downsampled.npy'
        coarse_downsample_path = movpath[:-4] + '_coarse_downsampled.npy'

        visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
        analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")

        def _coverage_ratios():
            """Function for coverage ratios.

            Returns:
                Result produced by the operation.
            """
            if visual_coverage != analysis_coverage:
                visual_coverage_arr = np.array(visual_coverage)
                analysis_coverage_arr = np.array(analysis_coverage)
                ratio_x = 1 - ((visual_coverage_arr[0] - visual_coverage_arr[1]) - (analysis_coverage_arr[0] - analysis_coverage_arr[1])) / (visual_coverage_arr[0] - visual_coverage_arr[1])
                ratio_y = 1 - ((visual_coverage_arr[2] - visual_coverage_arr[3]) - (analysis_coverage_arr[2] - analysis_coverage_arr[3])) / (visual_coverage_arr[2] - visual_coverage_arr[3])
            else:
                ratio_x = ratio_y = 1
            return ratio_x, ratio_y

        coarse_cache_path = os.path.join(wavelet_folder, "dwt_downsampled_videodata.npy")
        real_phase_path = os.path.join(wavelet_folder, "dwt_videodata_0.npy")
        imag_phase_path = os.path.join(wavelet_folder, "dwt_videodata_1.npy")
        coarse_phase_shape = (
            expected_frames,
            coarse_nx,
            coarse_ny,
            n_thetas,
            len(sigmas),
        )
        coarse_cache_shape = (3,) + coarse_phase_shape
        if scale == "coarse":
            current_wavelet_dir[0] = wavelet_folder
            coarse_cache_ready = _artifact_matches(coarse_cache_path, coarse_cache_shape)

            if coarse_cache_ready:
                update_progress(80, "Coarse wavelet decomposition", "Reusing coarse RF cache")
                print(f"Resume: found completed coarse RF wavelet cache, reusing {coarse_cache_path}")
                _write_recovery_step("coarse_cache_reused", path=coarse_cache_path, shape=coarse_cache_shape)
            else:
                update_progress(8, "Coarse wavelet decomposition", "Preparing coarse stimulus movie")
                print("Step 1/4: Preparing coarse stimulus movie...")
                if not os.path.exists(coarse_downsample_path):
                    _register_cancel_cleanup_path(coarse_downsample_path)
                if _artifact_matches(coarse_downsample_path, (expected_frames, coarse_ny, coarse_nx)):
                    print(f"Resume: found completed coarse downsampled movie, reusing {coarse_downsample_path}")
                    _write_recovery_step("coarse_downsampling_reused", path=coarse_downsample_path)
                else:
                    _write_recovery_step("wavelet_downsampling_started", wavelet_folder=wavelet_folder)
                    ratio_x, ratio_y = _coverage_ratios()
                    downsample_video_binary(
                        movpath,
                        visual_coverage,
                        analysis_coverage,
                        shape=(coarse_ny, coarse_nx),
                        chunk_size=video_downsample_chunk_size(),
                        ratios=(ratio_x, ratio_y),
                        save_path=coarse_downsample_path,
                        cancel_event=_current_cancel_event(),
                    )
                if not _artifact_matches(coarse_downsample_path, (expected_frames, coarse_ny, coarse_nx)):
                    raise FileNotFoundError(
                        "Coarse downsampled stimulus was not created. "
                        f"Check that Movie Path points to a readable video: {movpath}"
                    )
                _raise_if_cancelled()
                videodata = np.load(coarse_downsample_path, mmap_mode="r")
                videodata = videodata.astype(int) - np.logical_not(videodata).astype(int)

                update_progress(25, "Coarse wavelet decomposition", "Preparing coarse real phase")
                print("Step 2/4: Preparing coarse real phase wavelets...")
                if _artifact_matches(real_phase_path, coarse_phase_shape):
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
                            cancel_event=_current_cancel_event(),
                        )
                    else:
                        waveletDecomposition(
                            videodata,
                            0,
                            sigmas,
                            wavelet_folder,
                            coarse_lib_path,
                            cancel_event=_current_cancel_event(),
                        )
                    _write_recovery_step("coarse_phase_real_complete", path=real_phase_path)

                update_progress(45, "Coarse wavelet decomposition", "Preparing coarse imaginary phase")
                print("Step 3/4: Preparing coarse imaginary phase wavelets...")
                if _artifact_matches(imag_phase_path, coarse_phase_shape):
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
                            cancel_event=_current_cancel_event(),
                        )
                    else:
                        waveletDecomposition(
                            videodata,
                            1,
                            sigmas,
                            wavelet_folder,
                            coarse_lib_path,
                            cancel_event=_current_cancel_event(),
                        )
                    _write_recovery_step("coarse_phase_imaginary_complete", path=imag_phase_path)

                update_progress(68, "Coarse wavelet decomposition", "Generating coarse RF cache")
                print("Step 4/4: Generating coarse wavelet cache for RF analysis...")
                if not os.path.exists(coarse_cache_path):
                    _register_cancel_cleanup_path(coarse_cache_path)
                for scratch_name in ("dwt_r_downsampled.mmap", "dwt_i_downsampled.mmap", "dwt_c_downsampled.mmap"):
                    scratch_path = os.path.join(wavelet_folder, scratch_name)
                    if not os.path.exists(scratch_path):
                        _register_cancel_cleanup_path(scratch_path)
                coarseWavelet(
                    wavelet_folder,
                    False,
                    nx0=coarse_nx,
                    ny0=coarse_ny,
                    no=n_thetas,
                    ns=len(sigmas),
                    nf=1,
                    nx=coarse_nx,
                    ny=coarse_ny,
                    chunk_size=None,
                    cancel_event=_current_cancel_event(),
                )
                _raise_if_cancelled()
                if not _artifact_matches(coarse_cache_path, coarse_cache_shape):
                    raise ValueError(f"Coarse cache was written with an unexpected shape: {coarse_cache_path}")
                _write_recovery_step("coarse_cache_complete", path=coarse_cache_path)
                for intermediate_path in (real_phase_path, imag_phase_path):
                    try:
                        if os.path.exists(intermediate_path):
                            os.remove(intermediate_path)
                            print(f"Removed intermediate coarse phase file: {intermediate_path}")
                    except Exception as exc:
                        print(f"Could not remove intermediate file {intermediate_path}: {exc}")

                for scratch_name in ("dwt_r_downsampled.mmap", "dwt_i_downsampled.mmap", "dwt_c_downsampled.mmap"):
                    scratch_path = os.path.join(wavelet_folder, scratch_name)
                    try:
                        if os.path.exists(scratch_path):
                            os.remove(scratch_path)
                            print(f"Removed temporary coarse cache scratch file: {scratch_path}")
                    except Exception as exc:
                        print(f"Could not remove temporary scratch file {scratch_path}: {exc}")

            print(
                f"Coarse RF grid derived from config: {coarse_nx} x {coarse_ny} "
                f"(20% of {nx} x {ny})"
            )
            print(f"Coarse wavelet files are ready: {wavelet_folder}")
            update_progress(100, "Coarse wavelet decomposition", "Coarse RF cache ready")
            return True

        if scale == "full":
            update_progress(5, "Full wavelet decomposition", "Preparing full-model outputs")

        full_output_target = param_entries["Full Model Wavelet Path"].get().strip()
        if not full_output_target:
            full_output_target = parent_dir
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
        if _artifact_matches(full_downsample_path, (expected_frames, ny, nx)):
            print(f"Resume: found completed full-resolution downsampled movie, reusing {full_downsample_path}")
            _write_recovery_step("full_downsampling_reused", path=full_downsample_path)
        else:
            _register_cancel_cleanup_path(full_downsample_path)
            ratio_x, ratio_y = _coverage_ratios()
            downsample_video_binary(
                movpath,
                visual_coverage,
                analysis_coverage,
                shape=(ny, nx),
                chunk_size=video_downsample_chunk_size(),
                ratios=(ratio_x, ratio_y),
                save_path=full_downsample_path,
                cancel_event=_current_cancel_event(),
            )
        if not _artifact_matches(full_downsample_path, (expected_frames, ny, nx)):
            raise FileNotFoundError(
                "Full-resolution downsampled stimulus was not created. "
                f"Check that Movie Path points to a readable video: {movpath}"
            )
        _raise_if_cancelled()
        videodata = np.load(full_downsample_path, mmap_mode="r")
        videodata = videodata.astype(int) - np.logical_not(videodata).astype(int)
        full_model_shape = (
            videodata.shape[0],
            nx,
            ny,
            n_thetas,
            len(sigmas_full),
            max(1, len(frequencies)),
        )
        zarr_chunks = (
            max(1, int(param_entries["Hz"].get()) * 60),
            1,
            1,
            n_thetas,
            len(sigmas_full),
            max(1, len(frequencies)),
        )
        for phase in (0, 1):
            suffix = "_r" if phase == 0 else "_i"
            target_ext = ".zarr" if is_zarr_wavelet else ".npy"
            target = os.path.join(full_output, f"dwt_videodata2{suffix}{target_ext}")
            if _artifact_matches(target, full_model_shape):
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
        exp_info = parse_literal(param_entries["Experiment Info"].get(), "Experiment Info")
        block_end = int(param_entries["Block End"].get())
        nb_frames = int(param_entries["Number of Frames"].get())
        pathdata = Path(data_dirs[0]) / exp_info[0] / exp_info[1] / str(exp_info[2])
        pathsuite2p = pathdata / "suite2p"
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
            "block_end": block_end,
            "n_planes": n_planes,
            "nb_frames": nb_frames,
            "resolution": resolution,
            "sampling_rate": sampling_rate,
        }

    def create_neural_cache():
        """Create or validate the aligned ``pos`` and ``spikes`` cache pair."""
        context = _neural_alignment_context()
        spks_text = param_entries["Spks Path"].get().strip()
        if spks_text.lower() not in ("", "none", "null"):
            spks_path = Path(spks_text)
            update_progress(10, "Neural cache", "Validating existing aligned cache")
            spks, neuron_pos, loaded_spks_path, loaded_pos_path = load_neural_cache_pair(
                spks_path.parent,
                spks_path,
                mmap_mode=None,
            )
            print(f"Validated aligned spikes cache: {loaded_spks_path} {tuple(spks.shape)}")
            print(f"Validated neuron position cache: {loaded_pos_path} {tuple(np.asarray(neuron_pos).shape)}")
            param_entries["Spks Path"].delete(0, tk.END)
            param_entries["Spks Path"].insert(0, str(loaded_spks_path))
            update_progress(100, "Neural cache", "Existing cache ready")
            return True

        cache_pair = find_neural_cache_pair(context["pathdata"])
        if cache_pair is not None:
            update_progress(20, "Neural cache", "Loading existing aligned cache")
            spks, neuron_pos, loaded_spks_path, loaded_pos_path = load_neural_cache_pair(context["pathdata"])
            print(f"Resume: found aligned spikes cache: {loaded_spks_path} {tuple(spks.shape)}")
            print(f"Resume: found neuron position cache: {loaded_pos_path} {tuple(np.asarray(neuron_pos).shape)}")
            param_entries["Spks Path"].delete(0, tk.END)
            param_entries["Spks Path"].insert(0, str(loaded_spks_path))
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
            threshold=1.25,
            method="frame2ttl",
            save_dir=context["pathdata"],
            output_format=_selected_neural_cache_format(),
        )
        cache_pair = find_neural_cache_pair(context["pathdata"])
        if cache_pair is None:
            raise FileNotFoundError(
                f"Alignment finished but no spikes/pos cache pair was found in {context['pathdata']}."
            )
        saved_spks_path, saved_pos_path = cache_pair
        print(f"Created aligned spikes cache: {saved_spks_path} {tuple(aligned.spikes.shape)}")
        print(f"Created neuron position cache: {saved_pos_path} {tuple(np.asarray(aligned.neuron_pos).shape)}")
        param_entries["Spks Path"].delete(0, tk.END)
        param_entries["Spks Path"].insert(0, str(saved_spks_path))
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
            "show_sem_errorbars": _sem_enabled(),
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
            cache_pair = find_neural_cache_pair(Path(pathdata))
            if cache_pair is not None:
                update_progress(10, "Coarse receptive-field analysis", "Loading aligned neural cache")
                spks, neuron_pos, saved_spks_path, saved_pos_path = load_neural_cache_pair(Path(pathdata))
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
                        threshold=1.25,
                        method='frame2ttl',
                        save_dir=Path(pathdata),
                        output_format=_selected_neural_cache_format(),
                    )
                except NotImplementedError as exc:
                    print(exc)
                    return False
                spks = aligned.spikes
                neuron_pos = aligned.neuron_pos
                cache_pair = find_neural_cache_pair(Path(pathdata))
                saved_spks_path = cache_pair[0] if cache_pair else None
                if saved_spks_path is not None:
                    print(f"Saved aligned neural cache as {_selected_neural_cache_format().upper()}: {saved_spks_path}")
            if saved_spks_path is not None and saved_spks_path.exists():
                param_entries["Spks Path"].delete(0, tk.END)
                param_entries["Spks Path"].insert(0, str(saved_spks_path))
            if workflow == WORKFLOW_2P:
                neuron_pos = np.asarray(neuron_pos)
                neuron_pos[:, 1] = abs(neuron_pos[:, 1] - np.max(neuron_pos[:, 1]))
        else:
            try:
                update_progress(10, "Coarse receptive-field analysis", "Loading pre-aligned spikes")
                spks, neuron_pos, loaded_spks_path, loaded_pos_path = load_neural_cache_pair(
                    Path(spks_path).parent,
                    Path(spks_path),
                    mmap_mode=None,
                )
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
        update_progress(75, "Coarse receptive-field analysis", "Computing OSI/gOSI")
        orientation_selectivity = selectivity_for_rfs(rfs_gabor)
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

            def draw_individual_neuron(neuron_id):
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
                    sem_trials = _plot_trace_with_optional_sem(
                        ax2,
                        trial_spikes,
                        label=f"Neuron {neuron_id} trial-averaged activity",
                    )
                    _set_sem_caption(fig2, sem_trials)
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
                    neuron_osi, neuron_gosi = orientation_selectivity_from_tuning(ori_tun)
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
                    ax3[3].plot(ori_tun, 'o-', c='k')
                    ax3[3].set_title(f'Orientation (OSI {neuron_osi:.3f}, gOSI {neuron_gosi:.3f})')
                    n_ori = rfs_gabor[0].shape[3]
                    ax3[3].set_xticks(
                        [0, max(1, n_ori // 2), max(2, n_ori - 1)],
                        [0, 90, 180],
                    )
                    ax3[3].set_xlabel("Orientation (deg)")
                    ax3[3].set_ylabel("Correlation (a.u.)")
                    ax3[4].plot(s_tuning, 'o-', c='k')
                    ax3[4].set_title('Size (deg)')
                    ax3[4].set_xticks([0, len(sigmas) - 1], [sigmas_deg[0], sigmas_deg[-1]])
                    ax3[4].set_xlabel("Size (deg)")
                    ax3[4].set_ylabel("Correlation (a.u.)")
                    ax3[5].plot(f_tuning, 'o-', c='k')
                    ax3[5].set_title('Spatial Frequency')
                    ax3[5].set_xticks(range(len(frequencies)), [round(f, 3) for f in frequencies])
                    ax3[5].set_xlabel("Spatial frequency (cycles/deg)")
                    ax3[5].set_ylabel("Correlation (a.u.)")
                    _set_figure_export_payload(
                        fig3,
                        {
                            "source": "Inspect Single Neuron",
                            "neuron_id": neuron_id,
                            "rf2d": rf2d,
                            "x_tuning": x_tuning,
                            "y_tuning": y_tuning,
                            "orientation_tuning": ori_tun,
                            "osi": neuron_osi,
                            "gosi": neuron_gosi,
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
            )
            fig_osi_units = _plot_selectivity_by_unit(
                orientation_selectivity,
                neuron_pos,
                filter_mask,
            )
            embed_interactive_figure(fig1, frame_plot_all, title="Neuron Layout")
            embed_interactive_figure(fig10, frame_plot_all, title="Population Retinotopy Maps")
            embed_interactive_figure(fig_osi_population, frame_plot_all, title="OSI and gOSI by Neuron/Shank")
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
            _put_cached_entry(
                "coarse_rf",
                {
                    "analysis_state": _state_for_plot_cache(analysis_state),
                    "figures": _figure_records(
                        [
                            ("all", "Neuron Layout", fig1),
                            ("all", "Population Retinotopy Maps", fig10),
                            ("all", "OSI and gOSI by Neuron/Shank", fig_osi_population),
                            ("all", "OSI and gOSI by Unit", fig_osi_units),
                            ("individual", "Spike Train", fig2),
                            ("individual", "Selected Neuron Tuning", fig3),
                        ]
                    ),
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
        if "spks" in analysis_state and not (0 <= neuron_id < analysis_state["spks"].shape[2]):
            raise ValueError(f"Neuron ID {neuron_id} is outside the loaded range.")
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
            "show_sem_errorbars": _sem_enabled(),
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
                plotting=True,
                frames_per_minute=frames_per_minute,
                show_sem_errorbars=_sem_enabled(),
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
            "show_sem_errorbars": _sem_enabled(),
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
        sigmas_full = np.array(parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"))
        frequencies = np.array(parse_literal(gabor_entries["Frequencies"].get(), "Frequencies"))
        wavelet_path = param_entries["Full Model Wavelet Path"].get().strip() or state["wavelet_dir"]
        save_path = param_entries["Full Model Save Path"].get().strip() or state["wavelet_dir"]
        os.makedirs(save_path, exist_ok=True)
        frames_per_minute = int(param_entries["Hz"].get()) * 60

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
                tt=[0, min(state["nb_frames"], state["spks"].shape[1])],
                memmapping=True,
                train_idx=split_settings["train_idx"],
                test_idx=split_settings["test_idx"],
                double_wavelet_model=False,
                lastmin=split_settings["lastmin"],
                plotting=True,
                frames_per_minute=frames_per_minute,
                hz=int(param_entries["Hz"].get()),
                show_sem_errorbars=_sem_enabled(),
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
        state = {
            "workflow": workflow,
            "analysis_scale": _selected_analysis_scale(),
            "wavelet_backend": _selected_wavelet_backend(),
            "gabor": {key: entry.get() for key, entry in gabor_entries.items()},
            "analysis": {key: entry.get() for key, entry in param_entries.items()},
            "save_options": {
                "gabor_format": gabor_format_var.get(),
                "wavelet_format": wavelet_format_var.get(),
                "neural_cache_format": _selected_neural_cache_format(),
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
        """Function for load app state."""
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
            loaded_workflow = state.get("workflow")
            if loaded_workflow in (WORKFLOW_2P, WORKFLOW_EPHYS) and loaded_workflow != workflow:
                workflow_var.set(loaded_workflow)
                set_workflow_from_panel(loaded_workflow)
            loaded_scale = state.get("analysis_scale")
            if loaded_scale in {"coarse", "full"}:
                analysis_scale_var.set(loaded_scale)
                set_analysis_scale_from_panel(loaded_scale)
            loaded_backend = state.get("wavelet_backend")
            if loaded_backend in {"legacy", "convolution"}:
                wavelet_backend_var.set(loaded_backend)
                set_wavelet_backend_from_panel(loaded_backend)
            for key, value in state.get("gabor", {}).items():
                if key in gabor_entries:
                    gabor_entries[key].delete(0, tk.END)
                    gabor_entries[key].insert(0, str(value))
            analysis_values = state.get("analysis", {})
            render_parameter_fields(preserve_values=True, loaded_values=analysis_values)
            for key, value in analysis_values.items():
                if key in param_entries:
                    param_entries[key].delete(0, tk.END)
                    param_entries[key].insert(0, str(value))
            save_options = state.get("save_options", {})
            gabor_format_var.set(save_options.get("gabor_format", gabor_format_var.get()))
            wavelet_format_var.set(save_options.get("wavelet_format", wavelet_format_var.get()))
            neural_cache_format = save_options.get("neural_cache_format")
            if neural_cache_format in {"npy", "zarr"}:
                neural_cache_format_var.set(neural_cache_format)
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
            nx = int(gabor_entries["NX"].get())
            ny = int(gabor_entries["NY"].get())
            n_theta = int(gabor_entries["N_thetas"].get())
            sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
            fine_sigmas = _ordered_float_union(
                sigmas,
                parse_literal(param_entries["Sigmas Full Model"].get(), "Sigmas Full Model"),
            )
            offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
            frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
            coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
            coarse_bytes, coarse_shape = _gabor_library_npy_bytes(
                coarse_nx,
                coarse_ny,
                n_theta,
                sigmas,
                offsets,
                [0],
            )
            fine_bytes, fine_shape = _gabor_library_npy_bytes(
                nx,
                ny,
                n_theta,
                fine_sigmas,
                offsets,
                frequencies,
            )
            scale = _selected_analysis_scale()
            if gabor_format_var.get() == "zarr":
                kind = "fine" if scale == "full" else "coarse"
                exact = _folder_size_bytes(_zarr_output_path(_library_output_path(kind, gabor_entries["Save Path"].get())))
                expected_bytes = fine_bytes if scale == "full" else coarse_bytes
                expected_shape = fine_shape if scale == "full" else coarse_shape
                if exact is not None:
                    gabor_size_label.configure(
                        text=f"{_scale_label(scale)} Gabor Zarr current size: {_format_bytes(exact)}"
                    )
                else:
                    gabor_size_label.configure(
                        text=(
                            f"{_scale_label(scale)} Gabor Zarr size: compression-dependent; "
                            f"NPY equivalent {_format_bytes(expected_bytes)} {expected_shape}"
                        )
                    )
            else:
                expected_bytes = fine_bytes if scale == "full" else coarse_bytes
                expected_shape = fine_shape if scale == "full" else coarse_shape
                gabor_size_label.configure(
                    text=(
                        f"{_scale_label(scale)} Gabor NPY size: "
                        f"{_format_bytes(expected_bytes)} {expected_shape}"
                    )
                )
        except Exception:
            gabor_size_label.configure(text="Gabor disk size: enter valid dimensions to calculate")

    def estimate_wavelet_size():
        """Function for estimate wavelet size."""
        try:
            n_frames_entry = param_entries.get("Number of Frames")
            fallback_frames = int(n_frames_entry.get()) if n_frames_entry is not None else 0
            movie_entry = param_entries.get("Movie Path")
            movie_path = movie_entry.get().strip() if movie_entry is not None else ""
            n_frames = _movie_frame_count(movie_path, fallback_frames)
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
            n_frequencies = max(1, len(frequencies))
            coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
            bytes_per_float = np.dtype(np.float32).itemsize
            coarse_phase_bytes = 2 * n_frames * coarse_nx * coarse_ny * n_thetas * n_sigmas * bytes_per_float
            coarse_cache_bytes = 3 * n_frames * coarse_nx * coarse_ny * n_thetas * n_sigmas * bytes_per_float
            full_model_raw_bytes = 2 * n_frames * nx * ny * n_thetas * n_sigmas_full * n_frequencies * bytes_per_float
            scale = _selected_analysis_scale()
            if scale == "coarse":
                wavelet_size_label.configure(
                    text=(
                        f"Coarse RF wavelet NPY size: {_format_bytes(coarse_cache_bytes)} "
                        f"({n_frames} frames; temp phases +{_format_bytes(coarse_phase_bytes)})"
                    )
                )
            elif wavelet_format_var.get() == "zarr":
                full_path = param_entries.get("Full Model Wavelet Path")
                full_dir = full_path.get().strip() if full_path is not None else ""
                if full_dir:
                    exact_i = _folder_size_bytes(os.path.join(full_dir, "dwt_videodata2_i.zarr"))
                    exact_r = _folder_size_bytes(os.path.join(full_dir, "dwt_videodata2_r.zarr"))
                else:
                    exact_i = exact_r = None
                if exact_i is not None and exact_r is not None:
                    full_model_bytes = exact_i + exact_r
                    wavelet_size_label.configure(
                        text=(
                            f"Full-model wavelet current disk usage: {_format_bytes(full_model_bytes)} "
                            "(Zarr real + imaginary phases)"
                        )
                    )
                else:
                    wavelet_size_label.configure(
                        text=(
                            "Full-model wavelet Zarr size: compression-dependent; "
                            f"NPY equivalent {_format_bytes(full_model_raw_bytes)} "
                            f"({n_frames} frames)"
                        )
                    )
            else:
                wavelet_size_label.configure(
                    text=(
                        f"Full-model wavelet NPY size: {_format_bytes(full_model_raw_bytes)} "
                        f"({n_frames} frames)"
                    )
                )
        except Exception:
            wavelet_size_label.configure(text="Wavelet disk size: enter valid dimensions to calculate")

    def refresh_size_estimates():
        """Function for refresh size estimates."""
        estimate_gabor_library_size()
        estimate_wavelet_size()

    GABOR_ESTIMATE_KEYS = {
        "NX",
        "NY",
        "N_thetas",
        "Sigmas",
        "Sigmas Full Model",
        "Phases",
        "Frequencies",
        "Save Path",
    }
    WAVELET_ESTIMATE_KEYS = {"Movie Path", "Number of Frames", "NX", "NY", "N_thetas", "Sigmas", "Sigmas Full Model", "Frequencies", "Full Model Wavelet Path"}

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
        icon_base64 = ''
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

    def toggle_terminal():
        """Function for toggle terminal."""
        try:
            if str(frame_log) in right_pane.panes():
                right_pane.forget(frame_log)
            else:
                right_pane.add(frame_log, minsize=220, stretch="always")
                show_terminal_half()
        except Exception:
            pass

    left_panel_visible = [True]

    def toggle_left_panel():
        """Function for toggle left panel."""
        try:
            if str(left_frame) in paned_h.panes():
                paned_h.forget(left_frame)
                left_panel_visible[0] = False
            else:
                paned_h.add(left_frame, minsize=340, stretch="never")
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

    analysis_scale_var = tk.StringVar(value="coarse")
    wavelet_backend_var = tk.StringVar(value="legacy")
    neural_cache_format_var = tk.StringVar(value="npy")

    def refresh_scale_controls():
        """Refresh labels and estimates for the selected analysis scale."""
        scale = _selected_analysis_scale()
        label = _scale_label(scale)
        backend = _selected_wavelet_backend()
        backend_label = "Convolution" if backend == "convolution" else "Legacy"
        try:
            if backend == "convolution":
                btn_submit_gabor.configure(text=f"Prepare Convolution Kernels ({label})")
            else:
                btn_submit_gabor.configure(text=f"Build Gabor Library ({label})")
            btn_submit_wavelet.configure(text=f"Run Wavelet Decomposition ({label}, {backend_label})")
            btn_run_model_plots.configure(text=f"Run Model Plots ({label})")
            if scale == "full":
                wavelet_format_segment.configure(state="normal")
            else:
                wavelet_format_segment.configure(state="disabled")
            refresh_size_estimates()
        except NameError:
            pass

    def set_analysis_scale_from_panel(value):
        """Update the active coarse/full analysis scale."""
        if value not in {"coarse", "full"}:
            return
        analysis_scale_var.set(value)
        refresh_scale_controls()

    def set_wavelet_backend_from_panel(value):
        """Update the active legacy/convolution wavelet backend."""
        if value not in {"legacy", "convolution"}:
            return
        wavelet_backend_var.set(value)
        refresh_scale_controls()

    scale_frame = ctk.CTkFrame(frame_session, fg_color="transparent")
    scale_frame.pack(fill=tk.X, pady=(8, 8))
    ctk.CTkLabel(
        scale_frame,
        text="Analysis scale",
        text_color=text_color,
        font=ctk.CTkFont(size=12, weight="bold"),
    ).pack(anchor="w", pady=(0, 4))
    analysis_scale_segment = ctk.CTkSegmentedButton(
        scale_frame,
        values=["coarse", "full"],
        variable=analysis_scale_var,
        command=set_analysis_scale_from_panel,
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
    analysis_scale_segment.pack(fill=tk.X)
    analysis_scale_segment.set(analysis_scale_var.get())

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

    # --- Gabor filter bank ---
    frame_gabor = ttk.LabelFrame(frame_left, text="2 - Gabor Filter Bank", padding=15)
    frame_gabor.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_gabor.columnconfigure(1, weight=1)

    gabor_entries = {}
    for i, (label, default) in enumerate(gabor_param.items()):
        add_config_row(frame_gabor, label, default, gabor_entries, i, frame_color, GABOR_LABELS)

    gabor_format_var = tk.StringVar(value="npy")
    btn_submit_gabor = ctk.CTkButton(
        frame_gabor,
        text="Build Gabor Library (Coarse RF)",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(create_selected_gabor_library, "Gabor library construction"),
    )
    btn_submit_gabor.grid(row=len(gabor_param), column=0, columnspan=2, pady=(15, 0), sticky="ew")

    format_frame = ctk.CTkFrame(frame_gabor, fg_color="transparent")
    format_frame.grid(row=len(gabor_param)+1, column=0, columnspan=2, pady=(10, 0), sticky="w")
    ctk.CTkLabel(format_frame, text="Library format:", text_color=muted_text).pack(side=tk.LEFT)

    def _set_gabor_format(val):
        """Function for set gabor format.

        Args:
            val: Input value for this operation.
        """
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
        text="Gabor disk size: enter valid dimensions to calculate",
        font=main_font,
        background=frame_color,
        foreground=text_color,
    )
    gabor_size_label.grid(row=len(gabor_param)+2, column=0, columnspan=2, sticky="w", pady=(6, 0))

    # --- Stimulus wavelet pipeline ---
    frame_processing = ttk.LabelFrame(frame_left, text="3 - Stimulus Wavelet Pipeline", padding=15)
    frame_processing.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    wavelet_format_var = tk.StringVar(value="zarr")
    btn_submit_wavelet = ctk.CTkButton(
        frame_processing,
        text="Run Wavelet Decomposition (Coarse RF)",
        height=34,
        corner_radius=6,
        fg_color=primary_btn,
        hover_color="#1D4ED8",
        command=run_in_thread(run_wavelet, "Wavelet decomposition"),
    )
    btn_submit_wavelet.pack(fill=tk.X, pady=3)

    format_frame_wavelet = ctk.CTkFrame(frame_processing, fg_color="transparent")
    format_frame_wavelet.pack(anchor="w", pady=(10, 0))
    ctk.CTkLabel(format_frame_wavelet, text="Full-model format:", text_color=muted_text).pack(side=tk.LEFT)

    def _set_wavelet_format(val):
        """Function for set wavelet format.

        Args:
            val: Input value for this operation.
        """
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
        text="Wavelet disk size: enter valid dimensions to calculate",
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
            "Train Trial Indices",
            "Test Trial Indices",
            "Use Last Minute Holdout",
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

    def render_parameter_fields(preserve_values=False, loaded_values=None):
        """Function for render parameter fields.

        Args:
            preserve_values: Input value for this operation.
            loaded_values: Input value for this operation.
        """
        existing_values = {}
        if preserve_values:
            existing_values = {
                key: entry.get()
                for key, entry in param_entries.items()
                if key not in {"Neuron ID", "Spks Path"}
            }
        if loaded_values:
            existing_values.update({str(key): str(value) for key, value in loaded_values.items()})

        for widget in frame_params.winfo_children():
            widget.destroy()
        for key in list(param_entries):
            if key not in {"Neuron ID", "Spks Path"}:
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
    refresh_size_estimates()

    # --- Neural spike/position cache ---
    frame_neural_cache = ttk.LabelFrame(frame_left, text="1 - Neural Spike/Position Cache", padding=15)
    frame_neural_cache.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_neural_cache.columnconfigure(1, weight=1)

    add_config_row(
        frame_neural_cache,
        "Spks Path",
        param_defaults.get("Spks Path", "None"),
        param_entries,
        0,
        frame_color,
        ANALYSIS_LABELS,
    )

    neural_cache_format_frame = ctk.CTkFrame(frame_neural_cache, fg_color="transparent")
    neural_cache_format_frame.grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
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
    btn_create_neural_cache.grid(row=2, column=0, columnspan=2, pady=(12, 0), sticky="ew")

    # --- Neural & RF analysis ---
    frame_analysis = ttk.LabelFrame(frame_left, text="4 - Neural & RF Analysis", padding=15)
    frame_analysis.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    show_sem_var = tk.BooleanVar(value=False)
    sem_wrap = ttk.Frame(frame_analysis, style="TFrame")
    sem_wrap.pack(fill=tk.X, pady=(0, 10))
    ctk.CTkLabel(
        sem_wrap,
        text="Error bars",
        text_color=text_color,
        font=ctk.CTkFont(size=12, weight="bold"),
    ).pack(side=tk.LEFT, padx=(0, 10))
    ctk.CTkRadioButton(
        sem_wrap,
        text="Off",
        variable=show_sem_var,
        value=False,
        text_color=text_color,
    ).pack(side=tk.LEFT, padx=(0, 12))
    ctk.CTkRadioButton(
        sem_wrap,
        text="SEM",
        variable=show_sem_var,
        value=True,
        text_color=text_color,
    ).pack(side=tk.LEFT)

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
        text="Run Model Plots (Coarse RF)",
        height=34,
        corner_radius=6,
        fg_color="#374151",
        hover_color="#111827",
        command=run_in_thread(plot_selected_model_outputs, "model plot capture"),
    )
    btn_run_model_plots.pack(fill=tk.X, pady=(8, 0))

    # --- Export ---
    frame_export = ttk.LabelFrame(frame_left, text="5 - Export", padding=15)
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
        btn_create_neural_cache,
        btn_submit_gabor,
        btn_submit_wavelet,
        btn_submit_plot,
        btn_runRF,
        btn_run_model_plots,
        btn_export_all_results,
        btn_export_all_neurons,
        btn_export_individual_neuron,
        btn_save_state,
        btn_load_state,
    ]
    refresh_scale_controls()

    for section in (
        frame_session,
        frame_params,
        frame_neural_cache,
        frame_gabor,
        frame_processing,
        frame_analysis,
        frame_export,
        frame_controls,
    ):
        section.pack_forget()

    frame_session.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_params.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_neural_cache.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_gabor.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_processing.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_analysis.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_export.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    try:
        root.mainloop()
    finally:
        keep_awake.stop()
