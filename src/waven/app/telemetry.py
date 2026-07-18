"""Tk-safe terminal forwarding and task-resource telemetry.

These classes are deliberately independent of the rest of the GUI layout.
They are reused by every long-running button and never participate in the
scientific computation itself.
"""
from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from typing import Any, Dict, List, Optional, Tuple

import psutil

from ..runtime.performance import gpu_runtime_snapshot


class RedirectText:
    """Thread-safe stdout/stderr adapter that writes to a Tk text widget.

    Worker threads only enqueue text. Tk operations occur from the main loop,
    and :meth:`close` cancels the scheduled callback before the Tcl command is
    destroyed. That lifecycle prevents stale ``after`` callback errors during
    application shutdown.
    """

    def __init__(self, widget: Any, max_lines: int = 5000, flush_ms: int = 50) -> None:
        self.widget = widget
        self.max_lines = max(1, int(max_lines))
        self.flush_ms = max(1, int(flush_ms))
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._closed = False
        self._after_id: Optional[str] = None
        self._schedule_poll()

    def _schedule_poll(self) -> None:
        if self._closed:
            return
        try:
            self._after_id = self.widget.after(self.flush_ms, self._poll_flush)
        except tk.TclError:
            self._closed = True

    def close(self) -> None:
        """Stop scheduled callbacks before the containing Tk window closes."""
        self._closed = True
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def write(self, text: str) -> int:
        """Queue text written by a worker thread and return its length."""
        if self._closed or not text:
            return 0
        with self._lock:
            self._buffer.append(text)
        return len(text)

    def _poll_flush(self) -> None:
        self._after_id = None
        if self._closed:
            return
        try:
            self._flush()
        finally:
            self._schedule_poll()

    def _flush(self) -> None:
        with self._lock:
            chunk = "".join(self._buffer)
            self._buffer.clear()
        if not chunk or self._closed:
            return
        try:
            at_bottom = self.widget.yview()[1] >= 0.98
            self.widget.mark_set(tk.INSERT, tk.END)
            for part in chunk.splitlines(keepends=True):
                if "\r" not in part:
                    self.widget.insert(tk.END, part)
                    continue
                before, _, after = part.rpartition("\r")
                if before:
                    self.widget.insert(tk.END, before.replace("\r", ""))
                self.widget.delete("end-1c linestart", "end-1c lineend")
                if after:
                    self.widget.insert(tk.END, after.replace("\r", ""))
            line_count = int(float(self.widget.index("end-1c").split(".")[0]))
            if line_count > self.max_lines:
                self.widget.delete("1.0", f"{line_count - self.max_lines}.0")
            if at_bottom:
                self.widget.see(tk.END)
        except tk.TclError:
            self._closed = True

    def flush(self) -> None:
        """Synchronously flush only when called from Tk's main thread."""
        if threading.current_thread() is threading.main_thread():
            self._flush()

    def isatty(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return "utf-8"


class TaskResourceMonitor:
    """Sample process, disk, CPU, and optional GPU usage for one task.

    Sampling runs in a daemon thread at a low fixed rate. Missing NVML or CUDA
    support is treated as absent telemetry, never as an analysis failure.
    """

    def __init__(self, label: str, sample_interval_seconds: float = 2.0) -> None:
        self.label = label
        self.sample_interval_seconds = max(0.25, float(sample_interval_seconds))
        self.process = psutil.Process(os.getpid())
        self.start_perf: Optional[float] = None
        self.start_cpu: Any = None
        self.start_io: Any = None
        self.start_net: Any = None
        self.peak_rss = 0
        self.peak_cuda_allocated = 0
        self.peak_cuda_reserved = 0
        self.cpu_sample_count = 0
        self.cpu_percent_total = 0.0
        self.cpu_percent_peak = 0.0
        self.active_core_total = 0.0
        self.active_core_peak = 0
        self.gpu_samples: Dict[int, Dict[str, Any]] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Begin non-blocking telemetry sampling."""
        self.start_perf = time.perf_counter()
        self.start_cpu = self.process.cpu_times()
        self.start_io = self._io_counters()
        self.start_net = self._net_counters()
        self.peak_rss = self._rss()
        self._reset_cuda_peaks()
        try:
            self.process.cpu_percent(None)
            psutil.cpu_percent(None, percpu=True)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        """Stop sampling and return one stable metrics summary."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        return self.summary()

    def _sample_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_seconds):
            self.peak_rss = max(self.peak_rss, self._rss())
            allocated, reserved = self._cuda_peaks()
            self.peak_cuda_allocated = max(self.peak_cuda_allocated, allocated)
            self.peak_cuda_reserved = max(self.peak_cuda_reserved, reserved)
            self._sample_parallel_metrics()

    def _sample_parallel_metrics(self) -> None:
        try:
            process_cpu = max(0.0, float(self.process.cpu_percent(None)))
            core_usage = psutil.cpu_percent(None, percpu=True)
            active_cores = sum(value >= 20.0 for value in core_usage)
            self.cpu_sample_count += 1
            self.cpu_percent_total += process_cpu
            self.cpu_percent_peak = max(self.cpu_percent_peak, process_cpu)
            self.active_core_total += active_cores
            self.active_core_peak = max(self.active_core_peak, active_cores)
        except Exception:
            pass
        try:
            for sample in gpu_runtime_snapshot():
                self._record_gpu_sample(sample)
        except Exception:
            # NVML telemetry is optional and must never affect analysis work.
            pass

    def _record_gpu_sample(self, sample: Dict[str, Any]) -> None:
        device_id = int(sample["index"])
        record = self.gpu_samples.setdefault(
            device_id,
            {
                "name": str(sample.get("name", f"GPU {device_id}")),
                "util_total": 0.0,
                "util_count": 0,
                "util_peak": 0.0,
                "memory_util_total": 0.0,
                "memory_util_count": 0,
                "pcie_tx_total": 0,
                "pcie_tx_count": 0,
                "pcie_rx_total": 0,
                "pcie_rx_count": 0,
                "free_min": int(sample.get("free_bytes", 0)),
                "total_bytes": int(sample.get("total_bytes", 0)),
            },
        )
        record["free_min"] = min(record["free_min"], int(sample.get("free_bytes", 0)))
        record["total_bytes"] = max(record["total_bytes"], int(sample.get("total_bytes", 0)))
        self._record_average(record, sample, "compute_utilization", "util_total", "util_count", "util_peak")
        self._record_average(
            record,
            sample,
            "memory_utilization",
            "memory_util_total",
            "memory_util_count",
            None,
        )
        self._record_average(record, sample, "pcie_tx_bytes_per_second", "pcie_tx_total", "pcie_tx_count", None)
        self._record_average(record, sample, "pcie_rx_bytes_per_second", "pcie_rx_total", "pcie_rx_count", None)

    @staticmethod
    def _record_average(
        record: Dict[str, Any],
        sample: Dict[str, Any],
        sample_key: str,
        total_key: str,
        count_key: str,
        peak_key: Optional[str],
    ) -> None:
        value = sample.get(sample_key)
        if value is None:
            return
        record[total_key] += float(value)
        record[count_key] += 1
        if peak_key is not None:
            record[peak_key] = max(record[peak_key], float(value))

    def _rss(self) -> int:
        try:
            return int(self.process.memory_info().rss)
        except Exception:
            return 0

    def _io_counters(self) -> Any:
        try:
            return self.process.io_counters()
        except Exception:
            return None

    @staticmethod
    def _net_counters() -> Any:
        try:
            return psutil.net_io_counters()
        except Exception:
            return None

    @staticmethod
    def _reset_cuda_peaks() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device_id)
        except Exception:
            pass

    @staticmethod
    def _cuda_peaks() -> Tuple[int, int]:
        allocated = reserved = 0
        try:
            import torch

            if torch.cuda.is_available():
                for device_id in range(torch.cuda.device_count()):
                    allocated += int(torch.cuda.max_memory_allocated(device_id))
                    reserved += int(torch.cuda.max_memory_reserved(device_id))
        except Exception:
            pass
        return allocated, reserved

    def summary(self) -> Dict[str, Any]:
        """Return stable task metrics with the established GUI field names."""
        elapsed = max(time.perf_counter() - self.start_perf, 1e-6) if self.start_perf else 0.0
        cpu_seconds = 0.0
        try:
            current_cpu = self.process.cpu_times()
            cpu_seconds = (current_cpu.user - self.start_cpu.user) + (current_cpu.system - self.start_cpu.system)
        except Exception:
            pass
        io_now, net_now = self._io_counters(), self._net_counters()
        read_bytes = self._counter_delta(io_now, self.start_io, "read_bytes")
        write_bytes = self._counter_delta(io_now, self.start_io, "write_bytes")
        net_sent = self._counter_delta(net_now, self.start_net, "bytes_sent")
        net_recv = self._counter_delta(net_now, self.start_net, "bytes_recv")
        allocated, reserved = self._cuda_peaks()
        self.peak_cuda_allocated = max(self.peak_cuda_allocated, allocated)
        self.peak_cuda_reserved = max(self.peak_cuda_reserved, reserved)
        self.peak_rss = max(self.peak_rss, self._rss())
        sample_count = max(1, self.cpu_sample_count)
        sampled_cpu = self.cpu_percent_total / sample_count if self.cpu_sample_count else 100.0 * cpu_seconds / max(elapsed, 1e-6)
        return {
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime()),
            "elapsed": elapsed,
            "cpu_seconds": cpu_seconds,
            "cpu_percent": 100.0 * cpu_seconds / max(elapsed, 1e-6),
            "peak_ram": self.peak_rss,
            "disk_read": read_bytes,
            "disk_write": write_bytes,
            "gpu_allocated": self.peak_cuda_allocated,
            "gpu_reserved": self.peak_cuda_reserved,
            "net_sent": net_sent,
            "net_recv": net_recv,
            "cpu_percent_sampled": sampled_cpu,
            "cpu_percent_peak": self.cpu_percent_peak,
            "effective_cores_avg": sampled_cpu / 100.0,
            "effective_cores_peak": self.cpu_percent_peak / 100.0,
            "logical_cores": max(1, int(psutil.cpu_count(logical=True) or 1)),
            "active_cores_avg": self.active_core_total / sample_count if self.cpu_sample_count else None,
            "active_cores_peak": self.active_core_peak if self.cpu_sample_count else None,
            "disk_read_rate": read_bytes / max(elapsed, 1e-6),
            "disk_write_rate": write_bytes / max(elapsed, 1e-6),
            "gpu_devices": self._gpu_summary(),
        }

    @staticmethod
    def _counter_delta(current: Any, start: Any, name: str) -> int:
        if current is None or start is None:
            return 0
        return max(0, int(getattr(current, name, 0) - getattr(start, name, 0)))

    def _gpu_summary(self) -> List[Dict[str, Any]]:
        result = []
        for index, record in sorted(self.gpu_samples.items()):
            result.append(
                {
                    "index": index,
                    "name": record["name"],
                    "compute_utilization_avg": self._average(record, "util_total", "util_count"),
                    "compute_utilization_peak": record["util_peak"] if record["util_count"] else None,
                    "memory_utilization_avg": self._average(record, "memory_util_total", "memory_util_count"),
                    "pcie_tx_bytes_per_second": self._average(record, "pcie_tx_total", "pcie_tx_count"),
                    "pcie_rx_bytes_per_second": self._average(record, "pcie_rx_total", "pcie_rx_count"),
                    "free_vram_min": record["free_min"],
                    "total_vram": record["total_bytes"],
                }
            )
        return result

    @staticmethod
    def _average(record: Dict[str, Any], total_key: str, count_key: str) -> Optional[float]:
        count = int(record[count_key])
        return float(record[total_key]) / count if count else None
