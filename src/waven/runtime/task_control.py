"""Shared task progress and cooperative cancellation helpers."""
from __future__ import annotations

import time
from typing import Optional


_TERMINAL_WIDTH = 78


def _terminal_rule(left: str = "", fill: str = "-") -> str:
    """Return a bounded decorative terminal rule without relying on a TTY."""
    left = str(left).strip()
    if left:
        left = f" {left} "
    return f"+{left}{fill * max(1, _TERMINAL_WIDTH - len(left) - 2)}+"


def _terminal_row(label: str, value: str) -> str:
    """Format one stable dashboard row, truncating only exceptionally long text."""
    prefix = f"| {str(label):<10}"
    available = max(8, _TERMINAL_WIDTH - len(prefix) - 2)
    text = str(value).replace("\n", " ")
    if len(text) > available:
        text = f"{text[: max(1, available - 3)]}..."
    return f"{prefix}{text:<{available}} |"


def _terminal_bottom() -> str:
    return f"+{'-' * (_TERMINAL_WIDTH - 2)}+"


def _format_bytes(value: float) -> str:
    """Return a compact binary-size label independent of GUI helpers."""
    value = max(0.0, float(value or 0.0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TiB"


class OperationCancelled(RuntimeError):
    """Raised when a GUI task is cancelled by the user."""


def check_cancelled(cancel_event=None) -> None:
    """Raise :class:`OperationCancelled` when ``cancel_event`` is set."""
    if cancel_event is not None and cancel_event.is_set():
        raise OperationCancelled("Operation cancelled by user.")


def format_duration(seconds: float) -> str:
    """Return a compact human-readable duration."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.0f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.0f}s"


def progress_message(label, completed, total, start_time, unit="steps") -> str:
    """Format percent, elapsed time, ETA, and throughput for terminal logs."""
    elapsed = max(time.time() - start_time, 1e-6)
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    percent = 100 * completed / total
    speed = completed / elapsed
    remaining = max(total - completed, 0)
    eta = remaining / speed if speed > 0 else 0
    return (
        f"{label}: {completed}/{total} ({percent:.1f}%) | "
        f"elapsed {format_duration(elapsed)} | ETA {format_duration(eta)} | "
        f"{speed:.2f} {unit}/s"
    )


def task_start_message(task_name: str, started_at: Optional[float] = None) -> str:
    """Format the standard terminal banner for a user-initiated GUI task.

    Args:
        task_name: Human-readable action name, such as ``"Run Coarse RF"``.
        started_at: Optional Unix timestamp. When omitted, the current local
            time is used.

    Returns:
        A newline-delimited banner suitable for ``print``. The shared format
        keeps neural-cache, stimulus, Gabor, wavelet, RF, model, and export
        actions visually consistent in the GUI terminal.
    """
    timestamp = time.strftime("%H:%M:%S", time.localtime(started_at))
    return "\n".join(
        (
            "",
            _terminal_rule(f"WAVEN TASK - {str(task_name).upper()}"),
            _terminal_row("Status", f"RUNNING | started {timestamp}"),
            _terminal_row("Telemetry", "CPU | RAM | disk I/O | GPU sampled every 2 seconds"),
            _terminal_bottom(),
        )
    )


def task_finish_message(task_name: str, status: str, finished_at: Optional[float] = None) -> str:
    """Format the standard final line for a GUI task.

    Args:
        task_name: Human-readable action name used in the start banner.
        status: Final state, normally ``"finished"``, ``"failed"``, or
            ``"cancelled"``.
        finished_at: Optional Unix timestamp. When omitted, the current local
            time is used.

    Returns:
        A concise timestamped completion line. Detailed resource metrics are
        emitted separately by the GUI task monitor.
    """
    timestamp = time.strftime("%H:%M:%S", time.localtime(finished_at))
    symbol = {"finished": "[OK]", "failed": "[FAIL]", "cancelled": "[CANCELLED]"}.get(
        str(status).lower(), "[DONE]"
    )
    return f"{symbol} {str(task_name)} {str(status).lower()} at {timestamp}."


def task_progress_message(task_name, percent, detail, started_at) -> str:
    """Format one rate-limited stage update for every major GUI action."""
    elapsed = max(time.time() - float(started_at), 0.0)
    percent = max(0.0, min(float(percent), 100.0))
    eta = elapsed * (100.0 - percent) / percent if percent > 0 else 0.0
    detail = str(detail or "working")
    return (
        f"  > {percent:5.1f}%  {str(task_name)}  |  {detail}"
        f"  |  elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
    )


def task_summary_message(task_name, status, metrics) -> str:
    """Render the standard end-of-task dashboard for every GUI action."""
    metrics = metrics or {}
    status_key = str(status).lower()
    symbol = {"finished": "OK", "failed": "FAIL", "cancelled": "CANCELLED"}.get(status_key, "DONE")
    elapsed = format_duration(metrics.get("elapsed", 0.0))
    effective_cores = metrics.get("effective_cores_avg")
    peak_cores = metrics.get("effective_cores_peak")
    logical_cores = metrics.get("logical_cores")
    if effective_cores is None:
        cpu_text = f"{metrics.get('cpu_percent', 0.0):.1f}% process CPU"
    else:
        cpu_text = (
            f"{effective_cores:.1f} effective cores avg | {peak_cores:.1f} peak"
            + (f" / {int(logical_cores)} logical" if logical_cores else "")
        )
    disk_text = (
        f"read {_format_bytes(metrics.get('disk_read', 0))} "
        f"({_format_bytes(metrics.get('disk_read_rate', 0))}/s) | "
        f"write {_format_bytes(metrics.get('disk_write', 0))} "
        f"({_format_bytes(metrics.get('disk_write_rate', 0))}/s)"
    )
    gpu_devices = metrics.get("gpu_devices") or []
    if gpu_devices:
        gpu_rows = []
        for device in gpu_devices:
            utilization = device.get("compute_utilization_avg")
            util_text = f"{utilization:.0f}% busy" if utilization is not None else "utilization unavailable"
            pcie_parts = []
            if device.get("pcie_rx_bytes_per_second") is not None:
                pcie_parts.append(f"rx {_format_bytes(device['pcie_rx_bytes_per_second'])}/s")
            if device.get("pcie_tx_bytes_per_second") is not None:
                pcie_parts.append(f"tx {_format_bytes(device['pcie_tx_bytes_per_second'])}/s")
            gpu_rows.append(
                (
                    f"GPU {device.get('index', '?')}",
                    f"{util_text} | min free {_format_bytes(device.get('free_vram_min', 0))}"
                    + (f" | PCIe {' '.join(pcie_parts)}" if pcie_parts else ""),
                )
            )
    else:
        gpu_rows = [("GPU", f"peak allocation {_format_bytes(metrics.get('gpu_allocated', 0))}")]
    rows = [
        _terminal_rule(f"{symbol} {str(task_name).upper()} - {status_key.upper()}"),
        _terminal_row("Elapsed", elapsed),
        _terminal_row("CPU", cpu_text),
        _terminal_row("Memory", f"peak RAM {_format_bytes(metrics.get('peak_ram', 0))}"),
        _terminal_row("Disk", disk_text),
    ]
    rows.extend(_terminal_row(label, value) for label, value in gpu_rows)
    rows.append(_terminal_bottom())
    return "\n".join(rows)
