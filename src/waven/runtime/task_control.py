"""Shared task progress and cooperative cancellation helpers."""
from __future__ import annotations

import time
from typing import Optional


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
    return f"\n[{timestamp}] {str(task_name).upper()}\n  Starting...\n"


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
    return f"[{timestamp}] {task_name} {str(status).lower()}."
