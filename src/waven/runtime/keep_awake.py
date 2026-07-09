"""Cross-platform keep-awake support for long analysis runs.

The GUI can spend hours building Gabor banks and wavelet arrays. During that
time the operating system should not blank the display or suspend the machine.
This module wraps the native tools each desktop OS provides and exposes one
small lifecycle object for the GUI to start and stop.
"""
from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import threading
import time
from typing import List, Optional


class KeepAwake:
    """Prevent display sleep and system sleep while the process is active.

    Parameters
    ----------
    reason:
        Human-readable reason shown by some operating-system inhibit tools.

    Returns
    -------
    KeepAwake
        A start/stop controller. Calling ``start`` more than once is safe.
    """

    def __init__(self, reason: str = "waven analysis is running") -> None:
        self.reason = reason
        self._process: Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

    def start(self) -> None:
        """Activate the best available keep-awake mechanism for this OS."""
        if self._started:
            return
        self._started = True
        system = platform.system().lower()
        if system == "windows":
            self._start_windows()
        elif system == "darwin":
            self._start_process(["caffeinate", "-dimsu"])
        else:
            self._start_linux()

    def stop(self) -> None:
        """Release any sleep/display inhibition started by ``start``."""
        if not self._started:
            return
        self._started = False
        self._stop_event.set()
        if platform.system().lower() == "windows":
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except Exception:
                pass
        if self._process is not None:
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

    def _start_windows(self) -> None:
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED.
        flags = 0x80000000 | 0x00000001 | 0x00000002
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(flags)
        except Exception:
            return

        def refresh() -> None:
            while not self._stop_event.wait(30):
                try:
                    ctypes.windll.kernel32.SetThreadExecutionState(flags)
                except Exception:
                    break

        self._thread = threading.Thread(target=refresh, daemon=True)
        self._thread.start()

    def _start_linux(self) -> None:
        if shutil.which("systemd-inhibit"):
            self._start_process(
                [
                    "systemd-inhibit",
                    "--what=idle:sleep:handle-lid-switch",
                    "--who=waven",
                    f"--why={self.reason}",
                    "sleep",
                    "infinity",
                ]
            )
            if self._process is not None:
                return
        self._start_periodic_linux_reset()

    def _start_process(self, command: List[str]) -> None:
        try:
            self._process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            self._process = None

    def _start_periodic_linux_reset(self) -> None:
        def reset_loop() -> None:
            while not self._stop_event.wait(45):
                if os.environ.get("DISPLAY") and shutil.which("xdg-screensaver"):
                    subprocess.run(
                        ["xdg-screensaver", "reset"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                time.sleep(0.1)

        self._thread = threading.Thread(target=reset_loop, daemon=True)
        self._thread.start()
