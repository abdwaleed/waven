"""Small reusable helpers for the Tkinter GUI.

The main GUI module still owns widget construction and workflow callbacks. This
module keeps simple, stateless helpers out of that file so readers can find path
formatting, size formatting, tooltip behavior, and Gabor parameter normalization
without opening the full GUI implementation.
"""
import os
import hashlib
import tkinter as tk
from pathlib import Path

from ..config import parse_literal

class ToolTip(object):
    """Container for ToolTip."""
    def __init__(self, widget):
        """Function for init.

        Args:
            widget: Input value for this operation.
        """
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        """Function for enter.

        Args:
            event: Input value for this operation.
        """
        self.schedule()

    def leave(self, event=None):
        """Function for leave.

        Args:
            event: Input value for this operation.
        """
        self.unschedule()
        self.hidetip()

    def schedule(self):
        """Function for schedule."""
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        """Function for unschedule."""
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
        """Function for showtip.

        Args:
            event: Input value for this operation.
        """
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
        """Function for hidetip."""
        tw = self.tipwindow
        self.tipwindow = None
        if tw: tw.destroy()

def _parse_data_dir(value):
    """Function for parse data dir.

    Args:
        value: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for format bytes.

    Args:
        size: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if size is None or size < 0:
        return "0.00 GB"
    return f"{size / (1024 ** 3):.2f} GB"


def _build_size_text(gb_bytes):
    """Function for build size text.

    Args:
        gb_bytes: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return _format_bytes(gb_bytes)


def _folder_size_bytes(path):
    """Function for folder size bytes.

    Args:
        path: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
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
    """Function for zarr output path.

    Args:
        path: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    path = str(path).strip()
    if not path:
        return path
    if path.endswith(".zarr"):
        return path
    stem, ext = os.path.splitext(path)
    return (stem if ext else path) + ".zarr"


def _safe_name(value):
    """Function for safe name.

    Args:
        value: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value)).strip("_") or "plot"


def _export_safe_name(value, maximum_length=64):
    """Return a stable short filename component suitable for Windows exports.

    A graph title can include fitted values and easily become longer than the
    remaining Windows path budget once it is placed in a per-neuron directory.
    Keep a readable prefix and append a digest so distinct long titles remain
    distinct without duplicating a long title in both directory and filename.
    """
    safe = _safe_name(value)
    maximum_length = max(12, int(maximum_length))
    if len(safe) <= maximum_length:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:8]
    return f"{safe[:maximum_length - len(digest) - 1].rstrip('_')}_{digest}"


def _default_gabor_library_path(path_value, suffix):
    """Function for default gabor library path.

    Args:
        path_value: Input value for this operation.
        suffix: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    base = Path(str(path_value or "gabors_library.npy")).expanduser()
    if base.suffix:
        return str(base.with_name(f"{base.stem}{suffix}{base.suffix}"))
    return str(base.with_name(f"{base.name}{suffix}.npy"))


def _normalise_gabor_params(gabor_param):
    """Function for normalise gabor params.

    Args:
        gabor_param: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    normalised = dict(gabor_param or {})
    for key in ("N_thetas", "Sigmas", "Frequencies", "Phases", "NX", "NY", "Save Path"):
        normalised.setdefault(key, "")
    legacy_path = str(normalised.get("Save Path") or "").strip()
    normalised.setdefault(
        "Coarse Library Path",
        _default_gabor_library_path(legacy_path, "_coarse") if legacy_path else "",
    )
    normalised.setdefault(
        "Fine Library Path",
        _default_gabor_library_path(legacy_path, "_fine") if legacy_path else "",
    )
    return normalised


def _ordered_float_union(*sequences):
    """Function for ordered float union.

    Args:
        sequences: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    values = []
    seen = set()
    for sequence in sequences:
        if sequence is None:
            continue
        for item in sequence:
            value = float(item)
            key = repr(value)
            if key not in seen:
                seen.add(key)
                values.append(value)
    return values

