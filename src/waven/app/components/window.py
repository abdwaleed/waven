"""Window-level GUI setup helpers."""

from __future__ import annotations

import base64
import io
from pathlib import Path
import sys
import tempfile

import tkinter as tk
from PIL import Image


def apply_window_icon(root, icon_base64: str) -> Path | None:
    """Apply one hardcoded image to both the window and Windows taskbar icon."""
    if not icon_base64:
        return None
    app_icon = tk.PhotoImage(data=icon_base64)
    root.iconphoto(True, app_icon)
    root._waven_app_icon = app_icon
    if not sys.platform.startswith("win"):
        return None

    with tempfile.NamedTemporaryFile(prefix="waven-icon-", suffix=".ico", delete=False) as handle:
        icon_path = Path(handle.name)
    try:
        with Image.open(io.BytesIO(base64.b64decode(icon_base64))) as source:
            converted = source.convert("RGBA")
        try:
            converted.save(
                icon_path,
                format="ICO",
                sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
            )
        finally:
            converted.close()
        root.iconbitmap(default=str(icon_path))
        return icon_path
    except Exception:
        icon_path.unlink(missing_ok=True)
        raise
