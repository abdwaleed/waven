"""Ui module."""
from pathlib import Path
import json
import sys
import traceback

ORIGINAL_STDERR = sys.stderr
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))


def show_startup_error(title, message):
    """Function for show startup error.

    Args:
        title: Input value for this operation.
        message: Input value for this operation.
    """
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(title, message)
        root.destroy()
    except Exception:
        pass


def show_missing_dependency_error(exc):
    """Function for show missing dependency error.

    Args:
        exc: Input value for this operation.
    """
    missing = exc.name or "a required package"
    message = (
        f"Cannot launch waven GUI because Python is missing '{missing}'.\n\n"
        f"Interpreter being used:\n{sys.executable}\n\n"
        "Install the project dependencies in this interpreter, for example:\n"
        "python -m pip install -r requirements.txt"
    )
    show_startup_error("waven launch failed", message)
    print(message, file=ORIGINAL_STDERR)


try:
    import waven
    import waven.zebraGUI as zebra_gui

    config_path = PROJECT_ROOT / "pipeline_config.json"
    param_defaults = {}
    gabor_defaults = {}
    workflow = None
    gui_options = {}
    if config_path.exists():
        raw_text = config_path.read_text(encoding="utf-8")
        resolved_text = raw_text.replace("{PROJECT_ROOT}", str(PROJECT_ROOT).replace("\\", "/"))
        payload = json.loads(resolved_text) if resolved_text.strip() else {}
        if not isinstance(payload, dict):
            raise ValueError("pipeline_config.json must contain a JSON object when provided.")
        workflow = payload.get("workflow") or None
        gui_options = dict(payload.get("gui") or {})
        gabor_defaults = dict(payload.get("gabor") or payload.get("gabor_param") or {})
        common = dict(payload.get("common") or {})
        legacy = dict(payload.get("analysis") or payload.get("param_defaults") or {})
        if workflow == "ephys":
            workflow_values = dict(payload.get("ephys") or {})
        else:
            workflow_values = dict(
                payload.get("two_photon")
                or payload.get("2p")
                or payload.get("two_p")
                or {}
            )
        param_defaults = {**legacy, **common, **workflow_values}
    zebra_gui.run(
        param_defaults,
        gabor_defaults,
        workflow=workflow,
        gui_options=gui_options,
    )
except SystemExit:
    raise
except ModuleNotFoundError as exc:
    show_missing_dependency_error(exc)
    raise SystemExit(1) from exc
except Exception:
    log_path = PROJECT_ROOT / "waven_launch_error.log"
    details = traceback.format_exc()
    log_path.write_text(details, encoding="utf-8")
    message = (
        "The waven GUI crashed during startup.\n\n"
        f"Traceback saved to:\n{log_path}\n\n"
        f"{details}"
    )
    show_startup_error("waven startup failed", message)
    print(message, file=ORIGINAL_STDERR)
    raise SystemExit(1)
