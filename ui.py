from pathlib import Path
import sys
import traceback

ORIGINAL_STDERR = sys.stderr
PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))


def show_startup_error(title, message):
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

    config = waven.PipelineConfig.from_json(PROJECT_ROOT / "pipeline_config.json")
    zebra_gui.run(
        config.analysis.to_gui_mapping(),
        config.gabor.to_gui_mapping(),
        workflow=config.workflow,
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
