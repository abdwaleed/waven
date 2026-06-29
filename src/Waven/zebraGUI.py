"""Tkinter GUI for the waven Gabor-wavelet analysis pipeline.

Provides a staged workflow (Gabor bank → stimulus wavelets → neural RF analysis),
live terminal output, and embedded matplotlib visualizations.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import threading
import sys
import gc
from . import LoadPinkNoise as lpn
import traceback

# --- DPI Awareness ---
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    myappid = 'neuro.gabor.toolkit.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

from .config import (
    AnalysisConfig,
    DEFAULT_COMMON_PARAMS,
    DEFAULT_EPHYS_PARAMS,
    DEFAULT_TWO_PHOTON_PARAMS,
    WORKFLOW_2P,
    WORKFLOW_EPHYS,
    coarse_grid_dimensions,
    parse_literal,
)
from .WaveletGenerator import *
from .LoadPinkNoise import *
from .Analysis_Utils import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np
from .wavelet_io import convert_npy_to_zarr


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.widget.bind('<Enter>', self.enter)
        self.widget.bind('<Leave>', self.leave)

    def enter(self, event=None):
        self.schedule()

    def leave(self, event=None):
        self.unschedule()
        self.hidetip()

    def schedule(self):
        self.unschedule()
        self.id = self.widget.after(500, self.showtip)

    def unschedule(self):
        id = self.id
        self.id = None
        if id:
            self.widget.after_cancel(id)

    def showtip(self, event=None):
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
        tw = self.tipwindow
        self.tipwindow = None
        if tw: tw.destroy()

def _parse_data_dir(value):
    try:
        parsed = parse_literal(value, "Dir")
    except ValueError:
        parsed = value
    if parsed is None:
        return []
    if isinstance(parsed, (list, tuple)):
        return [str(path) for path in parsed]
    return [str(parsed)]


def _prompt_workflow() -> str:
    """Ask the user to choose a data workflow before opening the main GUI."""
    dialog = tk.Tk()
    dialog.title("Select Data Workflow")
    dialog.geometry("420x180")
    dialog.resizable(False, False)
    dialog.configure(bg="#F0F3F7")

    choice = {"workflow": None}

    tk.Label(
        dialog,
        text="Which neural data type are you analyzing?",
        bg="#F0F3F7",
        fg="#2B3A4A",
        font=("Segoe UI Variable Display", 11, "bold"),
    ).pack(pady=(24, 12))

    button_frame = tk.Frame(dialog, bg="#F0F3F7")
    button_frame.pack(pady=8)

    def select(workflow):
        choice["workflow"] = workflow
        dialog.destroy()

    tk.Button(
        button_frame,
        text="Two-Photon (2p)",
        width=18,
        command=lambda: select(WORKFLOW_2P),
    ).pack(side=tk.LEFT, padx=8)
    tk.Button(
        button_frame,
        text="Electrophysiology (Ephys)",
        width=22,
        command=lambda: select(WORKFLOW_EPHYS),
    ).pack(side=tk.LEFT, padx=8)

    dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
    dialog.grab_set()
    dialog.wait_window()

    if choice["workflow"] is None:
        raise SystemExit(0)
    return choice["workflow"]


def run(param_defaults, gabor_param, workflow=None):
    if workflow is None:
        workflow = _prompt_workflow()

    GABOR_LABELS = {
        "N_thetas": "Orientation Count",
        "Sigmas": "Filter Sizes (px)",
        "Frequencies": "Spatial Frequencies (cyc/px)",
        "Phases": "Phases (degrees)",
        "NX": "Stimulus Width (px)",
        "NY": "Stimulus Height (px)",
        "Save Path": "Gabor Library Output (.npy)",
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
        "Library Path": "Gabor Library Path (.npy)",
        "Spks Path": "Pre-aligned Spikes (.npy, optional)",
        "Full Model Wavelet Path": "Full-Model Wavelet Store",
        "Full Model Save Path": "Full-Model Results Directory",
        "Neuron ID": "Neuron Index",
    }

    workflow_param_keys = AnalysisConfig.gui_param_keys(workflow)
    merged_param_defaults = dict(DEFAULT_COMMON_PARAMS)
    if workflow == WORKFLOW_2P:
        merged_param_defaults.update(DEFAULT_TWO_PHOTON_PARAMS)
    else:
        merged_param_defaults.update(DEFAULT_EPHYS_PARAMS)
    merged_param_defaults.update(param_defaults)
    filtered_param_defaults = {
        key: merged_param_defaults.get(key, "")
        for key in workflow_param_keys
    }

    FIELD_LABELS = {**GABOR_LABELS, **ANALYSIS_LABELS}

    BROWSE_KIND = {
        "Save Path": "file",
        "Movie Path": "file",
        "Library Path": "file",
        "Spks Path": "file",
        "Path Directory": "dir",
        "Dir": "dir",
        "Full Model Wavelet Path": "dir",
        "Full Model Save Path": "dir",
    }

    BROWSE_ICONS = {"file": "📄", "dir": "📁"}

    class RedirectText:
        def __init__(self, widget): self.widget = widget
        def write(self, string):
            self.widget.insert(tk.END, string)
            self.widget.see(tk.END)
        def flush(self): pass

    task_state = {"name": None}

    def show_terminal_half():
        """Reveal the log pane and allocate the lower half of the window to it."""
        if not frame_log.winfo_ismapped():
            root_vpaned.add(frame_log, weight=1)
        root.update_idletasks()
        total_height = root_vpaned.winfo_height()
        if total_height > 120:
            root_vpaned.sashpos(0, max(200, int(total_height * 0.5)))

    def begin_task(task_name):
        task_state["name"] = task_name
        show_terminal_half()
        status_var.set(f"Running: {task_name}")
        progress_bar.configure(mode="indeterminate")
        progress_bar.start(12)
        for btn in all_buttons:
            btn.config(state=tk.DISABLED)
        print(f"\n--- {task_name} ---")

    def end_task():
        progress_bar.stop()
        progress_bar.configure(mode="determinate", value=0)
        if task_state["name"]:
            print(f"--- Finished: {task_state['name']} ---\n")
        task_state["name"] = None
        status_var.set("Ready")
        for btn in all_buttons:
            btn.config(state=tk.NORMAL)

    def run_in_thread(func, task_name=None):
        label = task_name or func.__name__.replace("_", " ").title()

        def wrapper(*args, **kwargs):
            root.after(0, lambda: begin_task(label))

            def thread_target():
                try:
                    func(*args, **kwargs)
                except Exception as exc:
                    # Print a clear header for the error
                    print(f"\n[!] AN ERROR OCCURRED IN TASK: {label}")
                    # This will dump the full "most recent call last" traceback 
                    # straight to your GUI terminal!
                    traceback.print_exc() 
                finally:
                    root.after(0, end_task)

            threading.Thread(target=thread_target, daemon=True).start()

        return wrapper
    
    def create_gabor():
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        nx = int(gabor_entries["NX"].get())
        ny = int(gabor_entries["NY"].get())
        n_theta = int(gabor_entries["N_thetas"].get())
        offsets = parse_literal(gabor_entries["Phases"].get(), "Phases")
        path_save = gabor_entries["Save Path"].get()
        xs = np.arange(nx)
        ys = np.arange(ny)
        thetas = np.array([(i * np.pi) / n_theta for i in range(n_theta)])
        sigmas = np.array(sigmas)
        offsets = np.array(offsets)
        frequencies = np.array(frequencies)
        
        if frequencies.size and np.any(frequencies != 0):
            L = makeFilterLibrary2(xs, ys, thetas, sigmas, offsets, frequencies)
        else:
            frequency = frequencies[0] if frequencies.size else 0
            L = makeFilterLibrary(xs, ys, thetas, sigmas, offsets, frequency, freq=False)
        np.save(path_save, L)
        print(f"Gabor library saved to: {path_save}")

    def run_wavelet():
        movpath = param_entries["Movie Path"].get()
        parent_dir = os.path.dirname(movpath)
        
        # Define paths to the high-res wavelets
        file_0 = os.path.join(parent_dir, "dwt_videodata_0.npy")
        file_1 = os.path.join(parent_dir, "dwt_videodata_1.npy")
        
        # Extract common parameters needed for both high-res and coarse wavelets
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        frequencies = parse_literal(gabor_entries["Frequencies"].get(), "Frequencies")
        lib_path = param_entries["Library Path"].get()
        nx = int(param_entries["NX"].get())
        ny = int(param_entries["NY"].get())
        n_thetas = int(gabor_entries["N_thetas"].get())

        # CHECK 1: Do the high-res files already exist?
        if os.path.exists(file_0) and os.path.exists(file_1):
            print(f"Found existing high-resolution wavelets in {parent_dir}. Skipping decomposition.")
        else:
            print("Step 1/2: Downsampling stimulus movie and real phase decomposition...")
            visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
            analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")

            if (visual_coverage != analysis_coverage):
                visual_coverage = np.array(visual_coverage)
                analysis_coverage = np.array(analysis_coverage)
                ratio_x = 1 - ((visual_coverage[0] - visual_coverage[1]) - (analysis_coverage[0] - analysis_coverage[1])) / (visual_coverage[0] - visual_coverage[1])
                ratio_y = 1 - ((visual_coverage[2] - visual_coverage[3]) - (analysis_coverage[2] - analysis_coverage[3])) / (visual_coverage[2] - visual_coverage[3])
            else:
                ratio_x = ratio_y = 1
                
            downsample_video_binary(movpath, visual_coverage, analysis_coverage, shape=(ny, nx), chunk_size=1000, ratios=(ratio_x, ratio_y))
            videodata = np.load(movpath[:-4] + '_downsampled.npy')
            videodata = videodata.astype(int) - np.logical_not(videodata).astype(int)
            
            waveletDecomposition(videodata, 0, sigmas, parent_dir, lib_path)
            
            print("Step 2/2: Wavelet decomposition (imaginary phase)...")
            waveletDecomposition(videodata, 1, sigmas, parent_dir, lib_path)

        # STEP 3: Generate the downsampled coarse wavelets (the missing file)
        print("Step 3: Generating coarse downsampled wavelets...")
        
        # NOTE: I am hardcoding nx=27, ny=11 here because they match your original defaults, 
        # and they do not appear to exist in your GUI input config. 
        # Update these if you eventually add "Coarse NX" and "Coarse NY" to the GUI.
        lpn.coarseWavelet(
            path=parent_dir,
            downsampling=False,
            nx0=nx,
            ny0=ny,
            no=n_thetas,
            ns=len(sigmas),
            nf=len(frequencies),
            nx=None, # Lets code handle the downsampling ratio itself
            ny=None, # Lets code handle the downsampling ratio itself
            chunk_size=None
        )
        
        
        print(f"All wavelet files are ready in: {parent_dir}")
    
    def embed_interactive_figure(fig, parent_container, title=None):
        """Embed a matplotlib figure with navigation toolbar in ``parent_container``."""
        section = ttk.LabelFrame(parent_container, text=title, padding=6) if title else ttk.Frame(parent_container, style="TFrame")
        section.pack(side=tk.TOP, fill=tk.BOTH, expand=False, pady=8)

        canvas = FigureCanvasTkAgg(fig, master=section)
        canvas.draw()

        toolbar = NavigationToolbar2Tk(canvas, section)
        toolbar.update()

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        return canvas
    
    def plot_data():
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
            return

        pathdata = os.path.join(data_dirs[0], exp_info[0], exp_info[1], str(exp_info[2]))
        pathsuite2p = os.path.join(pathdata, 'suite2p')
        deg_per_pix = abs(xM - xm) / nx
        sigmas_deg = np.trunc(2 * deg_per_pix * sigmas * 100) / 100

        if spks_path.strip().lower() in ("", "none", "null"):
            from . import time_alignment as ta

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
                )
            except NotImplementedError as exc:
                print(exc)
                return
            spks = aligned.spikes
            neuron_pos = aligned.neuron_pos
            if workflow == WORKFLOW_2P:
                neuron_pos[:, 1] = abs(neuron_pos[:, 1] - np.max(neuron_pos[:, 1]))
        else:
            try:
                spks = np.load(spks_path)
                parent_dir = os.path.dirname(spks_path)
                neuron_pos = np.load(os.path.join(parent_dir, 'pos.npy'))
            except Exception as e:
                print(f"File not found: {e}")
                return

        print("Loading neural data and coarse wavelets...")
        respcorr = repetability_trial3(spks, neuron_pos, plotting=False)
        skewness = np.array(compute_skewness_neurons(spks, plotting=False))
        filter_mask = np.logical_and(respcorr >= 0.2, skewness <= 20)

        point_alphas = np.where(filter_mask, 1.0, 0.05)

        parent_dir = os.path.dirname(movpath)
        try:
            wavelets_downsampled = np.load(os.path.join(parent_dir, 'dwt_downsampled_videodata.npy'))
            w_c_downsampled = wavelets_downsampled[2]
            del wavelets_downsampled
            gc.collect()
        except Exception as e:
            print(f"Decomposition loading failed: {e}")
            return
            
        n_frames = min(nb_frames, w_c_downsampled.shape[0], spks.shape[1])

        # Updated the PearsonCorrelationPinkNoise call to pass nf and frequencies
        rfs_gabor = PearsonCorrelationPinkNoise(w_c_downsampled[:n_frames].reshape(n_frames, -1),
                                                np.mean(spks[:, :n_frames], axis=0),
                                                neuron_pos, coarse_nx, coarse_ny, ns, nf, analysis_coverage, screen_ratio, sigmas_deg, frequencies,
                                                n_orientations=n_orientations,
                                                plotting=False)

        def render_gui_plots():
            for widget in frame_plot.winfo_children():
                widget.destroy()
            plt.close('all')

            fig1, ax1 = plt.subplots(figsize=(6, 5), constrained_layout=True)
            ax1.scatter(neuron_pos[:, 0], neuron_pos[:, 1], c='k', alpha=0.3, label="Neurons", picker=True, rasterized=True)
            ax1.set_title("Neuron Positions (µm)")
            ax1.set_xlabel("X (µm)")
            ax1.set_ylabel("Y (µm)")

            fig10, axes10 = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
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
                scatter = ax10[idx].scatter(
                    neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=values,
                    cmap=cmap, alpha=point_alphas, rasterized=True,
                )
                fig10.colorbar(scatter, ax=ax10[idx], fraction=0.046)
                ax10[idx].set_title(title)
                ax10[idx].set_xlabel("X (µm)")
                ax10[idx].set_ylabel("Y (µm)")

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

            def onpick(event):
                try:
                    neuron_id = event.ind[0]
                    entry_neuron.delete(0, tk.END)
                    entry_neuron.insert(0, str(neuron_id))

                    ax2.clear()
                    ax2.plot(np.mean(spks[:, :, neuron_id], axis=0), label=f"Neuron {neuron_id} Spike Times")
                    ax2.legend()
                    canvas2.draw()

                    rf2d, x_tuning, y_tuning, ori_tun, s_tuning, f_tuning = PlotTuningCurve(rfs_gabor, neuron_id, analysis_coverage, sigmas_deg, screen_ratio, frequencies, show=False)
                    for ax in ax3: ax.clear()

                    ax3[0].imshow(rf2d, cmap='coolwarm', aspect='auto')
                    ax3[0].set_xticks([0, rf2d.shape[1]], [xM, xm])
                    ax3[0].set_yticks([0, rf2d.shape[0]], [yM, ym])
                    ax3[0].set_title('2D')
                    ax3[1].plot(x_tuning[::-1], c='k')
                    ax3[1].set_title('Elevation (deg)')
                    ax3[1].set_xticks([0, rf2d.shape[0]], [ym, yM])
                    ax3[2].plot(y_tuning, c='k')
                    ax3[2].set_title('Azimuth')
                    ax3[2].set_xticks([0, rf2d.shape[1]], [xM, xm])
                    ax3[3].plot(ori_tun, 'o-', c='k')
                    ax3[3].set_title('Orientation')
                    ax3[3].set_xticks([0, 4, 8], [0, 90, 180])
                    ax3[4].plot(s_tuning, 'o-', c='k')
                    ax3[4].set_title('Size (deg)')
                    ax3[4].set_xticks([0, len(sigmas) - 1], [sigmas_deg[0], sigmas_deg[-1]])
                    ax3[5].plot(f_tuning, 'o-', c='k')
                    ax3[5].set_title('Spatial Frequency')
                    ax3[5].set_xticks(range(len(frequencies)), [round(f, 3) for f in frequencies])
                    canvas3.draw()
                except Exception as e:
                    print(f"Error drawing pick event: {e}")

            fig1.canvas.mpl_connect('pick_event', onpick)

            global canvas2, canvas3

            embed_interactive_figure(fig1, frame_plot, title="Neuron Layout")
            embed_interactive_figure(fig10, frame_plot, title="Population Retinotopy Maps")
            canvas2 = embed_interactive_figure(fig2, frame_plot, title="Spike Train")
            canvas3 = embed_interactive_figure(fig3, frame_plot, title="Selected Neuron Tuning")
            canvas_right.configure(scrollregion=canvas_right.bbox("all"))

            def click_RF():
                try:
                    neuron_id = int(param_entries["Neuron ID"].get())
                    ax2.clear()
                    ax2.plot(np.mean(spks[:, :, neuron_id], axis=0), label=f"Neuron {neuron_id} Spike Times")
                    ax2.legend()
                    canvas2.draw()

                    rf2d, x_tuning, y_tuning, ori_tun, s_tuning = PlotTuningCurve(rfs_gabor, neuron_id, analysis_coverage, sigmas_deg, screen_ratio, frequencies, show=False)
                    for ax in ax3: ax.clear()
                        
                    ax3[0].imshow(rf2d, cmap='coolwarm', aspect='auto')
                    ax3[0].set_xticks([0, rf2d.shape[1]], [xM, xm])
                    ax3[0].set_yticks([0, rf2d.shape[0]], [yM, ym])
                    ax3[0].set_title('2D')
                    ax3[1].plot(x_tuning[::-1], c='k')
                    ax3[1].set_title('Elevation (deg)')
                    ax3[1].set_xticks([0, rf2d.shape[0]], [ym, yM])
                    ax3[2].plot(y_tuning, c='k')
                    ax3[2].set_title('Azimuth')
                    ax3[2].set_xticks([0, rf2d.shape[1]], [xM, xm])
                    ax3[3].plot(ori_tun, 'o-', c='k')
                    ax3[3].set_title('Orientation')
                    ax3[3].set_xticks([0, 4, 8], [0, 90, 180])
                    ax3[4].plot(s_tuning, 'o-', c='k')
                    ax3[4].set_title('Size (deg)')
                    ax3[4].set_xticks([0, len(sigmas) - 1], [sigmas_deg[0], sigmas_deg[-1]])
                    canvas3.draw()
                except Exception as e:
                    print(f"Failed to plot RF: {e}")

            btn_runRF.config(command=click_RF)
            print("Plots rendered successfully.")

        root.after(0, render_gui_plots)

    def click_save(): print("Save Retinotopy functionality triggered.")
    def export_plots(): print("Exporting SVGs...")

    def convert_to_zarr_gui():
        movpath = param_entries["Movie Path"].get()
        if not movpath or not os.path.exists(movpath):
            print("Error: Please specify a valid Movie Path so we can locate your folder.")
            return
            
        npy_dir = os.path.dirname(movpath)
        zarr_dir = npy_dir 
        hz = int(param_entries["Hz"].get())
        
        print(f"Starting Zarr conversion...\nTarget Directory: {npy_dir}")
        convert_npy_to_zarr(npy_dir, zarr_dir, hz)

    def quit_app():
        root.quit()
        root.destroy()

    def browse_path(entry_widget, kind):
        """Open a file or directory picker appropriate for the field type."""
        if kind == "file":
            path = filedialog.askopenfilename(title="Select file")
        else:
            path = filedialog.askdirectory(title="Select directory")
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def add_config_row(parent, key, default, entries_dict, row, bg, labels_map):
        """Render one labeled configuration row with an optional typed browse button."""
        label_text = labels_map.get(key, key)
        tk.Label(parent, text=label_text, bg=bg, fg=text_color, font=main_font).grid(
            row=row, column=0, sticky="w", pady=3,
        )

        entry_wrap = ttk.Frame(parent, style="TFrame")
        entry_wrap.grid(row=row, column=1, pady=3, padx=(10, 0), sticky="ew")
        entry_wrap.columnconfigure(0, weight=1)

        entry = tk.Entry(entry_wrap, relief="solid", bd=1)
        entry.insert(0, default)
        entry.grid(row=0, column=0, sticky="ew")
        ToolTip(entry)
        entries_dict[key] = entry

        browse_kind = BROWSE_KIND.get(key)
        if browse_kind:
            icon = BROWSE_ICONS[browse_kind]
            ttk.Button(
                entry_wrap,
                text=icon,
                style="Browse.TButton",
                width=3,
                command=lambda e=entry, k=browse_kind: browse_path(e, k),
            ).grid(row=0, column=1, padx=(5, 0))

    # --- Root Window Setup & Theming ---
    root = tk.Tk()
    workflow_label = "Two-Photon" if workflow == WORKFLOW_2P else "Electrophysiology"
    root.title(f"Neuron Analysis Toolkit — {workflow_label}")

    def on_closing():
        root.quit()      
        root.destroy()   
        os._exit(0)      

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.geometry("1600x1000")

    try:
        icon_base64 = ''
        app_icon = tk.PhotoImage(data=icon_base64)
        root.iconphoto(True, app_icon)
    except Exception as e:
        print(f"Failed to load custom icon: {e}")
    
    bg_color = "#F0F3F7"        
    frame_color = "#FFFFFF"     
    text_color = "#2B3A4A"      
    primary_btn = "#0078D4"     
    success_btn = "#107C41"     
    danger_btn = "#D13438"      
    
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

    root.configure(bg=bg_color)

    # --- View Menu ---
    menubar = tk.Menu(root)
    view_menu = tk.Menu(menubar, tearoff=0)

    def toggle_terminal():
        if frame_log.winfo_ismapped():
            root_vpaned.forget(frame_log)
        else:
            root_vpaned.add(frame_log, weight=1)
            show_terminal_half()

    left_panel_visible = [True]

    def toggle_left_panel():
        if left_panel_visible[0]:
            paned_window.forget(container_left)
            left_panel_visible[0] = False
        else:
            paned_window.insert(0, container_left, weight=0)
            left_panel_visible[0] = True

    view_menu.add_command(label="Toggle Left Panel", command=toggle_left_panel)
    view_menu.add_command(label="Toggle Terminal", command=toggle_terminal)
    menubar.add_cascade(label="View", menu=view_menu)
    root.config(menu=menubar)

    root_vpaned = ttk.PanedWindow(root, orient=tk.VERTICAL)
    root_vpaned.pack(fill=tk.BOTH, expand=True, padx=15, pady=(15, 0))

    content_root = ttk.Frame(root_vpaned)
    root_vpaned.add(content_root, weight=3)

    status_frame = ttk.Frame(content_root, style="TFrame")
    status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=0, pady=(6, 0))
    status_var = tk.StringVar(value="Ready")
    ttk.Label(status_frame, textvariable=status_var, font=main_font).pack(side=tk.LEFT)
    progress_bar = ttk.Progressbar(status_frame, mode="determinate", length=260, maximum=100)
    progress_bar.pack(side=tk.RIGHT, padx=(8, 0))

    paned_window = ttk.PanedWindow(content_root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True)

    frame_log = ttk.LabelFrame(root_vpaned, text="Terminal", padding=10)
    log_scroll = ttk.Scrollbar(frame_log)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text_log = tk.Text(
        frame_log, height=10, bg="#1E1E1E", fg="#CCCCCC",
        font=("Consolas", 10), yscrollcommand=log_scroll.set, relief="flat",
    )
    text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.config(command=text_log.yview)
    sys.stdout = RedirectText(text_log)
    sys.stderr = RedirectText(text_log)
    root_vpaned.add(frame_log, weight=1)

    def _set_initial_terminal_height():
        root.update_idletasks()
        total_height = root_vpaned.winfo_height()
        if total_height > 120:
            root_vpaned.sashpos(0, int(total_height * 0.82))

    root.after(250, _set_initial_terminal_height)

    def clamp_sash(event):
        max_left_width = 500
        if paned_window.sashpos(0) > max_left_width:
            paned_window.sashpos(0, max_left_width)
    paned_window.bind("<B1-Motion>", clamp_sash)

    # --- Left Panel ---
    container_left = ttk.Frame(paned_window, style="TFrame")
    paned_window.add(container_left, weight=0) 

    canvas_left = tk.Canvas(container_left, bg=frame_color, highlightthickness=0, width=420)
    scrollbar_left = ttk.Scrollbar(container_left, orient="vertical", command=canvas_left.yview)
    canvas_left.configure(yscrollcommand=scrollbar_left.set)

    scrollbar_left.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_left = ttk.Frame(canvas_left, style="TFrame")
    frame_id_left = canvas_left.create_window((0, 0), window=frame_left, anchor="nw")

    # --- Right Panel ---
    container_right = ttk.LabelFrame(paned_window, text="Visualization Panel", padding=5)
    paned_window.add(container_right, weight=1) 

    canvas_right = tk.Canvas(container_right, bg=frame_color, highlightthickness=0)
    scrollbar_right = ttk.Scrollbar(container_right, orient="vertical", command=canvas_right.yview)
    canvas_right.configure(yscrollcommand=scrollbar_right.set)

    scrollbar_right.pack(side=tk.RIGHT, fill=tk.Y)
    canvas_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    frame_plot = ttk.Frame(canvas_right, style="TFrame")
    frame_id_right = canvas_right.create_window((0, 0), window=frame_plot, anchor="nw") 

    def debounce_resize(canvas, frame_id, event):
        if hasattr(canvas, '_resize_timer') and canvas._resize_timer:
            canvas.after_cancel(canvas._resize_timer)
        canvas._resize_timer = canvas.after(50, lambda: canvas.itemconfig(frame_id, width=event.width))

    canvas_left.bind("<Configure>", lambda e: debounce_resize(canvas_left, frame_id_left, e))
    canvas_right.bind("<Configure>", lambda e: debounce_resize(canvas_right, frame_id_right, e))

    def on_frame_configure(canvas, event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    frame_left.bind("<Configure>", lambda e: on_frame_configure(canvas_left, e))
    frame_plot.bind("<Configure>", lambda e: on_frame_configure(canvas_right, e))

    def _on_mousewheel_left(event): canvas_left.yview_scroll(int(-1 * (event.delta / 120)), "units")
    def _on_mousewheel_right(event): canvas_right.yview_scroll(int(-1 * (event.delta / 120)), "units")
    
    container_left.bind("<Enter>", lambda _: canvas_left.bind_all("<MouseWheel>", _on_mousewheel_left))
    container_left.bind("<Leave>", lambda _: canvas_left.unbind_all("<MouseWheel>"))
    container_right.bind("<Enter>", lambda _: canvas_right.bind_all("<MouseWheel>", _on_mousewheel_right))
    container_right.bind("<Leave>", lambda _: canvas_right.unbind_all("<MouseWheel>"))

    # --- 1 · Gabor filter bank ---
    frame_gabor = ttk.LabelFrame(frame_left, text="1 · Gabor Filter Bank", padding=15)
    frame_gabor.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_gabor.columnconfigure(1, weight=1)

    gabor_entries = {}
    for i, (label, default) in enumerate(gabor_param.items()):
        add_config_row(frame_gabor, label, default, gabor_entries, i, frame_color, GABOR_LABELS)

    btn_submit_gabor = ttk.Button(
        frame_gabor,
        text="Build Gabor Library",
        style="Primary.TButton",
        command=run_in_thread(create_gabor, "Gabor library construction"),
    )
    btn_submit_gabor.grid(row=len(gabor_param), column=0, columnspan=2, pady=(15, 0), sticky="ew")

    # --- 2 · Stimulus wavelet pipeline ---
    frame_processing = ttk.LabelFrame(frame_left, text="2 · Stimulus Wavelet Pipeline", padding=15)
    frame_processing.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    ttk.Label(
        frame_processing,
        text="Downsample the movie, then project it onto the Gabor bank (coarse wavelets for RF analysis).",
        wraplength=380, background=frame_color, foreground=text_color, font=main_font,
    ).pack(anchor="w", pady=(0, 8))

    btn_submit_wavelet = ttk.Button(
        frame_processing,
        text="Run Wavelet Decomposition",
        style="Primary.TButton",
        command=run_in_thread(run_wavelet, "Stimulus wavelet decomposition"),
    )
    btn_submit_wavelet.pack(fill=tk.X, pady=3)

    ttk.Separator(frame_processing, orient="horizontal").pack(fill=tk.X, pady=8)
    ttk.Label(
        frame_processing,
        text="Optional: compress full-resolution wavelet arrays to Zarr for the high-detail full model (lower RAM).",
        wraplength=380, background=frame_color, foreground=text_color, font=main_font,
    ).pack(anchor="w", pady=(0, 8))

    btn_convert_zarr = ttk.Button(
        frame_processing,
        text="Convert Wavelets NPY → Zarr",
        style="Primary.TButton",
        command=run_in_thread(convert_to_zarr_gui, "Wavelet Zarr conversion"),
    )
    btn_convert_zarr.pack(fill=tk.X, pady=3)

    # --- Experiment configuration ---
    frame_params = ttk.LabelFrame(
        frame_left,
        text=f"Experiment Configuration ({workflow_label})",
        padding=15,
    )
    frame_params.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_params.columnconfigure(1, weight=1)

    param_entries = {}
    for i, key in enumerate(workflow_param_keys):
        default = filtered_param_defaults.get(key, "")
        add_config_row(frame_params, key, default, param_entries, i, frame_color, ANALYSIS_LABELS)

    # --- 3 · Neural & RF analysis ---
    frame_analysis = ttk.LabelFrame(frame_left, text="3 · Neural & RF Analysis", padding=15)
    frame_analysis.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    ttk.Label(
        frame_analysis,
        text="Uses coarse wavelets + aligned spikes to estimate population retinotopy and per-neuron tuning.",
        wraplength=380, background=frame_color, foreground=text_color, font=main_font,
    ).pack(anchor="w", pady=(0, 8))

    btn_submit_plot = ttk.Button(
        frame_analysis,
        text="Run Coarse RF Analysis",
        style="Success.TButton",
        command=run_in_thread(plot_data, "Coarse receptive-field analysis"),
    )
    btn_submit_plot.pack(fill=tk.X, pady=(0, 10))

    ttk.Separator(frame_analysis, orient="horizontal").pack(fill=tk.X, pady=8)

    rf_wrap = ttk.Frame(frame_analysis, style="TFrame")
    rf_wrap.pack(fill=tk.X, pady=(0, 10))
    rf_wrap.columnconfigure(1, weight=1)

    tk.Label(
        rf_wrap, text=FIELD_LABELS["Neuron ID"], bg=frame_color,
        fg=text_color, font=bold_font,
    ).grid(row=0, column=0, sticky="w", pady=3, padx=(0, 10))

    entry_neuron = tk.Entry(rf_wrap, relief="solid", bd=1)
    entry_neuron.insert(0, '1173')
    entry_neuron.grid(row=0, column=1, sticky="ew")
    param_entries['Neuron ID'] = entry_neuron

    btn_runRF = ttk.Button(frame_analysis, text="Inspect Single Neuron", style="Primary.TButton")
    btn_runRF.pack(fill=tk.X)

    # --- 4 · Export ---
    frame_export = ttk.LabelFrame(frame_left, text="4 · Export", padding=15)
    frame_export.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    btn_save_ret = ttk.Button(frame_export, text="Export Retinotopy Matrix", style="Primary.TButton", command=click_save)
    btn_save_ret.pack(fill=tk.X, pady=3)

    btn_export = ttk.Button(frame_export, text="Export Plots as SVG", style="Primary.TButton", command=export_plots)
    btn_export.pack(fill=tk.X, pady=3)

    # --- Global Controls ---
    frame_controls = ttk.Frame(frame_left, style="TFrame")
    frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    btn_quit = ttk.Button(frame_controls, text="Quit Application", style="Danger.TButton", command=quit_app)
    btn_quit.pack(fill=tk.X, pady=(15, 5))

    all_buttons = [btn_submit_gabor, btn_submit_wavelet, btn_submit_plot, btn_runRF, btn_save_ret, btn_export, btn_convert_zarr, btn_quit]

    root.mainloop()