"""
Created on Wed Mar 25 19:31:32 2025

@author: Sophie Skriabine
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import threading
import sys
import gc

# --- DPI Awareness ---
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
    myappid = 'neuro.gabor.toolkit.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
except Exception:
    pass

from .config import parse_literal
from .WaveletGenerator import *
from .LoadPinkNoise import *
from .Analysis_Utils import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import numpy as np


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

def _parse_dirs(value):
    try:
        parsed = parse_literal(value, "Dirs")
    except ValueError:
        parsed = value
    if parsed is None: return []
    if isinstance(parsed, (list, tuple)): return [str(path) for path in parsed]
    return [str(parsed)]

def run(param_defaults, gabor_param):
    
    class RedirectText:
        def __init__(self, widget): self.widget = widget
        def write(self, string):
            self.widget.insert(tk.END, string)
            self.widget.see(tk.END)
        def flush(self): pass

    global loading_modal
    loading_modal = None

    def set_ui_state(state):
        global loading_modal
        for btn in all_buttons:
            btn.config(state=state)
        
        if state == tk.DISABLED:
            loading_modal = tk.Toplevel(root)
            loading_modal.overrideredirect(True)
            loading_modal.attributes("-topmost", True)
            
            w, h = 320, 120
            x = root.winfo_x() + (root.winfo_width() // 2) - (w // 2)
            y = root.winfo_y() + (root.winfo_height() // 2) - (h // 2)
            loading_modal.geometry(f"{w}x{h}+{x}+{y}")
            
            loading_modal.configure(bg="#FFFFFF", highlightbackground="#0078D4", highlightcolor="#0078D4", highlightthickness=2)
            tk.Label(loading_modal, text="Processing Data...", font=("Segoe UI Variable Display", 11, "bold"), bg="#FFFFFF", fg="#2B3A4A").pack(pady=(25, 10))
            
            pb = ttk.Progressbar(loading_modal, mode='indeterminate', length=250)
            pb.pack()
            pb.start(10)
            root.update()
        else:
            if loading_modal:
                loading_modal.destroy()
                loading_modal = None

    def run_in_thread(func):
        def wrapper(*args, **kwargs):
            set_ui_state(tk.DISABLED)
            def thread_target():
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}")
                finally:
                    root.after(0, lambda: set_ui_state(tk.NORMAL))
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
        print("Gabor Library Created Successfully.")

    def run_wavelet():
        sigmas = parse_literal(gabor_entries["Sigmas"].get(), "Sigmas")
        visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
        analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")
        movpath = param_entries["Movie Path"].get()
        lib_path = param_entries["Library Path"].get()
        nx = int(param_entries["NX"].get())
        ny = int(param_entries["NY"].get())

        if (visual_coverage != analysis_coverage):
            visual_coverage = np.array(visual_coverage)
            analysis_coverage = np.array(analysis_coverage)
            ratio_x = 1 - ((visual_coverage[0] - visual_coverage[1]) - (analysis_coverage[0] - analysis_coverage[1])) / (visual_coverage[0] - visual_coverage[1])
            ratio_y = 1 - ((visual_coverage[2] - visual_coverage[3]) - (analysis_coverage[2] - analysis_coverage[3])) / (visual_coverage[2] - visual_coverage[3])
        else:
            ratio_x = ratio_y = 1
            
        parent_dir = os.path.dirname(movpath)
        downsample_video_binary(movpath, visual_coverage, analysis_coverage, shape=(ny, nx), chunk_size=1000, ratios=(ratio_x, ratio_y))
        videodata = np.load(movpath[:-4] + '_downsampled.npy')
        videodata = videodata.astype(int) - np.logical_not(videodata).astype(int)
        waveletDecomposition(videodata, 0, sigmas, parent_dir, lib_path)
        waveletDecomposition(videodata, 1, sigmas, parent_dir, lib_path)
        print("Wavelet Transform Done.")

    def embed_interactive_figure(fig, parent_container):
        frame = ttk.Frame(parent_container, style="TFrame")
        # Notice we removed fill=tk.BOTH and expand=True
        frame.pack(side=tk.TOP, pady=15) 
        
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        
        toolbar = NavigationToolbar2Tk(canvas, frame)
        toolbar.update()
        
        tk_widget = canvas.get_tk_widget()
        # Just pack it to the top so it centers itself naturally
        tk_widget.pack(side=tk.TOP)

        return canvas
    
    def plot_data():
        # Read parameters early (must be done carefully in thread, but safe here)
        try:
            dirs = _parse_dirs(param_entries["Dirs"].get())
            exp_info = parse_literal(param_entries["Experiment Info"].get(), "Experiment Info")
            sigmas = np.array(parse_literal(param_entries["Sigmas"].get(), "Sigmas"))
            visual_coverage = parse_literal(param_entries["Visual Coverage"].get(), "Visual Coverage")
            analysis_coverage = parse_literal(param_entries["Analysis Coverage"].get(), "Analysis Coverage")
            n_planes = int(param_entries["Number of Planes"].get())
            block_end = int(param_entries["Block End"].get())
            nx = int(param_entries["NX"].get())
            ny = int(param_entries["NY"].get())
            ns = len(sigmas)
            resolution = float(param_entries["Resolution"].get())
            spks_path = param_entries["Spks Path"].get()
            nb_frames = int(param_entries["Number of Frames"].get())
            movpath = param_entries["Movie Path"].get()
            screen_ratio = abs(visual_coverage[0] - visual_coverage[1]) / nx
            xM, xm, yM, ym = analysis_coverage
        except Exception as e:
            print(f"Invalid input: {e}")
            return

        pathdata = os.path.join(dirs[0], exp_info[0], exp_info[1], str(exp_info[2]))
        pathsuite2p = os.path.join(pathdata, 'suite2p')
        deg_per_pix = abs(xM - xm) / nx
        sigmas_deg = np.trunc(2 * deg_per_pix * sigmas * 100) / 100
        
        # --- Heavy lifting done in background thread ---
        if spks_path.strip().lower() in ("", "none", "null"):
            spks, spks_n, neuron_pos = loadSPKMesoscope(exp_info, dirs, pathsuite2p, block_end, n_planes, nb_frames, threshold=1.25, last=True, method='frame2ttl')
            neuron_pos = correctNeuronPos(neuron_pos, resolution)
            neuron_pos[:, 1] = abs(neuron_pos[:, 1] - np.max(neuron_pos[:, 1]))
        else:
            try:
                spks = np.load(spks_path)
                parent_dir = os.path.dirname(spks_path)
                neuron_pos = np.load(os.path.join(parent_dir, 'pos.npy'))
            except Exception as e:
                print(f"File not found: {e}")
                return

        respcorr = repetability_trial3(spks, neuron_pos, plotting=False)
        skewness = np.array(compute_skewness_neurons(spks, plotting=False))
        filter_mask = np.logical_and(respcorr >= 0.2, skewness <= 20)

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
        rfs_gabor = PearsonCorrelationPinkNoise(w_c_downsampled[:n_frames].reshape(n_frames, -1),
                                                np.mean(spks[:, :n_frames], axis=0),
                                                neuron_pos, 27, 11, ns, analysis_coverage, screen_ratio, sigmas_deg,
                                                plotting=False) # MUST BE FALSE HERE

        # --- Plotting Phase: MUST strictly execute on the Main UI Thread ---
        def render_gui_plots():
            # Clear old plots safely
            for widget in frame_plot.winfo_children(): widget.destroy()
            plt.close('all')

            fig1, ax1 = plt.subplots(figsize=(6, 6), constrained_layout=True)
            ax1.scatter(neuron_pos[:, 0], neuron_pos[:, 1], c='k', alpha=0.3, label="Neurons", picker=True, rasterized=True)
            ax1.set_title("Neuron Positions (um)")

            fig2, ax2 = plt.subplots(figsize=(10, 3), constrained_layout=True)
            ax2.set_title("Spike Train")

            fig3 = plt.figure(figsize=(8, 10), constrained_layout=True)
            gs = fig3.add_gridspec(3, 2) # 3 rows, 2 columns
            
            ax3_0 = fig3.add_subplot(gs[0, :])  # Top row, spanning both columns (2D)
            ax3_1 = fig3.add_subplot(gs[1, 0])  # Middle row, left (Elevation)
            ax3_2 = fig3.add_subplot(gs[1, 1])  # Middle row, right (Azimuth)
            ax3_3 = fig3.add_subplot(gs[2, 0])  # Bottom row, left (Orientation)
            ax3_4 = fig3.add_subplot(gs[2, 1])  # Bottom row, right (Size)
            
            # Group them back into an array so the plotting logic stays exactly the same
            ax3 = [ax3_0, ax3_1, ax3_2, ax3_3, ax3_4]

            def onpick(event):
                try:
                    neuron_id = event.ind[0]
                    entry_neuron.delete(0, tk.END)
                    entry_neuron.insert(0, str(neuron_id))

                    ax2.clear()
                    ax2.plot(np.mean(spks[:, :, neuron_id], axis=0), label=f"Neuron {neuron_id} Spike Times")
                    ax2.legend()
                    canvas2.draw()

                    rf2d, x_tuning, y_tuning, ori_tun, s_tuning = PlotTuningCurve(rfs_gabor, neuron_id, analysis_coverage, sigmas_deg, screen_ratio, show=False)
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
                    print(f"Error drawing pick event: {e}")

            fig1.canvas.mpl_connect('pick_event', onpick)

            fig10, ax10 = plt.subplots(1, 4, figsize=(12, 4), constrained_layout=True)
            maxes1 = rfs_gabor[2]
            plt.rcParams['axes.facecolor'] = 'none'
            
            m = ax10[0].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=maxes1[0], cmap='jet', alpha=filter_mask, rasterized=True)
            fig10.colorbar(m, ax=ax10[0])
            ax10[0].set_title('Azimuth')
            
            m = ax10[1].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=maxes1[1], cmap='jet_r', alpha=filter_mask, rasterized=True)
            fig10.colorbar(m, ax=ax10[1])
            ax10[1].set_title('Elevation (deg)')
            
            m = ax10[2].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=maxes1[2], cmap='hsv', alpha=filter_mask, rasterized=True)
            fig10.colorbar(m, ax=ax10[2])
            ax10[2].set_title('Orientation')
            
            m = ax10[3].scatter(neuron_pos[:, 0], neuron_pos[:, 1], s=5, c=maxes1[3], cmap='coolwarm', alpha=filter_mask, rasterized=True)
            fig10.colorbar(m, ax=ax10[3])
            ax10[3].set_title('Size (deg)')

            # Deploy to Tkinter
            global canvas2, canvas3
            
            embed_interactive_figure(fig1, frame_plot)   
            canvas2 = embed_interactive_figure(fig2, frame_plot) 
            canvas3 = embed_interactive_figure(fig3, frame_plot)  
            embed_interactive_figure(fig10, frame_plot)  
            
            canvas_right.configure(scrollregion=canvas_right.bbox("all"))

            # Update RF button command so it references the newly built plots
            def click_RF():
                try:
                    neuron_id = int(param_entries["Neuron ID"].get())
                    ax2.clear()
                    ax2.plot(np.mean(spks[:, :, neuron_id], axis=0), label=f"Neuron {neuron_id} Spike Times")
                    ax2.legend()
                    canvas2.draw()

                    rf2d, x_tuning, y_tuning, ori_tun, s_tuning = PlotTuningCurve(rfs_gabor, neuron_id, analysis_coverage, sigmas_deg, screen_ratio, show=False)
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

        # PUSH Plot creation strictly onto the Tkinter Main thread
        root.after(0, render_gui_plots)

    def click_save(): print("Save Retinotopy functionality triggered.")
    def export_plots(): print("Exporting SVGs...")
    def quit_app():
        root.quit()
        root.destroy()

    def popup_browser(entry_widget):
        menu = tk.Menu(root, tearoff=0)
        def set_file():
            p = filedialog.askopenfilename()
            if p:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, p)
        def set_folder():
            p = filedialog.askdirectory()
            if p:
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, p)
        
        menu.add_command(label="Select File...", command=set_file)
        menu.add_command(label="Select Folder...", command=set_folder)
        x = root.winfo_pointerx()
        y = root.winfo_pointery()
        menu.tk_popup(x, y)

    # --- Root Window Setup & Theming ---
    root = tk.Tk()
    root.title("Neuron Analysis Toolkit")

    def on_closing():
        root.quit()      # Stops the Tkinter mainloop
        root.destroy()   # Destroys the GUI window
        os._exit(0)      # Force-kills the Python process and all background threads

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    root.geometry("1600x1000")

    try:
        # Replace the string below with your actual massive base64 string!
        icon_base64 = "iVBORw0KGgoAAAANSUhEUgAAAkUAAAJFCAYAAADTfoPBAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAAJcEhZcwAAEnQAABJ0Ad5mH3gAAO48SURBVHhe7J3tku24bmR1+tXnvX3mhw077+oECFLUx67aK0IhIpGEVAVIpe6+nvnz//7f//t7fPlf/vz5Q2lIZ0/H8zb+/h2PRuWpcsEOzygfdH076fR9xcPYaYydxtgx8ozyR9OTMerb2fyReKiNYqcxdnQ8x4TvLDO96njv8pzNv5VR30f5jNV9P51/KHz58kl8H+w+V/xR6NTseM4wqj/KfwJ3zvnMtWa8V/ITevzlHXw/ir78eP7+/bv88l7dF8S1qzpVLuh4Klb+aPz586fcV+WCkWd0jWPCUzHKO2b2uB4zdlSezux0WNk/c+2O58uXT+H7UfQlZeaPwk9D/yDM/IGoqPbP1p/xZlT9rXJHI380PJofeY+GZ/TxtJq7iqqHVa4DZ/ZsvU/nif5++Uy+H0VfSn7jy6T6A1LlOpzdr+ysNUNnJjoesrJnJ3de/8reZbUz/cuXL//H96Poy49m9g9Bx6//BP7b6Hw4jDzVv9HJ9KDa+yaemI/OXIZn5FNmvF++fDqv/yiafYB/E/EH4uzxqYzmYpQnK/6V+az8q7kZRn2vciNGe0f5Y4NnNXc08h1Gfarys/O0OoPH4D7IyDvKvxm+D1eOL57V2XySV38U6S/z036xV7PzQdxZ6y3Mzsusn5zd3+Xq61SzUOWORn6Gs7Wq/VXuS87Vs/cEu2ZhV52fxKf+/X7tR5H7JTrtN3LFA3hFzU9h11zN/FPRjPcuqhmockczP/KQkX9UczX3FN15iNnp+it21XkzWa8zfZXRPP4m3Ew57Y288qOo+uXtfCF8Ik88dE9c8wwzszHj7XJ2Rs/svYJR/8/mKzp7K0+Ve4LVvp6dqRFX1X2aJ/r/2z+Oqlmqcm/hdR9Fn/BLe4rf+KBdOQ9X1g7uuMaT3DGTd1zjDqpZqHJfvnwKnTnueJ7kVR9Fb/9lPclP+cPwFrqztuOfzqsamf4JjGay80/MHc/RvNanks1AzE2W79Ld3/F1PJ/AHfNyxzXewuycznjv5jUfRSu/pJU9n8hdD9dd13ma7tyob/ah38GZ6832MvPP6sEof8Az658h+/By2lVkvcz0s8S8Rv2rrtPhyWs/TTZ7X947F6/4KDrzyzmz9xP4PlCe1b5392W+TO+Q7c30u8hmbFafwdVwGqk8VS5jZs+M90myecp0pePJqPZWuS+fzZnentl7FY9/FO34peyo8UY+5SX80xjN098T/9Yo25fpO9A5umOmRteo8lUuqDxVbgdRP+tXps+wOl+dfaP80fRknNn75fPY0e8dNXby+EfRly87qR6wKhd0PEH8EZrZcxTXcPpK/Rmyj4hZ/fifXJU/BvuDs54s19UZn8X1L+ur0ypWZrDjrTyz19tB1ZMsl+lP8KZ7+VLz/Sh6KU89RNl1Z/VjkLuC6kVd5YKOJ+PM3lnuvJZytp8z+zveylPlOmT7Z3/3s/4ZztTu7O147iDrxSq763V56rqfwFtm7fh+FL2Tpx+e7Pqz+jHIVVT/NOp0pwVVLuh4RlT3fBV6vSuvnfXxT+P/q4NRPqOzr/Jk+irZ7zfTr2LXnHVqdDxvJOt9pt/F09d/M2+ZtUc/it7yS7iS6qXtmPE+QXZ/mX4McrPMzkzH3/HMEH+0qrpVbgdn6nf71fHt8IzyR9NzNJ/HUe/OktXO9KM5Uyt06mWeWX03WS+d9iZm72/W/+Ucj30U3fXgfBKfPvxP3X82S5mudDxnqOq7nNOOQu8ws7fbw45vl+do+pzHaZ/GTP9W6NTveEbsqPFT+Alz+VN55KPo+3D8m7c9JG+7n4xsljJd6Xo6voodNZ6Es8BY+TPxn9PU4zTS9byFmZ4776656dTY5dlN1s9Mr1jZcyVvu5838MSMkds/it7wQ7+NT3s4rr7fbEYynXR8s55df6DIFTWfojMXd3o+hStmIOY1anfmd5R3ZHWddgefNhefdr938NTsBLd+FD39w76RNz8U1b1luUx/E505zDz8YzNDd0/XdwfsJ+PQnK50PErHn3moMX6Sbm+7PrI6m8Fo7yivzHhnWOnnyp67ePO9PcWZ9+xZbvsoeuKHezuf8DCs3OPKnhVWZqqzp+M5JnxKd0/XdyWdPu7yZHT27vJczdU97dTf5VnlbO2qj1ku09/EJ9zjU5ydmVku+yjSL727f6i3UA16lZvhj/zvLFaPVc7svYpq1qpc0PEoO+b77P5dzPaz4x95OjPY9cww6+8QfZyZia4vY+Ud2/FWHnc9xhUdb6fnyoyX6Ltw9dhBVafK/QbczF3FJR9Fd938G3hyWHdde1Snymc56oxn6cxU5alyQceTMfPHyXmcRtTT8e+m28POHwvmGTs6nqDr28WoH6O80vF2Zy2js7fjGbGjRpD1NNM7nNmr7KpTccc13s7OecrY/lF0x02/hSeHdPe1z9Tr7u36yNmZ6uzveN7AUx9G7B3jINOVylPlgsxDnfHbcP1zGul4OnTqdDxPMupxla9yK+yu1+Wp6z7F1TO59aPo6pv9KZwd4rP7V/gz+Kf0KncFnDXGQaYrHc8Mfyf+rZFCP+M7qfqZ5TJdudqT6ccgdzezvV2dqRGdeh3PIfd4F6N+jvJXcPaao/dsxsqeT+bKOdv2UXTlTX75P64c/itrd9k5R51aHc8Zsj9mjO9i5bqjuei8yDseJfzVnlH+aNz7WVZ+nxmulpudWUb7R/kj8TjtLs729ez+iitrf/k/rpq/LR9FV93cJ3PFg3FFzTfRnaOOb6en41thV91ddVbozGTHc4ar62dc/XvfUb9bo+PreLrsrOV4aiaCp6//W7hijk5/FF1xU5/OFQ/EFTVXqO6jyq3C+WKcaSM6e9Tz98KPox2s3NtKv3RPZ//I86f5b3s6Hrf+rXBeGTtG+S6s07l2Raf/yoz3St5yHz+dM7PlOP1R9GWOlQdlZc8qnWt1PMeE7wzugXCaMsofhSde8Fm+wu2hNorPkvUk01eIP2JVTeYZO0aeKvck7CFjpzHu0JnNKndIjQzmGD9NZwY6nl2sXivbl+lf9vH9KLqRlYFe2XOWzjUzD3XGT9N5iXc8x4RPuWvPDFWPNJetlUxXKk+VC2Y9s+uddHrX8YzYUWOVu6+d9SrTlY5nN09c87excwZPfRTtvJG38Kaf6VMfprfed6e3HY/yV/7pfHbvG/kj/zYm+2jI+pvpytWe7D5n12/nzNx19ozyT5H1KNOVjucqnry24639PcOun2npo6jzUN3N2+7nLE8/RJ3rdzxX4nrutKPQD/yBOYvWympSczG1N/On+M9akas8js6+UX6V3fUU9tXFmVbN1AqjWlm+q335T66cqxl29WpXnbex9FH05Vre8vC85T66ZA9pph+D3C461+h43kg1I1Uu6H7UdDw7ueJ6Kz1e2TPLyjVGe0b538oVc/VlL9+PopvoPgxd312M7qf7Ry2Y8ZKVF221p8od/5PPjlm4Z7bOjJe433lomjuzVkYzEXnWokYqj+qsO1rvRPs02zP6GXfgnPLIWM1VrO6b7U3H3/HcxZP3stoTZTRLT7LjvqY/inZc9CrefG8dnnxYfiLVPGS5zh+QI/njM2LkGeWDri9jdc6qj4r4KKGujPJBx5flM71i9HNdRdXHKhdw/mb2ODL9GOz7Mk9nrjqeu/mEGTh7j1MfRWcvdgc77nFHjVne+AB8MlUPs1ymd+ns73iexn0kOC2LHR0PGe2p8tn9VnsyVvaM4BxozJyj46nI9mf6W+n0puN5gs59dTxf/s2ZOW5/FJ25yN38nfgnp92sDPHKni//SbfXmS/TZ9HZy+ZQY+aUKncVsx8QlefP4N8eab7jcYxyo/VdVL2s5oFz5OZplaxOppOu7yo6fex4nuTt9/fJrM5n66NotfgbePu9f8JD0bnHylPlVqn6muVW9OyYpbun65vFfTx0PxQyX7bHXSuInMtXuaNR160rRr5RfhXtcdXvKlfBWa1mdpd+FVf14E3M/oyz/rOMZujNrNzz8KNopejbeOvPcPdwfxo7+5bVmtXPoDWz9ROszGG2J9OPQY5U3iyX6RWxp7O343maapay3KyeMev/8n+8dbZ+Qk9nf4Z/fRT9bfzTxSfytp9l10PwB/8JYuV4Izv7ldWa1RU+J93nRT2d9Spn+1rtd7lqlq7MMR8xz1w7Rvmj6VllZgY4dyvzp8zqGbP+GWZ/9yO/ztCO4yw7amR050OZ9b+ZmefkPz6KRuZPZ+bnm/FW7HpgyM66nVqj/DH5B2gnVa+y3KzehQ/far3YpzXO1AuqvlS5I8lnsxO6yx1mX+WvckdyX0ehHydyXbRPrn8zveQ8dfdlZPtn9WPy59jJjh7t5o33tMoTPb2T6uf734+iyvRb+f5O/pM3PvTsEWNH5sn0s1xV9yr4sUJtxMg7yldke909Z3GHlT2B9rvb+2xPd/8sWd1MVzqepznTvzM8dd2dfEJ/d5D9nP8cRfLLO7nqwevU7XieojPHmYf6X/NP53rMons666vJPnaqjwtqo7zyZ/Bve5SRN9Mrqj1V7g6yGViZB84pD3odma50PGRlzwqdfnY8q+yuvbvel//DzeQ/TvzJvOHnXR3y6g/Fl3/T7TV9jB38Y+P+6JBRfjfZvHS1gDlX12mVfkhu5HFUexR6Rvuq3NXofHRmhbPX3VPFGd36X/6b1Tla3fdlHc71v/6H1r8B/hK+vJ/ZnnX99DHeTdTX62Tru+i+iDPfrH4kOacdhX4gl60zjfETrPZ7dd9h9jLOtE+g09OOZwer11ndt4NP7ftZ9Of+lR9Fn8iTD8rT8EFl3IX7RvEKf80/vWd1Vc88T8F5Yxys6FnuKPKZfhQfQ7HO9h2DnNL1dRj1mnPhjrOwBuNM+3R29rHD3dc7w0/s9wzx8//aj6JPGYDqj8EVdK43yl8J/2C4NXF/SEZxaNkxS/dela5vRNYv6i52mltHnO1xeuD0bE+lVzDvYmo76fZyZU4Uzmk1r9QZZ9oMZ/f/BK6cqy97+fv37+/9KPpSM3qQR/mfTuePzk7uuMYZsnmodP0QqT5KurrW2sXOWkR7utJfzmBVo8rNkN3zrvo/lSvn6Mtevh9Fi6wO+cy+Ge+X/2PHC3q2xqw/WN33JrI5ndWPIjerZ8z6j8aerIeZfmz4oLhqT8czYkeNFUZ9GuWvZuf1d9b68p98P4oKZgbvj/kn3VHsiDod79O8+R7/mn96psZYtRW0nqvjNKXKZazsUdhDN3suDo055kd6pc3obu1ipzF+Cs4hc9lczcC9jJ3G+I28pYcj3Pw66GHsNMZK97pfvh9FKdUAVbkzXFV3lbfdT/ZHI1sr1BlXnPljlO3J9GOQO8NMP52XGmOnxcs407NY9Q70MX4Ls709M3fZ3I5ip2ncWb+Nt83Dyv2s7Pkyz/ejaJLuYHZ9waz/Lt56X0HnpZzpxPmoMe6Q7cn0Y5Bb4WwfO/vpYRxk+jHIKR1fx3Mn3Z6Gr+tfgbUZOy171rL1W3jbHARX3NcVNX8bv/ajqBqeKuegn/GIWb/yR/5J/OzxRrov2bMvZrfHacf/6Hp0qHxV7i66/c981BkH1N3subjjC2b1jBl/NguZHjDHuAPn0dVw+ih2GuMMd723oXN19ljlzN4j2e+0oMp9+W9+7UfRCmcGKtub6RU7HsYZ7rrOLJ2XrvNQY5xpGfrHqLPPeZx2FPoqrpdOC7Kc6vQwDo17OjFruTg05oJZ/WrO9nR23hT6R3GmHdAzzzHI/RSyeT2Dq+U0R9enrOz5iXw/ik5yZpBW9q7s+WmceclyL+NMm+HsfuLqOW2GzhzR4z4+Mk+lMVaqXFB5stysfjdn+3kGXpsxNZcfsbJnB0/0d/aalb/KBR1Pl521PpXLPor+mn+le+a4CzcUf5J/AnBal5W9K3tmuLp+hfa46vcoF3m35jVcLWq6l7kK58+0Kq6Y8SpZnyvd5agxdpqLncbYaVUczOpHcr0uKz1xe6jNzqD6uSfTqpiaywfdXOW7mtX+dpmtP+snM/vPzPcsnMPV404u+yjazd2/mFWuGrar6n4KVf+ZY9yF+xiHNvOgdn1vopq1LEddX7yxVo+LFcZOq2LmgkwPRvlV3BxQc/HsrGX+jjaKncaYuLzT7uSqHpPZ63T9u31XsbPPO2uN2P5RlD2UO7iq7tWcHc6z+2e481pdqr5XuQy3hxpjR8dzTPhG7KpT0e1/5wMkI/NTZ+yoPFku0+9kdy879VY8jB3h6Xif5u7e33098tT1r5iFv8VH/062fhRdfbPHTddQsqHKdNL1ZZzdv4K75p8L/pXrmV5yL+OAOmOnMa7oPqT0uH2jODSn76bqtea45pywjovpd3FozIema+aDTN/NXT06Fq5Fr9s/igOnO+0p7ur3iLP30d3f9V2Jm6cruPIaWz+K7uKKX0j1Mu3S3b/b9yRX3iP7PIqpce1i1nCx85Gu7zDXeIpO7yoPP0aYc1oWVzmnsb7zB1Uu6HiugLPQmaHZWaOPsdOqmLkdnKn5VO9m2H2PnP+Kru8sZ3q4wlXX2/JR5B68q7n7eoobMqedYXe9K5m5127fuj4le3F3a3V9I7I6mU66vqthXxkHma6seBg7jXFAnbGj4znLnb2tngHGDnqqemSU/4105qvjqTi7/9O4Ys5OfxRdcVNdnrj2zNDNeJWZfX/kPy+sHo5MfwOjvo/yDreH2t+T/4QeOnFexlfR6fPsnFDTmHPH2i4mTjuMzloZHc8OZnpKb2f21ON81FzstA7OR41xRtc3S9ZnnckzR4eOz3l2X+OY8HW4qmcddl/71EfR7ptZ4cp74NAwDjJ9hZlaM96KXXXuwPXbaYHLUXMxNRKejq9ilA+6vlW6MxA+53cvbo135ahx7fyOrm+Vbs+yOerO2DF5rSp2msZVrtLeys4Z6Nbq+jp0a3V9K3xSvzssfxS96Rdxxb1cNUS76u6qE7h6TnsbVe9djhrjK7jjGmdgnxmTUT5QH/cwdlq1P9Nm2VHjaWbni37GjspT5Z6i09eO5ydxxc/7xt6fZfmj6Ms8u4ZyV50OvBbjq+k+dPQxdhpjp/1t/NN619Oh67uCPxP/luUYfLTM5BgrVU411hkx491N1uNMd3RnLvNQG8VOY/wWXG+pMd7F7BxmnK1xdv+nsXMWvx9FJ3HDt/pgdPZ0PKtcWbsiBjp7gTsyX1fXmH88GGc+xygfdDxPErMwmmWX15hrxtmasVtXmsKfZeTfjet1NieZHozyR+Jh7DQXUxvpDqc77ScxmrFRPsPte2Km38iumfoxH0W7fiEz7BzETq2O59Po9q3jyzzUNWZuF64utYipK1Xuajhvo4+To/DQz9hp2X76qP2RPxI8383O/nVqVR6Xo8Z4lmz/rP4beGomd/FTe/djPoo+mc7D0fH8VLKHr6PTU+Wc9rf4p+IsFzgP45F+mHt6Av24cB8aqrucrkexkuXoCzqeN5H1NnQ3Q2TkcfNHTali5oJZXel4zvDUHHSu2/F8uZfvR9EHUD048UfFHSu4vaN4N9lL8qzOWOEfitCqfBC5LH8U1870Y5B7C6N5qdaMuzld0/tWql5ms+M0Jdt3mJl0PtWYdzG1kX41rvejONM66LzxyBjlv7yP70fRTWQPRqYHVb7KHY38Ga6sPaJ6mQfUGY/I/Jk+S1Wnyp1BX9B8WZ9ZK5knWyvUsz30hRZ6Zx2xW+8g62GmH4McqbxZzunVs8R4FrffabOc6dWZvRVn6mZ7M/3Ltfyoj6K/yT+1vJXR0Ff5KqfEH4Lq2EHU2VUvI+tvph9J7swfg2rOIpfllcpT5c6iPdIZWFm7Oi6XrfVQLfPr2nkZZ2t3PVdzJ92eZr7RbFW5wOVVY56xkuUyPRjlK67ojfa9OkZUnip3FPlMf5ozPXw7P+qj6CdRPQxVbgX30DOeYbTXvbxdTE3JctRdHJq7houdb5Q7TJ5nx2ruKXR23B+Qzlo11oq1eojTdrBS1/WbcC6Y03Osnfdo1HI5xoR5F1Or9KDK3YHrp9POUNXjs/GpPN3Hq/l+FG1mx9BXNarclTx13ZkHcMbr4P5RrFQ5xfmcRsITf3g6ew55Ee/oHz9OGJMs77wjRvuZd54ZVvZ3e6KM9lT5KkfoZUyyfKZXZHuoMz7LqIej/CpX1Q2urv/b+X4UPcTKYK/s2cmT13cvTKcdC/phcoyDv8UHyShXke112pO4Dw7G1Kq1HpknW3MfPU5z+YpZ/yquz047ilk5JOfy1DRmjozyI7r7uz7HXb3q8KZ7+TLH96PoZWQPU6VnuVlm68z6Vxm9KF0++8MQVH8QqlwQ9V0+0x0dX8ezg+inzlSss16r7rwza3fdbO32E7fvbqo5CSJXeQLnGV1DdXqqHKnyWS7TdzPb21l/RjVXu/Qv9/Lqj6LRw/4pdIc983X0eDjdscqZvatUvXY5pzmylz/3u9xoDiu9ymXode8ies2ZIurLvG72RmunuTXjLFfh7u8uOj3l3GTz57SAe1hPyXL0kdH1r2S2d7N+ReeFh3ocmf5b4Bx+Aq/9KDrzSzyzdzfdhyLzzeqOGS85s3cE+8Q4w/mcFmQveu6pcsGsfphcxNSVKkdmvEH2cq8IH/2jPw5Z3nk7ZDX0Z6E+ouPJ+Nt86dPD2GmMg0yv6Mz30fBRYxxk+jHI7WC2/xkzezNvpn8iZ/p2Zu+dvPKj6FN+eU+x8pDFHwz9w+Ho5DOq3AxV/6vcYfKjF3zQ9R3/k3eeTD8kl+WDUV7p1HPM9Gk0M6pz7Q56VR+t3X6uNeYeR8ezm27fMk+lj2prLlsTl3NaUOUyVvZkjPpZ5XXWKl9Gd89M/a7vTWRzmOlv4nUfRdkvLNM/ke6QO5/TVjjz4M+y2rtqn8uF5nJEPdm60o6BPvPwd31vgrPTXes+5rI149E6g7V2cLZ3szOb+SqduWzes7WS6ccgF3Q8s8z01Hk5l2dxdZzm6Po+nSvmYBev+iga/aJG+Z/OVQ/MVXW7uL46bRedl38Q+cyX6bP8NX+8yCg/otvn7I8E40D1zjqj8mc5+gLqjB3Vz01txEyv6GWcaUGWy/SjeAY664qOr+P5dGbnRTmz98t5XvNR9BselAw+BDMv4a5vRFVnNddh1PcqP8pFnmv1VDAfdbq65jKNuYyub0TVryoXcDZX1tmhea5dnWyt+x285lXM9szNSugkm7mR3kF92ZpUuaDj2U3V410zcKbGmb1vpdvnmZm8k1d8FM38Yma8Pwn38OjLf/eLfqbOyDvKd6j6XuVm6P4BOIp8pTPH2NHx7EBnibqbK2pn1k5bWWc1nX4nrvcjwu/2Oe1o6p31DJ19lafKrVD1usp1iXnSI3RCjfGXd/L4R9Huh+K3UD1gVe4TqWakys2Q/YH4W/xBm9WPQS6gh/fzFNlcqb6yZry6VjL9ajgv7FeVI6N8hx01CGsydlSeKvcmRjM1yh9Nz5dnWfoo+oQh/oR7XKXzYMUfG/7RqZjxdb1dOv2qPPHHiJ7sj9DKmrVX9BGxp9pf5WbJetnV37iu7pu5TL+CTm8zYg/3jXQXV2v1u7VqTicdzxN0+q2z0fH/dK7q5c66O2otfRTtYscPULGzPh8Kxk5j7LRRnGmz8AHv1FTPyD/K76LzMp55ybt1dY3IMV/pXKuP67uoZoC681I7s3bayjqr6fS3kM2BW+ueGV3RXLau2O3rkvUw0w8zH26txKycmRnuY+w0xk5zsdOuYnc/r+TsvU5/FJ294CfSGbYVD2MHPYzP4GpR05i5EbP+nWQv/M66S7aHenadbH2W+L3P9C7LZ7qSXae7Zry6VlhXyXSl2p+x2sNsDmbqzXgrsut36tPDeJWsD9S7c+HI9BVYi7HTGDs6HmXW/5M4M3vtj6K/g38C+XIO9xIexZk2Q1xX61Q1u76g41nhqnnkHwI9HFmOuvNcSfV7z3I6B5wJ1bIZcHt1z46D13JrjTO6vreQzaHTgmzPiGxuO3Vmrtf1deAMdHBzNUtWYxRn2m52XGNnnzJm5qbLar32R9FuVm/4uOgXqOwYJNYYxSOcPzR9MHnMoP5s3WVlz7E4F9U8ZC/1bF0R13H+LLdynadh7xiHxvmLdabtWPNgLsPtIaP8LNFvnYsz6yA06oHLaXxmTfQ+Z5j1H5itLtpTzssMOj/ZLDGehfsZZ9qbWOmrcnY/WanX+ihaKVyxu17FndcKdgzuqMYoH3R9uzhzvZVecU/2Mj+znoV7GVPjmrFbd3Ev7xlib1ZD9cwTjLxZPlszjp/VHTNk/kx3RK+y/s2uu1R7stqdtUKd8ZXM9GAXu665Umdlz1PcOQczzN5X+VH01/wTx1l21+tw9TXPDi73Mz4L/0Cs/rFwuDqMZ1ntV+zj3J5ds16lB8y5e8vWjJnbietfBmdH97m1ejKNtVydbO32z3J2/11U85blnH5mrTHr3o2bB65n4CztmAu3lxrjWc7uH/FEj6+45kzN8qNoNzM3tpud154ZRPdgMR5BP+Mz8OHX2p21kukruBf6TrIXf7aO2N1TpkfuLbhZVDo9JpwdzpHTsnW2h/lq3SFqzu67C52nbK6O4sPE6Rp31sTVnGVHjRmyGeF65xyw1mz9Ga9jdL0qd3d/KuJedt5Pt1b6UdQt0GV3vaeohuoKeD3GV8CXxmh9B/qAXDVLWnfnNdx9Z+u74Msz1qpXPe7kXP1qzditnaZxdbg9I2b9M5yZgcyf6UdxvWrPKlV9xleTzcxVvZ2tSz/j387OeenU+tdH0d+NX2dRa1e9nVx1T6OB5kNJP2PF5aKGy/0UqhfsiO7ezJfNbzbbqrOO5rlWnPYUOl+cNc6yO7t158j2drSMjuduOvNwFHPFnEP1zBPovVCbhT/bmzg7C5xX1mOsuJzTAuYY/wZ2zs+o1r8+inYxuvBPYWZAZ7zHwJ89iN3DofqZ9R3whZut1e/y1Zxyf+aNnMs7LSOr8WZ0nnh2a8bZmjHXTrubas4UN3PO56i8nZy7R/VUa1efNbP1HXD2srX6R8Te0eHI9GOQc8z6r+CuPna5634u+Si66+Zn2X1fuwd3d72K7FqZ/lbcy7xan2WllvtjwToujoO65p/A/WGIWHWuGRPmGXNNrXvsgL/7qrdBph/oZ9Z7ZSWn9alxTZjL9tH3Fjgrjky/gtlrVf4qt8pb+3jHff3HR9GOC+6o0eXOa61QDStzjMkov0L2x+KKa72B7OXt1tVs/S3+YEWOea3L3Aj6eb/Mj2CvI+Y8ZAfr8JzVcvuYp48x16q9iWwOHOxnxqhmlSPdaypd35vhDHH+djKqOcqTyl/l7uDO2dhxrarG1n9TVF3oU9k1bLN1Kn+V2wlfIKP1G6lmUv+IcE2Pq6M553Ea6XiOCd/dRP85E4y5rjTmeGT7urAej7vo9DSbLeZGedVGa8bMvQ03E1zfRXXN0XxVuTNcVfdJrpzJbR9FV97kp7Br+FiH8Z1kL5ls/TTZyzxbk8rHuEvs437GoTn9LKMejfJKvNxjD2eBMdeVxrNbM1Y9O2aY9c+QzYISMzDyzFDNtdL1Pclsf2b9Ozlz7WpvlfstnJ3PbP8/x4YX8Zm9n0Q1iFVuRLW3yh2NPwZxdFBfdw9Z3TfDaN44z9l6VjvwB8t5slzEVZ7alVR94uzwoJf7mFOdeZfj2a0Zj+h4FF6jQ7d/zqfzkZHNTqB551HN5YORz2l3M+pNZ4ZmiHmoDvVyb0aVOwb5KvdbODuLbv/pf1Pkit5J9gJ4Ej4k5Ewu8qNrEH14syNDc13f1WjfuzMwetkHXd8xuHbksvzRqH8l7BfjCs6Mi/Xs1oyrs1tXsaPjuYLRDATOM5qhKu9y9GU559NaWXwVM70b+dzcjI5Zqj2ruaORD7q+HVzZ9w67r3/qo2j3zfxUqgGtck+g97N6b2deJhnZS7d6kWdke3gNfeGPWPFU9eNeXG6VTk86eXqoxdppsWaN0KuzW7NOaC4XjPJXwV5mPc50MsorlVdzzuc0xeWdtgP2jPOQMZqlq6muxRzjLqv7znBVn2fZeR/LH0U7b+IT6Q5g5atyx0se4Oq6VS7oeDJixkYv7Qr6s1qVL2I9HJWHccWMN4g9s3v/DD4QNO881GNNjYfTuZ86PdQzKk+m302nf9V8BVWemsZu7bQR2bWvgHOQUeV2cuY6Z/Z++W92zd3SR9Gui+/kjfc0S/fB6PpmqepWf1iOwd4zrPSVL3X3oq5e+Mxl+93R8Tiv+km1Rxnlg06vXL9D46F5Pese52ENxk7jmZ7Mrz7i9pFMJ+wxe5LpkWM88vPI8tQ0dmunuXoVs/7dVD2rcmfgvGVUs0a6vrt5qq9Xs/RR9GWdbMCpzz5cZw5Hpl/NyouUL253HmmxZqzovdGbeTqseLv+rMcrOlEt1pwt6lzrfuZZQ716jrXzUVPd5Q7zszJWun1wvqyXTlPd5ZlzeXqzXKYR1ukw65+l6lUF52326ND1fVlnx3x9P4puRB+K6gFZ8Z1hV527cC//7gs6e9lzf6fWMeELeO9uP3+uO8he7k6nFutKV20U61n1FV+mdVndp3R6OfJk+jHIKerj7Ll1VdflnBbwek+Q9TLTZ3CzybWLM7q+L/v5fhRt4OwAd/fry727p0tVjzm9j7tZebG6FzL/CDBW3R3Kao7QM/LvIOul0zl/Llavnulxa1czy9PnNKJ5atSvopoFapm3minm6FM/97NWphHW6TDrX6Xq65V917rVNbq+jJU9X2qmP4ruGuYV3nBvZ4f07ENylieuyRe3Y5QP+JJnvOohes/MVznHyJfVclpG94+/86g2ykesOq+dabqfXq5drLrm6XEaGeXJTC+Cbl/Vl3ldTvOx1jPzHQ/9jpFnplaH2V7N+nfz9PV3sKNvb2X6o+jt/ORm3cVbHtrVXlYvW77sXc556CVVnn8EeDiy6zLOtIrqY4HwA4NQz/yVrjmna87pGmd65mGc3U+F+/2P+ks4I9RXcHvdXGXX5H7GTmP8FFnvMn0HV9Z+E2/p8VX8uI+iN5G9lFcenrP7V+AfltCqeDd8YTvCk/lcLovdHweNq2tVecYOehgHrv4qrseqd/Kqaa6js062jjjTWccxyu/A9YZz4Tzh03XmyWp0ctSqONOOpB7peK7A9Zga411wLt26QmeUa/V8yTk7d5d/FPFBXTl+MmeHPR6cM4fWWmF13yrduaAnYtWzWpxB56tymh+hvu6e3UQPORMKZ8bNDvdX+iie1V2OPmqxpucOqtkJqhlbybm4Ojufy1FTqtwb0BlYPWbRPd39Xd8b4TzOHndy6UfR3T/Mp/GWB+OKmiNGszHKH8nLOHuQIladPvVwf5DpR5FjXR4j3L2vUvU6cu5lzzXjWd3lnG9Gp+b2KE57imxGMmZz1DTOcpmeMcpnrO4jT/Yzm7VszZi5n8SO/u6o0eWyj6KdP8TOWp8AX+zuJb8b1nYxtbO4F7/TlCyX7evqsXZ6pY1yqo+o/IzPMNNHzh/3utxo7eJM6+pkxhuM8jNwDjgPjiqf1aGe5SJ2uvqJ81VUtamRUb7C9Y2ai6ntQutm69/AmZ6SnbUqLvsoepK7fnln2PFw7KhxhjuvX71Uu3rEs7rGemS5jh4wn63VP4t+FKimOj38g8Fec0+23+ncw3hFZ70Kt5e4a47QvmVov9l7h8tn+6qaEauuvuysXsbuOiMyf6Y7sp51ObO34qq6b2KmTzPeNzH1UfTkD/nktc/Cl/kOdtUhV9W9i2xORi/7TK/IPJXO40o6c+c8mTfgnmwfdT131kGma+xqUNdzZ81rdVjtqc7hynx0vHoNp2dxkO1/O663X67lqhm5qq7S/ijq3szswzzDVXWvpPNAZp5sfTW8FuO7yfqezZrqHU+lR0x9lM/0DPqytYsrzvbuT/HRwrjaw9jtGeWpj2o7/wqdfTM9ychquFly2mhP6HpWnVrg9ivuOqrfCXvF+Ck4yzPrL//J1TPV/ij6MseVQ82H5+zh6lZkPtY7S/ZSzfQR1R+EOPMPAF/41FwtaqFn6yzO1i6eRXvFtdOyNc/Mx5qxeqt91VphjmtqDvU4XK7qQfRo5iDUKp+rk2maU5hj/lNh7zTmjKwcWd0d7KpzF2+ZmdX7+H4UXcCnDXF2v5l+DHKrrA5x9gJn7DT9I+D0Vdx+vUd3vx1W9iidl7jTmHfezMOYuY7H6VkNt840xhmVz/XEaR10llmDcdDRWc+teVbor7yEHt7LGbK+jOYgqHIz7Krz5Xm2fhTtGvSKO65xF3zBP/lgZdfO9J3MvCRHvqg1emlXnkp396q65pzWhXUyqtydVLOc5WLd8UZMnZ4MelZqkOx3n+kduNfNUGfeWEfzjtBdXnO8Hqmuv5usZ5lOur43oTP7m7lyvrZ+FP12uoPa8amns95NVntW38XZF63ud3X0ZU4vtUpnTrXReieuH04L/uADpVo7za0Zc60w59YRU481Y7fWc0XHQ9hLzoBqmZ4x2lfpmtOYOXqyPJnxKit7zrDS0xmymcvW3Mf1iBnvlz7/7BrKXXU6dK7V8ezADflZsockW+8kqzurX0n3pR059dGvcZWrtKOp8z641iPzsOYZstl1c/aneKE7D2OunUdz9Hc8maZQY0zN5StmepR5OQcZWd7p1GbnijnGIzo/z26y3mX63WRzVs3vk3T61/G8nda/KRr9oKP8FXSu2fGcYWZoq0F3D8TM+uzBuqopXZ3xWTp9jJeuHqqTkWeUozbS38ao14p6uWase1zOedxet2/Go94szryB0502IubCzYHTgmyf05zO/dyjuvM4TeH1dhM9Gv3Oqzxzrud6nTOHq+vWM6zuO4PrqdNIx/NGeN+tj6KfBH8BV8KB3vGQBPogniWr5R50JdNJVt8RL9qZPnW9+pLPXvhZTu+LdbIa3D9aX0Xn989ec4/TncbYrfXgnqq22+P2at7tpy+ociOu7mM2U5WuROxygfO4utl+x4z3DKPeZblMX6FbS33ZmoxyVf6n0pnFUd4x/ChaKfp27v6ZqoHNHorO+gpYn3GmXYEbesYVfKGPNL3ezHWOws/ruWu5+CrcCzRinrO11mC9LOd81DVfebI6VZ64fXcTPc8O9RGnHYV+TM45PbyfbJ/S8TzB1T3P5i9bd7n6vn8qnTlUT/lR1Cn2JJ3763h2woFlrGQPSWd9JX/MH04yyu+GL+U4V4d6WCNwOa55UKeXOK2CdWb3dxn1OfLO43IaZ+uIVXfeTOea9VRjTJz2JjgHqjOmVzXqmndrhXtdnB3qOwvnIcPNguacfgXZ7GVrUuXuptO/juftxM9QfhR1+Am/jFk4sIxHdP3ZA8T1joN1ScejaN1dzL5once9uJ3vMH8EMt3Vcx63V3NcP4HrczYjXKvHrXWP82TXoI/rao/meB3mM83lK6oeutno4PZlMWs7LQhdPc7rtBV21HE9VLTfGW4mzhysW62J1nA+p2Xwfrrs6MtTdO694wnSj6JOkY7najr3kHkyvaI7cPQxniF7uM7UJK4WNcZk9YG8A+111nf+YdC1I9OVznUz9B5Uc2tH1Q/qbq6oOd3l3Dpi6tSoO79ba0xNz9Sd5nJXE73WQ3OEGuMgdK1Jb+ahX++NNZ6i0yvOhOq7cDO3ysz+Ge8KnT53PJ9E+lH0SXSaknky/QydQc0eIu7NcvRdRfZCIR3PDrRffEnzUI9b8+zWAetSZ0yNqM/tGZF5V/oQe9x8UdNDtWodMfWslsa6nz6u3R5qLs60o9BnGPW30jlHmcaYOr0uH7AeNc05/S7YG8ahOf0KOrNHzeUrZv1X82T/r+Dv37/XfhTxAV093sAdw9h9WOg7e7C20zO6vifQ+XFzpDm31thpSqa59QqufsBc1hPqjDtarDkn2drt6aw11jr0uT26j1oVh+bWK7AvZJRXnI+ai901ItYcPUqVO3Ad1hzt3YHro4PzovrZg9dQQqOXMMd4ltH17oKzsXpUdD0d7EdRZ3PHs4vOtTqeVbqDRR9jp+kDQ21E1zfC3UMFfYx3w952HgAlvHp2a0Kd3oidpnGWq+j6ZuGL0vW+0lxea1b1s7XTGAfUuaZGX8C4w2gPe+1wM6MwTy/zlUepYnetrIbTHR3PFbD3hDrjVVwd1Zivckr28zB+K0/NQUbnfv71UdTZtMvzyXAoGSurg109OIzPMlvvj/lDcyWjedIXtntxR6x65tF1VavC1dI4u8/IP4GbN6cx79YdT6wZZ7mAusupNqLj7XiOzb1jLcZOY6xaNm9HMq/q5z7Gb6Lbq66vSzarGZWPsdMY382bZ+AM//oo+rIXN7hOC5j7U/wBuAvew4iOZwU+hIxJvNSzF7sy+kOgusZun3p0zbyj49lN9nLm3GWHerK91Xrk11j99BHqzvMWRvPhZomzN/KozjXpenhUjPK74YzcSTV3jBXO7Ayr+z6Fzvzs8PzHR9HIfDQ9V9C57sgzyu9g12CyTvaQxUN09tBaXOu1ZlnZE3RftM6T6Qdy1Vr9umZtrhkrjEOrdOYYj3A9oKZxtmaczYrTR2unMa+1A/qyNQ/ur9Y7yPpInZrLZ3CvarF2mls7XP0uK3uugn3ecbjaxOWcNmJlzwpVz6pc0PG8kY/6N0WdX3LH8waqB4lxRtfXIbufndfYQfZiVp25yKvPrenVdVWTa8ZVTsn0Hbg+Vi/0bM04q6F6tVaY47ral+2hR8nyznuGTl+dhxrjTAtGM6izna3VrzAOYn+1dyczvarm4yzd+dHrd5nx7qLqWZULOp4VrqqrbP8o4gPxyYyGcebB6vjocfHKQ/VTyeaML/bROoOeiKkx38llWpDpV5C90DlvVax7nS9buxr00M9cxBn0VXurOl2q3rkZ0lyGm6XQ3Jp++hzOV+2pciNW9x1JD98A54gxoeZial/Gs9OZyyo/9VFUFToa+R10rtHxjHDD6LSMzEs9Yp4VanzYzh4j6MvWV+P6Wj0Amss8R+OPQay1BteM3Vpj6gFrkSo3oupV1dOq/9TUW60JtWyfi3WP09y60t4AZ496N8d5q/ZkaC3u13yXGa/C3qrOfLZWv8Znj6o2ocY40zJmvGfp9K7jeTP/+1E0+kFG+S//Rzak1Bk7Kk+VmyF7qFlfXwBurdpZ+OLuzp/zUWPtOLs1cXtdnK07cUbXV1H1qOo9Nbd2M+HWyijnPC7WPfSETpxGWIdksxK6yxHOSsTcS98s3K/XcdesruFy1LTuWaoeHMhz7WZlVG+Wql7nmlVOyXyZHozyv5FsNv85iuSXHtnAdXUXO62Kz6LX5PpORi/SLMcXu/NlnmytMTXms5yuNabm8kqWy/QOozmLfKZxTjrrrEaVc3VcnKE57qVOrUvVh+gtPd3+M8d1Voc5t4det85wtVfYUWMF1/MzsBZjpzHehavrtE+lMzOrnqn/fPYW3A/yZtwwdjTGmfbp8AU9gi/tbF/mq/Yo9FQ1uHaH+lg7oJfX6TKakyrvcvEHRHOxVr1a88xa9BB6XMwcr0HoUT0jy3X7NOopZ0A1HtkerllPyTRXg9fNvBlV7qfg5oMa42CX/hQ/ob8f+VF0NH75o/wVuAF1mqPyRa7y/ETYw1Hc0ZjPctVL3vm4dlpWc7QvY+Tpzkt8GKjfaZrTtc5nZ1bpy9bqV+hnzJzqSqY5vcuoJy6fzYLzOrI9XOv8OU8nDpye1XXeA/ek2k+gM0OZp6Nnnie5u3dXXW/LR5Eb7jsYXXOU38lVQ+rq8uE4e7yZmK1RL9XT8R8yH9nebK1xdajXUdWvmPFW/XX9d3Ph5iWLY63Q42Jqbq3Qw3zGrH+FbA6I67/z06NrHtwz2q9rjTvwmoQ1K+/dcPbOHlp3lWzvrH4Fo/5VuSu54rpbPoqu5Iof+izVMGa5rs44cLrTVog6+oDzYb+C7EHL9GCUD7KX8uqaR1CttQa1Edn+WVb7GDPAWaDGfOVhzD3MZWtFaztPpgej/A7Yy6qPzqv+al0d6uGasVuzlmOUz1jd14VztJtsfjoa4yDTv1zLqz+KrnxIVjk7qLq/qpXlrnqwR/fltDNkvc30QPPuJe32Z3nd21lHPLPWmHWZVw81+maJl7YezI1wHmqsG2e9Bvco3DNaK3odXjNjlCez/sP0l2ueY805cJ7uWmuxrovdWhl5nHYUP9cOtDdZ/xnfQTavSpbL9IxZ/x1c1e+rePyjSB8SHup5ip1Dlj0cozjI9CvRF4t7yazQ7SfnoEK9M/uOYm82g/SM1oydh/4niP7yqDzMqYd+9egeV0tzozW1Cr2e86vGtfOfZTQT6ss8o3XgNDLKK6PrKaP8Lq7q0wruPqgxDjL9GOSORv5OtOcxA9XxBm79KOIvYOaXkPmddgejwVvJU2McuJf1mWOWlT3HoIdZLtD8yBtkezrXC9TLfVzrHrf+VDgvnBuN3drNC3XuU1jPrdVbHQ7m6c32jeB8uHWHbJ666+xa2T3F2sVVPTLrV1b2rMD5WD1cPV6nioNMD0Z5x8qeGbJeZXrFypzt5raPoit/wCtrr7AyhNWeLJfps/ChfgPa02zttE6ecbZmLcaE+6u10zrr0T3MoC/w1b7r3mytXuaqtZJ5Mr+juqcMl896EDPD2RmteR5pO9Za363V62A9tx4R15vZQ2b6H8x4u2Q1M/0ocpn+RrLeZfon0Pooqn7AUe7s0N/NzoFcqcU9jINMX8X9waiY9XeZmRf1Zmt6ncetO/HKETWy647WvIcVst6FPjror9bqZc7VzdZK5sn8B54XXnMV7cNqL7jfrXccWW2uI3ZQz2qwHunkZxj10eV39L8Lr8M4yPSjeb+j/CcSs1LNRJVbofVR9CayX0CmX81oEEcv4K7O+EpG96x0PKvMvGgdnT3uGozVe3Y9up9ZZup1ezqC85GtGWdrxtV65NdYjy4z3lU4Y05jX90szaxdbXedIPNl/i5n968yOwcrsL6LqYVe0cmPPMqMtyLrZaY/wcq9fNxH0ZtYGS7dk+3P9GOQ+0Q6Q+s8qrmXt0LdveDD42rRt2u9Wp9UucDNTbxM9ZhF93E/c5lPyfxuL/WZw5HpZ6h6yJja6ny49d/kGVDUE+vMl8VuXdUKqtxOrugxWbmG2+O0XVxZ+23MztZlH0WzN/LJdAcs83X1iFXny37lqOj4RnnH6CWZofu4dh4l8xP6ssN5tYZbR8w91Ny6ouPpwNnoHLGP52ytaE415rhWLz2s51Av93T2j8h66PrktEBzWb3OWjVej17NZ2uS1aj2EF77DJyJDM7AmYN1qzgY6aP8b2LXbIy47KPoSnY+PF1GQ+gejIzM67Qj0bMaK+gDuKvmDNFPHmfJaqmWeZxP9SxPbZTX8wjd22Wlp7Gnc4Sf52zNazBWjT6u6WNMbYYze2d6lM2F1mBO9WztDsK6znMkvqruW6h6WOXOwLrZHFZ6l6zGbmZ6PON9K6c+in7CL6DDlYMXtbNrZPpu7rrOLDpjs2vF6XyxM1afnrO107I8c1ns6HgqdrxMObd6zta6t+OZ8bs4tOwY0fEQ9sbNAdcKPYw7a8ZEPfQ7zwzu2rye5leuU/W/09ts/1sY3dcoH3R9u5jt4052Xnv5o2jnTSj6oKw+NMpo7+7BmXngnMdpR6HfSfelswt9ka6ulUw/TI4+V5trPXNN6HO1KpzHaUR7p/0c9dZ53NmtFedxWqyrfazh4grn7ezLyH7/7C377PZRy+bFHW6P7uM680Rc6SPcHtZaYaVP7PVudAb1TGb1M1xRc4boNY8r2FV3+aOoYvXmVvetctXA6EPReRBdPtOo6zVWD1LlPgk+iHwpu3nLfO7QPZXfebgm2X56Ktg/xg7OxmgOIqc+7nG687jcrEcJjx4ZWd5pXar+uP6y18xxj3qcn3tdTvdqrGQ+1qT+JrI+cj5WD5JpM7oS+ZHvLs72uJq3itV9M2z/KFq94dE+l6fG+E10hjkbfMZBps+ysw6PFc4M/mhf1A6fxk7nXl0zztZaM6PysN5VdPvF/mbriKnTozgPvZnHHWSU3wFnQ3U9EzcrrDVaOy3WrqZb60EPmdV/KtmsBplewT2Mn6Lb246v47mb7R9Fs1QPHOn4Op4R2fBl+gxZDadnL+5M3wHrxrUyfSczvXMv7yzHfJDlVGcN9fMazGU+l3dnkumzsHeMqfNwnlgz5zzZtd2ZWpB5Ml+GyzttxKh3gfNxrR7muKZf83rQq3T1bP0G2LNsbkJzs3IF2TWq+1Puus+zcB4YV8x47+DRj6Knfxl3X18fhGzQz+qMncY4NKdfjXtxM+fWRPMOXifL8To8VM/WGus1uFYvqXJKx3MUL2HGoVU6j8jRq3u4rjw8u7VCnZ7IUw80766V7evgZkFzWZ7z4mYm1t1jVNfphHtcfa6zWjsY9XUnrt5o7mb0UW6Gs/vv5soZmeWf0c2M8qtcVXeWp+8jG95KZy40faAqjXFWr0PXdwbt0ahffCFXOffSpsdp2UEP67CWMqPTM0J77vSRdiR6pnHGuI6YZ6e5HPPUXUzd5Spm/Qp7yP5Rr2aHB30ul3mqtZLpRD30j65xJVW/dR5Gh/OHxnqOrs74tzAzFzPeEaxV/psimoNZnXR9P51s+Gf1q+D1+DKgfiedF3Cs9UWcvZQzT3Z2axdnNRm7c8B4hqwv7kVOb6ZHjmv1ZmvnZ0yPyzEfZNdyVLkduL6NeswZ0TxjPTvNnXkQ1enprCu4p7uvSzYTAWdjJ6yb3QvjYNe97ajxJGdnItuf6Y7yo+gKZm7ujcTw6nGWrEb2YIXm9FWynyfTd7A6C+6lPfuSDX92ZB7VY13VzDwaP4nrq+u3zoHmYq0610rmydYKc5mPWnh1D3PUVqh66XLZLFDjLNGnnkzjnspDn8Jaru4duL6FXsWhZfvPkNWrrpXpZ6hqVrm3cPcskVs/inb8sKzB2GmMdxODpsO/MnxuD+vRE9c8c+xiZy0HX9wOvrCrg3vcfqexBtdaaxRz7Wqqh1pG1gvtufM4zcE6bk2cJ9PcOnC5zJfpxGlnYD9dTmP6M4176B/prna1dkfFKE9m/d0+cTYqdE5WD9bTs6PKddD9net1YT8YO+hhvMKOGqvc9lF05Q/Zqd3xnMENqYuZCzqDzRzjVbI6mX78Ty4O6jtgvxgrkeN5xF/zR8Ct3Zm+kafycp/6MiqP64HTlKyXmT5aa6w13JpxtQ6ynNbknhFuD+MVXK+cFmRzotoo5jk7uIfrTGOsGus77xlcT5ymrM5EF9Zl7DTGs+j+s7WOZp/oYbyTK2tXpB9F2Q3N6scg9xR33JM+hNnacTZ/lqx+pl8FX66O6sWr+2ePbL/qsx63HlF5qtwMrqfUoveq06NETvfsWKumZHrAvYyJyzF2VP3VnMY86GXe6aM9zEes6+pQWI95pcrN4n7/rk8ddA520bmXkUfnUu/RrSP+yVTzk81epXdIP4pmqC5W5b7UQ915cPkQrRysQ6gxJlmdLqOZqV7GobncDHoNrclDda4j7qx1PzX6u7AHGmvfZ3Bz4uZoZu200Vr3Ks5D3yiepdObrJduTc3NRKxdnOmjmiMqr9ZzZHpQ5bv9iV6P/DpPcT57ZERu1aOay/8GqtmYJaul+paPoozsBr78N50hp4fxDqqa1QMbjF4Mu8jmiS9k+vRlPrPmETlXV3NOZ8x1dg0XnyXrafRR86PeOv/smnFnzTjWgerM7WSmN67v2Trmwc2E6llMXet1DrdfYZxpitajPmK2f5yTq6lmjbHTGH/5Tzozcpa4xqUfRTu445fheOKBqq6n91P5dtB5oYzyx0X3yXnIXtqhqd5dM6bu1npkenVkuJzTumQ9oa4x50FjarPrM3UYj+Ael+vW6sL+zq415sxwfkZrp/EaLs5y1INMD0b5q3C9ZbwLdy3S8TzB7v5w1t6O/SjKfgCnO+0o9Bmyh4+x0xi/Hb6sOw9Lx9Nlxx+EM3uPwcOT6YQzEzVnDt03qqX5bJ/itED3OkZ5x86espbO6sr6zH0Frqa71mhPxSjvYJ84I9m6OtTD/fRSV7jfebmmplS6nlfIfvfsdeY7khm4iuo6OnefCPvI2GkaMzdLtn+XHvzroyjbkOlXMbreKH80PXegD0HngaCHcaZdjXu5MHZ0PBlZD/8WL209KlhjdGR7Yp3VVd15mHOay8/AHriZjP66l3e2P/PQH7gZCrL9nbXTuHb+LqO9rj/Usn7GLHQ1jamP1u4go2tQd4zyq1R9iFzlOWQWRr5ZXD1qjMkofyeu3xUdzyfwr4+iN3DlL5cP913oQ9h9ILnHoQ/46lHR9WWs7lPYL/ewOk2JvuuhucrrNOqdNffSp9CzA/ZC++py2ZreEdUezteZdcBY0X276fRp1FenHck8ZWs9nN7dFzmi+6hTu4Kqd8wxJjE/O46MUT7oeO7iqj7eNSNneNVH0dW/sCtrH42XtYN7Rsz6O7BOXIN6kOnHIDeDmwWnjcj2qBYe9XLtNK47NSq4P7SdVH0NNO/WnI/OmnE3N7NmPfpcTjW3h3tncf1jf13POTtcdzxcd2Knc61nrl38BDt7OMPsdd18Us88M5zdv5s3zIjj79+/13wUvfUHvgsdwGoYKx9jai5/hjP1djy0s7iXOV/ezFN3a+7LNLefe7hmnnuI064g611n3rT3ozXj3Ws9u7XGqmcxc13YN8aV5uYimyWnM595eC1qWifIrtGh67uK6D+PnYxmp8pnOfpCc/qIlT1v4445+o+PouyCTh89OLOs7rua0SDtfsBma+n1Vw/W6jDj7XCm/5xFfXnzIE7nnuzgfq1Dnfuou/0dWLOL6532lHORrdVfrRkzR211rWeH81T+WVwvXRya0xWt113rUemB0yo90Bxru71Om6Hq05l+xgydPTqot7sn4HVm97+N1XmY2ZNdw2nBJf+maIbsponzjeJM+xT04XHrXVT1eL3KG3DPLK5nqnXWjpghPULXPL1VfHZNslzoPM7i+pT1uzODblYZc824Wo/gPj2ch9pO2J+sb07XmOuOh2teo6NRj7WeuXaM8o5RP7LeznJmbwZnULXddH7+UX6E6x81xpnm6PoqdtQIHv8o2s3OX84VcIhHA1t5GZ9hVCt7wLP1LrKXr76oleylnXnpzzTNMeba7cl01iRVTqk82cw5TRnlA5ejxvmZWUesZ7d2e+kho/xVsPejI/Zwr8YznsxPMq/mK7J8dj3F9bCCfdeY+atw9zy65oz3DYz6djQ9u9l1zaWPouzimU7igej634Z74Dpw+N0D1EX3njlYk8cbGL2EOUuxVl1nrtI7+1w+81b6KHcW9lBj6qM8ZyKLFZdza6dVVH6392z+DLN9pW80Q27tDq2nfqfRr7rLVWvHKO840xe3183NjmOVHfszVn7fd+Pmq6LrW+F/P4rcRWZussPuep9Cd+DVV613srteh6tmwL2Ys5nTh9B5z+aydcShOZx3lm5fnU815t0sUmOs0Ke687JWteY1WV/XvJYyyq+ifXWH82nMnNMzT+UPsjy9o5xbPw37r+sren0kM+rWJPIu9xuImVqZn2xPpjv+OSY3ZIxqjPJvZfW+ZwebD+po7eJVdtWZQV+q1LhmnK0ZMxdadW2XrzQXO11x2hNUs+U056emM6/r0Z6uJ/OTrI5S5ULj9Tqs9pbzM5olp7t1VWeUJ5l+IMd1tW+G2V7cjZuXbM7oq5jxZuzqwVs4+/Nk+5f+8xnJivNh+7JnuPXB0xf36ku8gnV5XbeehS9QtyY6V5wxt8/53dH1Mpfto697rLLS/1EfWTOrT4+ryzz1yKmH6wrn4X11c2fY0U/d69asv7p2GtcRK1kuW+9i1Cf2lDOkxwqs4eqt1iasu8oVfbgDzifJ9C5u//RHkSvi6PpmYV33CxvFT6OD3l3Pwoe1Oir/Gc7uJ67XSuTUV61V0yPTmVv1dWC9mb0jRn3RPOfArelRNJf5OG/ZWnEejfXs1ozdmlS5UX/Yx87BvVmdzNNdB1me3lGuYpR/Ap2fznEWrZOtVzi738F+MXYa47s5e/3pj6JP5ewvqsPsg6O+zpoxc6vM1rniHjJc3/gSZsy1emJNTfc4b5ab8QXq5zFixkt0Ntk3xk5zfY+aWax+esgZjzuzBusxDrgvND1ndPrCnncO3ZfVyDzddZDls7Xuq3B5p3XRHrPf7GHWt0xfJbsH3k9obv1JnOnflbj7cpoj/ShyBbraF8+Zwc8esFi7h+6n4GaML+WIQ8viWDuNB/XOHuYyX8Yov0I2G9Q5V+6svmp/xNTd2mkuT92t9RxoXOXeBGdltJ71uzXJ8pmXMbWdPN03zqXOYeRH6wxXT/VYr3BlT64mu/dMd3S86UcR6RQLZry/ATfgV8GHtHNU+57AzY/TMuhlXGk8mMvWPDKf1qsY5XfCPlexmwvOC9f0ZWvGhJ4K5mfjHXAmOANK5cnmiHuoz6ypZTDv9jDOtLvRHnM9ezzJruu/oSezuHnbTfujyOFuzmmfyF0/Bx/O1fUZ3lCn8/t2D0RoTndQ1/3Vod5sH3XuiVjX7sjyu+j2iX8AXEzUQ3/mUW3kj1jP1NyafgfzjDPYG8aEfa16rvnRetbv9jIfsZ6dR3XitLtgD6v52EFWP1tHfDdP9uTt/LP6y1ndN4IPqbuO03biXgKrcODdg7FjzevsQuu7615F9bvPZoO6W1PTvbp2R5ZTnZ5RzBxrKU6bRfvnzvSN4mwOOrMSussdSQ166XGo7vzM72ClV24GODOd9d+/f4//+q//+o+Yh9vrNOa5Vi3TidM6dHvDOenum0VnsrpmtdYZZI1KPwN//4x3E7Ohxx2sXudf/6bozE2v7Ov8spxOzcXUKma8HapBzvSz6APUPXaxs1ZwpiexV+cg06q1i93R2U89tKsY9Thy6nF76NOzWyuVhzoPtydwHmr0u5g6ya4/S9Zzp2eezpqHy432dvKB04JMvwvtG3vIvnKGOsfT7LgH7fEIelzsNAdnKvNl0L9SI+C+f30UObhpBv7gPN7E1fczGmI+pNk64lG9DlqLRwZzLu5oHUY9qWYpNDdzbu08WZ6+as16imrq1Zga8x06v3/mdU92dj7Woc41iXyV45oezVfXCjSX1e3g+kGNfdW4o3XWcVT/xmi0NzTFeRxV7khml/EsK31b2aNwP+fNrd+M60voV8PZ6szZFbQ+ila5+4eZxd2f084y8zBkDxIfKtXPktXIdKXj6ZL97t3Dwbjy8ZytdQ89PDJfts5i+l0cWkaVO5o94jxVe7JcNq/ZWvcxF2se9BH16FnzzPG8i9k+HsnszK5dnGnUY63ngDFhTVLlzpL1jv2uZmcWrcO62dppzO++zy7sDeMn2Xkvo1pLH0WuKDXGb4EP/NkHtTvAzEfc2TtC76E6Vji7f5ZRL5jP+si15mfWPDJfZ50dXWa8FaNeMh+x06lR1zx1xvSQLJfVYt5BnfEu2Gv23+Xp7awZay31ME9vQG+Wc/ljsH8F9pLxTnSu3LGC7nXrUd1Rfhe7+nUlbq5GcYeljyKycuEniPt88n75IKiewT1cV3sV9XMv41XO7j8a/WFe42xNTR+oWOtBPdvTXUfs0OtU++k/A/uumnqYy/bp3kzXPPWsXqzp7/i4p8o5Op6AfWMui7mvE2drxrHOtExXzenMvw3XN6cFOjtujmbQvWfqOFyt3dd4M6szN7tvy0fRHXR+sI7nDnRIVwfW7avqMp6hu/fOB3DUy+rFzJc3fRFrzq15ZL7OmnEndwXZDLGvjFXTOWCNzMODOvdETN9oD7WZXOBqKlUuyPqosVu7mVA9WzNmLtOcnsUjWLezZ0Tnd+0Y7Rv1eIaopfVc7VH+p7Gj/08w/VH05A/KazN2GuMr4YNBslzo3M94hD6cPGbo7BvlO4x6M8ofyYtYX8h65qG61svOZ9fuCHRNur6zZD1V3XmqeenoHQ+pcsGo/jG49xHsIWEu6yN9Qeh6Hbdm7NZOc2v69axed9xJp2fMM840RecjO0aor7Nn1n817K3r9yj+JP7jo8j9IE5TXN5pv43uYKuHmoN1u8Q+d4ygh3GQ6Wfgi9c9kBnh4znWqmcH96+utY5Saaw1A3vBualmyPlUz+LQZnXN66F5eghzmcetK20Xbg6iv51D/dyb5ehTLaBOj8u9lWw2FOfR2eExy8w+Xof7NOaa3t+Cm0FqjJ1WxeW/KeLGDit7PoFdQ9h96PgQEKetUt1L9sDOcGZvRryk9dAc1zzHWnV6XP3Vtasd0Mt84PZ0yH7/rrequXXXF2s9Mp211Zetebh9ESuspTq1FbSHDvY3670eVT7LuTX3dvORW0H3VvWcNkPWNzcjqu/E1XMz5tg1f7+Bzqx0PCT2lB9FT5I9QIzv5IoHqUKvl11bH3p3zDDaU+Vn9cD1eBWt5daMuc689K2uXcycgznGGa5fjAP1un1B5lN/tj90lwuyfKYfpm4WO+hzOeodtEfsMftHL+FeF3fWTuM6Yj2rZ/Zw9UmVOyb6p5pjxkt0FtxBVM88PwHXWxc731s59VHEH5JxaCsHa2RUuVk4vIyvZvZ6HS8fXneQKke6vrNU81GhftbIdLfPaTPrLFa4j7V2oT0b9Y5ety+bgUw/TK0sps79Wf0juYbmMugdUfWImusxY3fQo/HMWs8un3k0Xx1voeohdZ2l7BjR8QRas1v/U+FMcF66B6HG+AxTH0V64c5NdDwrsC7js3BQzwztqI578Oircjvg9d/C6KHoHOrXs9Zw9Uba7NrF1Lk3yHL0zaIvZmpunflV55Hpo1oK/VxT07NbjzwruF643ro8Y50Np2k8u444y9NLbUTHcyecD+pXUNU9M3O859F6tv4Orup/p27H06H9UbTrgruo7qfKvYXZgdUHwq15rOD2ZjWddgb2jPEIvsizI7zcpzgvtZU140D1WVhrFTdTutZY9+jazUOmH2Z2yYqu54zsvrO6h/k9ZGS9YL+rODTm3fysrqkFqtPvYoe7xg5Gv/tOf8is/zBz6+Znti7r8ZiF97RSw1H1s8o9wer9pB9FswXpZ3yW3Q9YsHNgOMgcTLcmzrcC76M6SKYHVe4MnRfpKE/Uq3vd2mlag9rMehS7M/1BpldoT7V/K71cmR23h2semYdk+1yse67C9dXlGGe5zLOyDjKNMT1K+PWYZWXPGTgHqnUPh9PVX+29kzvvY3dvz9br7v/fj6LuBseZvW/hrkEhel19gKi5mLlVsgeFGuMRs/4KvnSrlzF1t09j7ulos2vGAXN6VqrcCDczTnNrxlyrx615ZD6SeVhPdd2rcC/17noW9mzU++rI/LPrSqNeeTJcPtOcPqLTG86G6oyp7YA13T3TM4I1tE63Vte3g5XezrJyDd2T7f/nKJIZs/4zVNfq/IAVq0MSg8hjldn97qGY2Z/RqTPK342+rLNDfdUe1nTeLN9ZMw6YU92dd6G9dDM1ynHt/KTKHUW+o7t7YM7pnfUq3Z5XOaedXTst1oHqCvcTpzttFe2Nag7nDX0HOk/ZtQhnrLPnjXBWMqpZOQvrMh7h/Ol/PlPcxooZvz5g7giP+n8b3YeHD2f3qOh4lG7dGdhzxhU6RxHrmZpba+zynTVj5kJjjp47YR81znSlmgXWymLqbr/idMY7cH1x/VNd48rL3lM7sw4yjUemZzmtRZx2FtdvB2eLudWjouNR1JutK2auN+OdJZsDzoqbmxEz3lVaH0Uz3HHTnwSHm0dGlQs6ng68F1fXaYf5mXYT88QHiA8T8wp11uBel6/izpqxW/Mca9XpOQv71uml8+uaR5arain0c03N7RvlzuB6E2v2iPkqp4dqZ9Y8mHP7AtUzRn6n7aTqKXM7Z4C4WauupfPr1ldxdf0rmJmfGW9w+qNo9qL6AM7snfF+Mnc/FEr32pHL8rvo9JxzxPmKHDXWzmL6s7izZsy1I/NWezK0b+wf49Cq3Gh/kOWye2GeVLrLa5zlsnzFTD/opZ851qV2xVrPnXXs15zm76bbt2PSO0tVm/M2w+q+T0BnaXZ+Rv5Rnvzro2h0U7O5Mz8sif1na90xXN0XLX0OerJ49egy6z9D1V+dA6dna50bN5fUWMP5sly2Zpzluoy80a9u39hjjV2OVHNFnWseIz3bq9oqZ/Yepr8Z9DHOPDvWQaZlh+bp1/hJZmaB87NyaJ0ZVvZcwZl7yPp+ZgY4c2dqzfD3799/fxTt5I4fZFcTdqAPxyydPR3PCnwws/VdsI+ux9QYOw/XsU/3uzXjbo5rxrO5VTiX7K/rMTX1uTWPLJftJx2da/V01qw/ikdo76hp7HocaxdT27WutNAj585cM3b+O2CfiZufs7BO1K7uhXGF8zpthZU6rrfZ+m7OXvvSj6K7eEszjsGA6UPCB4a4h+pqsvu78z6q/lUP4l/zUneH7hvFqjPu5ro+xgFzZ8l6yjg00p2LLJfpQZbv6FU+u2fmSaZnZD0a9bXKOW3n2mmB6orzOLL9T8A5uArWruYrGM0hoY/xlbyhl6N7qPJV7hh9FPEBIau5EXFdPX4S2QB3Hgo+2DuOt6K9dzPQ0ViDM+XW9GQ+7slyzrdKVsNpK3AuGDsqD3XW1bWLqWudTGeengruOwN7r2vGWS7zrK4Dt6ZW6XpU/ivZ1SfO2sqhdUaMfMzTy/guVvrKWVmpEVR7q9ws5UdRBW+C8SqjOqN80PXtIhtUPjjE6dQYZ9oZ9B6zB1LzznuG7gMTHvW6fayn+/TI8vRmuRkfUZ0+FxO9xipVLzXHfOT06OhuzZqE/mxNLaAn1plvB6437BNz2nONqa2uA9amb3RoHVc7o8rNwn5nPeaa3l09d3VcfcYVbv9T7OzdzlrKrrr/7Ch0pkb2wO1gdz3HjsHt1HAexjtwNfXamlfN3d9uqpewzo/TVNc4y7MG42zNmGvGVY4x9RnYG/aLeYX9517nm9FHeVLpmnc+pznoYVxR9cblOj2ntrLWI/PEOtD1jJbVcNeYJesF++/W9F5Fp342o7Oc3b+bTm91Fjv+s8xcY/nfFO1g5kYdZ/af2buD0SBnDwx1jUdHhsuN9uwieyj4Iq089PLsDvVXB32dNePZ3FV0ZyHzMDfyaVzpCq+R6axDv8K9Vc75GJNO7zTP3isuR23XOuiu3f5M437HKO8Y9WInOhujI/wO+jK6nipeJetFppOuL2Nlv87eGVjjsY8i3siV8FqMr+DJAc+4uv7R/LlnYK/4wtUztUDjkZ9r+rprxqNcoB7mZuj2ofJVOSV8eqgea/rdXgd17lWdcQa9oVXxKtpv6qpxzTk4uw6yfGftzgo1xtS0/puY7X02kw7nYezoeFZ4y+//LffxyEfREz/8E9ck7mEgzhOa02fRWlpTa63UJZ0ao54wn72M+ZKNs9OzNTXWYdxZM65yqu3EzY2j8rkcY5LtcXqgeXoZc0+mVbndsH+up+y7rtWruczfWVdHtifI1jtiMsofppcz6F7W0dnaNR/V/I1w99HZ9xPozMHVXPJRVP1gVe4MV9VV+OCsDir3MQ4yXdl5P1qD591U/WKOcUA9Yr7MM51r9TJeXbOerjXOYB3u69TIcL2tZok5zl61h7g6jixHvXMPs7kVOv2gJ+tvpnXX3TrqYdzxOa2KMyrfbH+0p3rmerZuoPt5HfVUcP8sq/t2UfXrDFVdN2+7mfoo6txM5alyI87szdhd0w35aHBHD5TTeQ1F890jg7mIqe8kesLeZDHPmlct4tkj9rLmzNodGV3fGUZ9D5xPZ4C5VZ2xHqTSdZ35Ouje1RraO64ZVznVZtYa69mtO3szXZmNu4x6UPWLcYbOTOcgen3mq307ueMaM6z2+zi59yztj6KzN3l2/7Gpxipnhu3M3qPYn+mzjB4mzWfnq2DPGVPTlzdzJPNFDR6am10zDjSvumOUr8j6FL2tZoB59VHvrnmwLqGPR+iEWuXX3Blcn9hfNxOMY+20ap3F1KgzrzqhnvmzmPoK7FXWV8I84x1072WEm0mnKVXuLN2+dX0VZ2us7m9/FJ1h9eZWuft6HTiojEPLBn6ku9zVxDWza2d6BvuWxTxrnhr1WDPmWvcyN7sONKbXeSKmtkpnTlbnSf26n3UYB+p3h3oIr01YJ7RZZvd0+kZP9Jvz0V0zZi406o6O12mhV3GmzTLbEzdnZ6jmk3Fo7rqZvsrOWk+yY0Zmufyj6Iof6oqaT7AytNXDo7mZQ/drnSvovkQVlwuNZ82PNI2ztZJ5RmsXZzn6noRzoDFnyM0Lde5VnKa67mWOPh4VlSfTZ9Aedvpa+Udrxq5etuYe5uml7uKMrq9ipjdVj4/B7IwOraG1SKb/NHb0llxRs+LSj6Irf5gra1+Fe5hUdxp1pcqtwpqMM7o+R6eX2YvYafHi5pH5FfXy7PKdNWus4PY7rYubr6yHzue8q3qVc1S5itH1rkLngRoP+jONa8a690wdrRHQx7PTMqrcWdjrK3ucwXv4RLo96vpWuLI2WfoouvMGz17r7P438NTLPGrzvBPXH6cdhR7EC1l91dod9PKse2fX7nDeDl2f6xk1N1v8Y8L8jK7zozn6Mk31KseYR7WHuss5un2oYP9H88E141gHzLtDcTF92bU0n+UqZv0Znf7N9Dkjq5HpSjZrTgsy/c1wdt7O0kfRXXR+kR3PlawMttLxKO56qnWPUU2iucw3qtGBL1PXX825NYkcD81nXs2zvru2WzPm3oA+7l9lpifaQ/YzmxPVs/1uj67p5eFgjnGX7B6DlbrsGfvpelvt0biTczVYn9DDmswrVS4Y5XfQ6VPl4dyNDpLpQbWXqGe0p8pdTbevXd8ZdlzjtR9FMz/cjPduRsOszDwwx8kHodq7mnPM+hV9IY80zoDTQs/WelSaQm+21n0ux7oZM16lM1tVjoQ3q+l0vQfmNE+4T31Z7KA3ND3vwPU3gzPAPazFPS7H2iO9EyusxRw1xeWcVuF65XqqsM/OcwbOlXLVNY/Bde9gtnefwOmPouoBWYV1GN/JrqFjDcYkrsuDOK2LqxkxdVLd0ypVn6ucoi/lWPNF7TxE83HmmrWztdvjqHLHxO+gi/aafdSYM+hmxNWo9MPUJSOdOXc/lZfa3VT9zGaJM8R1td9pXCuVxv0O3s8OXM9Cczkly2cz0sXto8aYZPfgtE+Cfc9m7Qy7653+KMpYvdHVfcfi3pU9FTND3PUp7uFRrToyqnyVy5j1d+FLmA+Yxs6T+d3BGtzv6mRrxsxpLeZU24X2lL1ijvkgy410l1MqX6Yf5ufYRadu1UuS+VyNbhxr1uJa89Wae9x1idvjdM07fYZqHg7pHT2qM6foLFaH28O1I6uxk6uucbZ3K/vdfF3BZR9FK+z4QVdqrOxxrAzemaGd3dP160Pdwfmc1qUz/PQwJp281tHY6bHm/mxNL3HaKrP9m0Xr6uy6tR7cw1x2v9xP3V3TwTx9mq/qrMIZ4GxwZpzGWD1O55oH9zm/0ompkVF+B1mPg0wPRnnSnZeOp8vOWo6sT5n+U3jNR1H3F931vZHOEPOlPNozyhPW1v2MlSp3LNzHYV7mzFGbhTW4Zl5zXKt3dq1x51CctovoWTYLunYexfkdWc7di3qpUSf00sP9zO9Ee+j6ybzTXMyc050nqHSS1Yv47VR95pw4T4Xzq8b86Bqj/B2c6Wl3L+fuLbzio+iKX4yr6bRVVge32lfp7hjlnbdLtSfLZfoMrkd8Abs1Y6e7w+WzWtRirTgP41hnZLWvgD3Tealmh/PlfMyph3sYz+r0MH83q33mjMzMUOahL3RCL69dQR/j0DKcvyLr7Uzfu97wZYfzVusumX+l1g5m+jPDVXVXefyj6MpfiKvttIxs8DJdGQ3uKH8n1b1QZ3wnfOnq2sWu19ScVzXmq7VCD+trTnXGXPM6GXwZ66H5FbJ5yfSjyGX6Udyj093PGLrzdZnxBp1esbfa+2qte5TMz3WmMdb69LhzBn30M16l6hNzjCu63tm5muXK2h129emt6M/3Hx9Fd//gZ693dv8ddIb56gfqMH809FAP9zg9NO5/C+6F7jSSaaxRrbnXrUl2P8R5nDai0zPOhes345F+DHJHMVcjPdYOty/QupXvDG5+HPRwfqjRV63pd4eDe6tzrFmL8Qzaj1FvRvlDPKybHTvo1Olcc5R/kjM9PpK5uRJ3rdAe+zdF7qae4I5Bm7mG+kb79EEaebtUdTJd6XgyOjOhD0+sGdPL/MwxqlWtdS9zzheMfKFRX4UzFD2sZqED67lcFqvuyPSKzvUc+nN092Roz9jnLm42RmvOS3Y95x9pGj+B68mufmWszgT3dfaP8kGn1tt5aoaURz6K3vCDvwUd4lh3hjvLZ3pFdT29J2oj3F5HNQ+ay9YRj/KEL3ZqzGfxaO005qkzt5vo+ag3Du7RWlnNWf0o5ifTQ6tqdthRI8h6WPWZOcbUO2uN3fXcvspLzZHpZxn136H91HPmn4W1OEPMd1nZ8+Uct38UXfWgPAWHv6LrI3qNTg16eXTIvF2NZJ5sHtxL2sGXN9eVlzkH92Tx6pra1WR9JeHh7HT2HkW/RzWy64x0h7vvjv8MZ3rK2XDxrrUe1B3O62CdyrcC+6N9pZbFmRZwbtxRMcp/qVmdjV2kH0V6Y2fWSqZ/Cp0HosOOGmfIHvBY8/4YZ1pQ5TpUc9J54R6SC7/uy3Sn0U9fluP6DbAv2m/OguYd2Z4uureqVelV7FCPXrOzt0PVa87FGW1lrQdx2oG9Dl4nI6vhtAz2iHGmddg5C2f3H8XMfxoz/SXVzGS6WyuZTtKPohHZTWQXzvQuZ/ffyV0DfcV1VmtyH+Mu3T47Xzwwkcs8cebcuv3OQ3+Wc2uttaJpbkT0wL1kNUed2ojYw32qMxdk+lHkqnuPM3Mj1J+tK1zPXOz6SL9qf4tZmlkroTtvpUdMqpzS9ZFRD7K8m4NR/CY69xY/Y8d7J7M9XoHX0DibNcaO5Y+iGTo30mFXnauYGU71ZusO+lCceUBGe3iP1Fy8G9f/eGEzpw8FPe7hIZWHOV6La91Dz6w2C/vR7VHH4+CMkNH1I195umgtva9O7Y5nN+wvZ+Ds2sWqu5zC/Y4q9yScAT3PwrnijIXHrRXuZY2Kypvpu7mjz6NrjPJH4cn0f46LB/mqumTndXYMVTW0Xfiw8Kigd3TEHt1PzcUjZv2ObD6p0RdxpnUO3beKq+m0t8JZ4dyQjs79jKlX11Td5QOXq+oGzDF2aJ/jPKsFGrt86LpmXeqOLBf7uD+7DnPUV+n83gP1ujW1maOL+lljpg7p7D17jTeRzdIOXN1L/02Ru+CnsGugRg+B6tn6bvRB7tD1rcAZ4guaefVQy/JZPWqddcTVWjVllA9G+VU4q9XcBrqH3ioXjHIV3DvyK9zrNOZncf3MNDdPoz7P5Ll2R5br7KfnDlx/nFYx6x/BGdrFVXW//JtLP4rOog8c9bvZMZC7B/uKWq5mR9v9s83AOXEv7Mzn4IvexdmasVsr1HmmL4sJe8H+RKzHTrJ6o2t17qeTzxjtPQb7r4b9dzOU6fQEoTNPnTnitBEre0Zof2KdaVUvq9xb4M/wCfdMOFtv57UfRaNf4ij/ZnYOdjw0o6PC5c9oQZUj7OfoQXK5bI++9PXIcrqPNTprxg76MqoaGdXvvTMPhLNU7Xf5iCvd7VMqj2quZrbPwXvleQecj6y/9O1aq+ZwfuZI5WENkuXc79xpwSjH2ejAGcqO8O5g5v7exGqf34b9KHr65nl9xr8ZPohdOn7ncQ+805RMPwa5wPVbX9CR50Po8hWsybqZ3lkzjrWeCfP0aa0ZOr/zo/gD4BjlD7zc6ct0R+VZrePuv1NjRNUfzdHn5sTpZ9Yk8jyyvNNcjmutt4tR30b53fAavH52VIzynwB7zvht/P37998fRbxpjZm7gjuu8Qb0oXDr7DgDa/G6pKspVb2gyhH3ciXOw5ezHhX0ZXFnrdei1j1XjDzV75kzUHlHaM+rOp1r8Z6cN8vpfczgajlGHte7rO+a17X6XDy7ro6AOvPqexOjfjjcHs5bNl9ddN+oBq838p/hjmt0eNsckX99FCl8cFbXXVb2PMUbhmsXfIj5czEmo3yXlf5nL/AjedlXOnF7ZtfuGurRs6PKzRJ96vTLeaqXajVDzn9M/kEY5YPwsG5nb0Dv6GdxPWLPFfads8G9Gs+sWV91ejLodXuoMX8l7EfW60xz+hlGs/Llvzk7I515yzyZPyg/iu5kdKNvRh+u2QftzEMU1zp7aD23rrTdVHPAHAedh4Oe0RF7dG+1HsGaVc55RrBHjDM4C4Q5Nz9E8/QxrhjN5VHoit4zD0eVW0H7yd663Mjj5i/WrMFaRPexRuQzP7Usr6ivw6gPrlfUYk1faGeOL/N0ex90/B3PiFd8FO34QXbQHXD1zPpd/AnwnkfxlVQv26PxgnfQ6w71ca2oltWgl2eF13GeEdEf9qma+dmX/sg3U6/yak49PM+Q1doNe9edC9W5h+sspsY8NbdXfatonRHZDIzQPW7/KH+W3fV313sLnRm4m3+evqmnr78TDi5jd34bet/8Gcgofwb3so618+mRQd/IfyR/CLJ9XQ/XI5/D1cnIeuT0lZ66PYwdnDM9SKZnzPrPkvXA9XhFy+oH4XF7NOeO8GRojmvWcXSvM2JHP7XG7hnZVSu7R+pk989zF2dm4goe/TdFb/tlXEE1yIwrYuD12MVsvRnvseDfhXuBO/hypzfbF/Clz7XGzqN1Knhfoem5y0pPOH+uhuqZ55DrZ/mAecZBVs/F1K5g1NdK4143N/RldPyV7o6MUX4X7OFsP2f9HdzMu3V2aI0ursZuOv2kh3GHlT274LUf+yjijayyq86T8MFwhyPTd6PXueuaDu119QLWHH3ZHgf3huby9Gb7Mo9bs+YuOj3k/GV7NNfxzDCqe0xeP8uTrs/h+lhBj5uTiN1cxLq6LvPUsyOjk3frO8h6x/5nvqfgDPN+A6fdCfvJ+Aw7a53hPz6K7rqpM9dxe6m5mFqXbDjfgr7wVw+t5dafAvvMmLoeI9SXrVfhPROnvQ3OC+PQOHMk86iWeYJMV7QGj1V0JhyclY7fwdljzDW9zK2yq84Mnf50PAr7v3JEHa35ZZ6758lx+78pesMP/SZmHyQ+iHfyxDU5L+5F7DSFfwycnzn16Jn7FLePserEaUGV24F7yRN6nC/TifpG9ao40FpxzrwrnK2l/XO9dHNCXXNZjTi7dYXbkx3E6Zl2J6MZGOVnuGruzhL3070n17e7ueP61TVu/SiqbuTLPN1BH8GHxtWlNvLfQfYAZzpRT+zhPqfH2mmxdnX0XOluf5dqX9anTFcyD2dH0Rdy5cmI3MhT+Xj96l5WqH7fFdp39j7TeS3mneb2uEPzxHky3xvRfu/sfQfO3pce2SzN6iukH0V8CKq1PiSf9sC8DX2Ju4Mwnx1d1Du7N1jZM4ubp5g9ziJ1t4faCF6DayW7vu6jFmvuUapcBvub0fUFnJVsD30rjOY603fT+f13PMT1nTPGuIJ7nM5coDo9bp/TPhmdtepwqJ55vuRkczTSsxnMdJJ+FI3oFA9mvDOwLuNMW+WTB5sP8eiB3sVKffZsNMxVjkQtHsTlNM7W9FLX3CxVzVWq/nA+ODdub6Yfg2sFZ2vrmV7GV8K5ocbY9ZX9Zl41+ly9Ebov2+90pymj/CraY/a76jVzOnM8dnNl7atwvaPGONNmqWpUuaDjIcsfRV1WbsrRrdP1rbI6zNUD+2kPSXavV/8c0VvtMTWeu/w1fxC0lltHTJwWMMc4OHONs8z0sOp5ph/mD9EI5+H+zJNR5WbIeuF0aqN5quJYU2Osuh4jVvxvhXO2q/cB53l3/TfR6XPH06WqVeWCjke59KNo9mZG7K53F/rSdusZ+OCt1MjQWt11dn3mGK9SzUDk9JwdXVirQj28ntuvscs/getRp3eZJ3SXU5wn2+s0hTnnZ3yWUe9Gva7ynB/mHfRTz46KUd6xsudKtO9uLlYY1cjy2Xx/+Td3ztGlH0U7uPOXsQMd8tlhp1/rVA+P5jsHoe48byN76Wu+gn8MeIzg9Tt7Avq5ZkzPDthvzkDG7CytkNVw2rHhmcv2ZHoGexdalq+8LiZaL9bcozpzjpGX9ejj/XwqnPPqyBjlFfVyPUvn3r7kvPqj6JMeKg5gd6iz/Gh/pp+FdRlnnHkIXZ+zly7Rl7CeuXZUeb02a6rOXOVnzsVKlTtDp0/uxer20aM6cTVJlgud+ez+6Astuz61Ua0K9o2x0pkP5jNN41m0hl7DXY9Uubth31y/GZNR3uGuQ0b5w9y/W38ao/l5G6/9KPqkX2IM7I7BHT1co3yHqDGqxYey8r4Bfamrlh30OrI92b7Mwzi0LE9vBu/rKdx86HPBXKUH2Yw6TRldN8Ndb2Z/F/Y6tFGeGuEsuJrZob6MysdrP0mnZ+yx6/0ZWIfxDDrPn8zTc9Hlfz+K3nTDvBcXO62Kd7PzARqRPbB8mHnMMrNnxtuh6hdfuJ2Xb/h4OOhxfqc53XkIPSN/sLovw83JGc2ReVTvzizzEeteepi/C/aGsdNGcWijWet4Zsjq7ai9g9Xezs6FzikPejLcnlnO7l+FvWbstFH8FnSW/wmBhtH6KnZeY2etjCeGs0v33p56yMjoJdvJVZ7DzLM7Mj9xtYjq9DBWfSa+EjcX/CPAXDZPTsugN+JMr9aO7B53wj4xdtBTzUg2V87j9O6h+z4F9vaOfpPqWViFM76zdkWn9x3PMeG7muw+/vWfz7KH4My6y8qeJ9g9iFc8PIHWnrkOH763EbOiL26eM3SPgznONa9Jv9Nn7k3X2b5RfBY3J05TmGMcmtODUf5oeo7E57QryeZAcR49nIderp0nyx/F3ozMm+kH7uMpRr2P+XBHRcdDurWf4so+rdTO9qh+Zh3866PoKdzN/XbcA8MH1Xk6ZHud9onoCzg7KujRuKpR+ahzD7XdZL3NZiFwOdVcPhjlKjr3FWddZ7hcVf8qsj5nPecMOd+sp0P4s9pOq/a+BfZ7NGcV3Md4pO9AZ/+qa1zBmbk4s3fEKz6KrvwBz8Bh30WnVsejzPodrkZHY3wXOjd8UXfgi7s61O9gjj7GoWX6m6mei3gxO93F6qXHQY+7F16fexwdT4espyTzOS3gfDlv5nG6Hi5HRpq7FnNZfBb2XDXqkdPzCtXeKncFO67nenk1d11nhsc/it74S7mD6mGlXj3civp4rOD2OW2Wsz3nw8uXcWhnr3MMXvSHuf7oulXuSPJOm4V9Y5yhvmyWQnM5MqqRoftiPbvnarRPnIOqh86bzRL1rifL8+gy430Dbl4YV+jMZbWoBZn+5b952yw9+lH0tl/GVVQvZ8ZXkT3MM7i9Tusw+yLmC35m3+iYpbuXPj27vU5Tsn2O0Qs8o5rVio6363H3XDHjDWavkdHtx5H0T2PmAreP7PIo4dd9XGfalezqXYbO4Jlrub1ZTdVc/s2w54w7rOy5isc+ilZ/Cav7noCDzfgpeB+dh9DlnbYD97JVfTdxHXdEvoL3GVoVK27/XWQ9pO7i0dyM8l2qGpHTc+bv3PMKs/2rvDM5xk7rznAQP4vzjzSXvxLXR6d9Erx/zvfb2NnznbXO8MhH0Vt++KvQAZ4Z5hnvWUbXGuWvZvZlqy//jr8L/6jwvmavRT/jIGq76+6CPWYcZLrDvcT/LH6IzO6b8Tpmrtftfeajxl6z//TRO/I4eJ3M56i83VodzwjXL6cFVe5o5LvsqOOepafZ0bOKq+t3uP2jaPcP3X0A72ZmkLOXsdND02MVtz/TiNPuhi99nQO+7N0xi17H7WdtXo97GIfm9CtxPXd0Z26UP3DNTt2OJ2N139W4Pmea05UZT8cb0D+7dwTrZ5zp/4graweu/ux1u75PZzQLV3PrR9HTP+wddAZ3lyeY8V5F3MPMQ76KztGZmdIXcnaEL4Ne1TOcP/QOXd9ddF/uLt/VHJWvyinO1/15duBmgXFAnbFq2XlE3E/nvrq1R/mzuD45LahyZ7lrbn46V89MxW0fRU/+kG8iHpjdL96dtd7M6EWcvdBX4fU6dStPlsv0Y5C7mmymMl1xntGc6vORMVOj8gVd3xlme5jNMfXM485Ea7k6HUZ7ND/yXoXrrdNmiJm5Y3aUu6/3JG5enLabbR9F1fAzvpq7r7eTs0N/Zu+bWXlpn33hk6iR1VSN16781J9k5mW/y5Oxch8j75N0+u1yoz3HRO2Ob4Wol9Ud5T+B7jxWZDU+ZYbvxs2L03ay5aNIb5I3zPinMhrkUX6Ee5Ay9MHr7jmSe+xqV1HNj75oOy969XWP2Kf7Cb3Oo3Q8I3iPV6Mv7Kr/Wd5pO9C6o2vQO/KvwFnIYO86+6gzVlir8io6Vzwir2fFaY6ubwbXT6eRUT7o+pS4/szeGe/b2dXnrE6mn2XLR1HGzpveWesqRgM9ymes7gtW97t9XW0FfflSz2LmdjP6I3DVvUTtTs2O58AfCZ4rRl6nO21EtSfuvfJ02FGjQ7cnWY+z/lNnHJo7E+7JfMGoXjBT6y5c3xlndHyu/hlGtUb5p9nd36xepp/h1EfRFTdE+OB+IqMBHuV3sOOPCveerRdkL1GnBVXuarL7VcLDw0FP5ttF9GxH74LZWVD/7N6jseds/V3s6GV3JkY+N2OjPRWsRTL9LFf3M+rfeZ2MKvcmql5XM/JGlj+K7vghR9cY5T+J0cOxE17HXXsUj8j8nZ5VD9FIz17UqncP3VtBfwWvMbvvKtwMVHRe6MR5qXXqulxoLvc00beq55muMJ/tUZ3nEXqPPCL/FtycMJ7l7P4Md68j1D+7dxc7+72zFtlZe/mj6Gp2/pBvo/viH3nOUNWtchWr+3bBmeELfRb3x0BrZbWdtosrarNvZ17G3Ms5Xpnpys/aeuZ6xMq9KVVvmNPZdDPlcsxTI6G53Aq8JnOj+7iLqodVbic7r7Oz1k9m15xNfxTtunDFHdfYSTa0mV4xejFHvvJ0cXUYz2hX4l641Yv4CvSPjLum6npvK/ep3pl9Z4m+cjZm+s29q+yqc4anr38U80YqH+eQxwwz/hnvU8ScPdlrvYcn72M37P/KvN3N9EdRl9Uf3O1z2pvRweaAj+IZ+CCdqXUFO+5ppfd88PgHoHvo/gr6K5yP13Wes5ztw1H88XDaFeh1Vq7J/dTeDucimxXVMw+pfCPd5alV9c/i+ue0oMrNELOj9RgTt+c3ctUs7OCyj6IV3vyLOstPfwiufNg5F4xV09zZF7Huj3WnvtMUrVV5s7zTrsT1NDT+QejivE4LZq9Tec7ee5eZPnEmsvlwOmPVZ+G1o8borFS5J3C9ddoMM/tnvBW76nypec1H0VseoCvJhvrP4GMi8pUn6Pqu5I7ru3lx2hXoH4tM1z8qPH4KK33W+dS5Xql1mGeDa2odZv0VrudOq+jOTuXhDLpjRHic12kOvVZnz85enIWzRjI9o6rntJ9Gd+6Crne2LnnFR9GZH+AnsfNB0AeuW9f5utpd6KxUL1fNdQ7nZa0M+j+B6KGeq75WuQzW7NTgzHb2BM7rtKtxs5DNFTUl04Nsr+qZx9H1Bc7vtDvgrGWaMsorI5/Lz9TPqPbvqL9K1udMr1jZcyWPfxTt/IVktTL9SZ4c6C7u/rraDjp9o4d/ELo4L2tprB6u6Xkb7JeL9SBOC1wuqzMi9ujelTqruOt3mZ0BnS83Z4HLV95ZeA+8nqtZ5d7C6gwei/13zNZx/jM/xy6yPmf6p/H4R9FussbwAc98SsdD3MA6bUQM/8pex+56d9Pt2R1U91HlMq7+2aLn7L+bhUrjftL1rXBFTVJdo8p9EnwH8qhwead9OtHrMz3nPsbkzLW+7OUjP4o6D3AX1mFcsTrEq/t2weu7B5Jxpr2FnTPRYXS96o9NJ0cy/Q24+bmaXdd74t5nyOaEdHxVbobRdY6N17qbN8/Cl3v4yI+i3cQDvPIgP/UQnb3u7P5Z/5WwTy6m5uh4jqZv5OneUzDjrbiqb1H3qvoV7tq8D8ahub072dU3x5naZ/fO7J/x3k3V9yr35ffw/Sh6GfridozyM+ysdTedF2+8zLMjPGfY+QeDOd6r82Swt1f0+Yqas/DnVPQDyP0unPZJuPkgozyhl3FG1/dGOB9f/pNP7u0KP/6jaFdDd9UhZx7GeJjP1CCz9Wa8d7DSJ/3j4o5P4209cXDOGK+yq8aOOl1mZ2zkPzu32exTY95p3PPT4KwwJpEfeb48x4//KPotPPEg3XHN6oXKF3TlPUv2h0KJXOW5mm5PwscX+pW4+qM/EDM88TOtoLOkM8P5cnM0M2MjD+9j5A/oze6J8R10+t3xdOjW6fqOwpvpX/bz/Sj6AOIPx+jB6Po+DfcCzuLQ3LGLqlaVexrOh551ZpxnBbeX1yK8x6d54j50XnfNE5+FnXXfRqdnHQ/R2VzZv8qd1/ry/Sj6saw8uJk/0z8N/lE488dhtHfHNZ5Ae52tZ1iZQ+Xs/jNU173zvjoztHPWOLtad0f9T2Sl13fOyC5mZqjr+zS+H0U3c/dDcvf17oAPo8bMdXAvf+ZHjDz8w8JjltV9O6hmKvtD4LQRK3tm0Wtk976TMz1b2Tu7p/Izx/itVD2tckHX0/F9eT8/5qPoUx5Qx9UPVNSvrlHl3gA/AthvxmfQj5VR3Y5HybyZvoOzvY392Qx1tTN0ZriD2z+qW+UqOEe7e9ytO8ofG+aY8Scw6utoLj6d6Nkn9u5KfsxH0U/kiofS1WT8KfBFzrU7vlyDm6srGV1L7yfWuofxiFn/DrJ5zfRg56y7Z4drdy2nfQJuVq7izmuR6A/PFR3PT+DHfhS9oYGzw86XdsaOB6m7v+t7E1Xv9SVf+UhnT8fzFOxjxN2ZI0++0Hfce8Uov5uZednxR4zPAH0jfUTH80Z29p21nnpWdvGpPV3hx34UHY1GjvI72P0g7Kw3elCr3FuZ7ems/1jc06H6Q7UTflSM5uDNuHvXmLkuq/tWYe+j/7vmgHUzRvmKTv2ncHMSVLkVdtb6cj8/+qPopxMP8xMP4RPXPJKXdqZ1X9IdDxnV1uuPvFfgZoPxk+i9uHtdZUeNt9GZnzvn7M5rrZLNQaavwrndVX9XnS/zfD+K/oc3P+Addv5hOT7wodSXNHvJ2KEver7wR/vpz+h4VtnVe50jrbmrNulqHVb26R7+DlfqdXFzNmLGG3T3uHvJnodP50xf3bPx5Wfx/Si6iTsfoDuv9QQrL+iVF3tnzyjv6P6h0fzI+0bcHI7+mESu8qyS1azuKdMrOr3dwcocKW6/05QZ3WlvYaWvK3uOE/vIrjpd3ty/K/l+FN1IZ6jdCzo06hWz/mB1392ceXnz6ND1zTKqO8rfQczDnXN4LF5vBH+WWVb3Kd2eZr5MD6rZrnIj3J7VWp9Gp+cdz5f38/0oeogdL9cOd1xjN2detLpvtUbG6L46f3BczmmOru+NvGkO33Qvs8QMfNosjJ6LitV9d/LkTF1x7ZnfOb2MP43vR9H/MBqss/kvPXY8UDtq3E11z1XubcRz8H0ezvFJPSedDyCX7+z78i5+Yr8e/Sha+YWu7LmL7x+C/cSL8vvC/By+z8E53Jw7bZYdNUbMXmP0bGf6l2e4qh9X1V3h0Y+iHbzpl/lljdGL8Wpmr9vxP/nzfPlcODM6R8yRKl/lOuj+7HllHGT6ly/KW+bk4z+Kjod+md9/Gj5H9mJVqlyXbo2uT+ns0Z+z4//y+4i5ODMfZ2tU85npHc7s/ZLz/ftzHT/io4h8H8T3cWVPRrWrF/4qu+t9qbnr/zDhy3/yxJw/cc0vX4LHPop+8uB/X+CeHT3fUSPjytodnr7+J/Dpz1XW40z/FLr33/V9+Z28YT4e+Sha+cFX9pzl01/Ab+CJvp1h9n5n/cfini+fz2zfR/6/H/qfZT/tfj+Ft/29+tQ+P/JRtMqn/pJ/K2/rV/d+ur5g5g9T+Lr+L1+u5qpZHNUd5b/0mf0giv+aMbuvyyf39vUfRW/65V41QL+NqqfxgVF5HCt73shP+Bm6fJ+n56nmjbkzz9iZvV+e4zf27PUfRVey8lJe2fObuOMhuuMaK8R9de+v6/vJ/PbnaXYG4uNidp9jR41Vnrz2ly8VH/1R9JYH67e/2MkdfbnjGsHMtWa8X34Pbi6cFlS5GXZ9QH35nXT/tlUz9mkz+NqPok/7RX65hmoGqlxGd646noq4zuh6Ve4noy/bq//3DZ+GzkRnhlao6s1ez3md5rjq5/vN7HyezvaFs/wJvPaj6Am6g9TxfJmHD03E1JUq9yZm7nPG++X38Ma5cPfEP4TO8+XLW7n9o2jlAVnZ8+Vz6Lw4O54ZZmvN+h0zNWa8b+H7Dwt9ruxvPCtXXsPxxDV/Mt1/SFdm/W/k6Rm6/aPoy5cr2flA8SXPeBdX1HwrP+GlfSU/fRaqn0+fr8r35b9Z+Wj6Mub7UfTl1/KmF++b7uVqvi9yD2eAccUub5UjM16lu6/r+0285UMo602mB6P8G3jlR9GbfnFvGMCfhPb270X/5uWYqN3xfNnH9/d9HZ2ZH+W/fBmRzVCmk67vKV75UfSmD5G3N/DT0N5e+U893dodz5e9fJ+pa+jM/Cg/w85aX3p0PnyvJpuzTFc6nqd55UfRly938KaH80338uUZOAOMK3Z5q9wuqmtE7hP+eF7N6OPnDR9IP5HvR9GXH8XOFylfzIx34Wo67e18X9Dn+cS+B517rzxV7rcy+0zN+t/I03PwER9Fd/6SfsJQfRqdj42OZ4bZWrN+R7dG1/flc7myx/GsXHmNjCeu+eXLTl77UfTEw9X9IOr6vszBnkdMXalyGSt7VnnyD9Qn8X2m/g83K7tnaGctovd65XW+5LzxefqUWbj9o2jmFzPj/fIzqWagymV096ivu0fp/hHreH4i+tKO9Rtf5E/TnaOd7L5mVSuuVXm+zPGm5+kT+3r7R9Es1S+1yl1BNmSZ/mWNzktylH+Czn0rM96fyG96blyvnRZUuRlmZ7LC1Vmtv7Lny/3s6NPqjDzF6z+KlN2/2JWX8sqe38TZB6Czt+N5krff35v47c/T7KzE8zW7z7FSY2XPl5/PT5qLj/ko+km/9N/OFb3c9Yfiakb3OMp/+bKT6rnJ9C66312H8Zd1/i78n+ev7PkNfMxH0afxHbb30XkJu5f3iNU9X758+fIE379POY98FF39B+Hq+iO+A/ffrHwsjNhd7ywr95PtyfQvv5PRPMTzNfLdwcw9zHi/fCarPV7dt5NHPoqOl/zwI74fN/t4+uXduXbH8+XLKtV8Vbkn2X1fnXodz5f386l9fOyj6NP5fjB9Dlc8nE9/5P024nn79OfuE2fmE+/5p/Ppz8Gb+X4UfXmE+KioXrhVrku3Rte3Qudn/Ulc9cK+qu7TxFycmY+zMzbaW+UqVvd9+fIU34+iTXz/l/zrjF7onfwZZvd3/B3PT+T7DJyDc6Ozz9wMZ/aS7F6cdhT6ly/KW+bk0Y+ilV8C92QP6Cyjl/ko/+UadvT2KXbN5pffxVUzs7uuq+c0x4yv6/1yPVf14qq6Kzz6UfSbqT6yqtwsO2vdwY6HY0eNVVZf4tWeKvdWOHeMO1z1vyPaXe8KPrHngXsGGCuRc/s+hU+YqStw/XLaJ/H9KHqQmQdpxnuc+M95q/t2MvNypO+qF2zUq2pWucB5nBZorvJdic7D7GzM+q/kzM8R3Pl8uH7rfK8wmuErcdc9+/PcRfS96n2Vu5IrrjvTjxnvJ/D9KPofrm7s6IEKnCc0l1M6D26XHTWuIl7sKz3TvaMao/wq3esrXd+VdOcwyPyMFe6pvCtovauuMeJsL0f7O7OV5Tp7lRnvmzjT89l37Iz3TXxiX3fw/Sj6IXzqg3cnVz3kn1Z3hL70eT6D+2OSaXfjfk53bx06HwqjfJcz11r5AMpwOae9gZWeOnbV6XL39X4r34+iH8D3YfnPPw7uZew0h/tDUe2trqmw5m7cH3DGmabwo+AqXG2nzbCy/66f13F2Jjp7O54juRd9Frpz/hbu6mX3ufvyOfzqj6I7HvAzD4jbGxrPV+Ae+CDTr8b1TDV9efOFntH1KSP/zPWvwM0He1bl7mDX9VmH8Swre86Szcjd89O51pNzvUrMxdW9zepn+pf38aM/ip5+aM88CLHX1ahyM9zxkngjZz9Ysj2r9YKz99WFfV+ZAfdHZqXOKnwGqmtXuYyVPWep+u80JdsXjPJK5sn0oFv/bXCO38Db7uc38WM/ij714byLp/6Y7abqs8s5rcOOPypPkvV4xxys7lul+yGU/Wyd/WTGO0s2L06fmcMMrZHVynKj+JPZ2WNXi9rKHD7FT+rziB/7UfRp3PFgxB8Kdy2nvR19UPUF7tbuBV+xsuduuvd2R2/vusaZ6+h+1mGsZHveysrsVn7mXPypsKfVO/IK7roO0RlhPzM6np/Aj/ko+pSGuYfAaTs5+5Cf2bsLPrjZegXuZzyC91bhfDP7g1n/GWY+Ctwflc6+ETtqjNh9jeirHrvp1B3lV8nqdu7pTrK+ZnowypNZ/5d38mM+ir54fuKDWr1wq1xF9YfLaUGVU7S2Xqu7/y6yj5lsXbGyR4mPq5W9XVib8W529rtTq+NRKv9q7pPpzkPmy/Qv7+X7UfQBxIPVfcBW/5jM+t9K9oLmx8jqR0ln79lrXEHWX+oaV7kO9DN2dDw7ya5XPUNVbic6P2fmaGYvZ3fXPdzF7r5Er2d7PuOtmL3uGd70vnqK70fRh9B5KO58eO6CD2n1wNLH4zcw6r/OiHo5O6M6K7jrko4nY2XPE4zm0eUZk86c73oedO/ZWrvpzABnfZaz+yuuqnsFb+r7Tr4fRU3eOABX/BHbVecuoi9X96dTv+O5A9dDpx2FfhfxB2b1PrhvFHdZ3TeDflC4jxU3T85HqpyitUY1ifOyBuOfRndG6GNMRvkv1/LjP4re/lDyAWA84uwfFLfXaW/E9dZpI/iHgQd9FR3PLCv1Rj0c5TucqeFmj/EMujfWTos1r8X8naz0d8TqHLq5Z07jDl3fU8Q8rPSdexgTN5tf3sWP/yj6RO54YEbXYJ5xaE6/g87LWV/w1XGGXXUO83O4uowrnuzPlfAPy+hnzHyZ/il0Z2+Uz4h9nWv8FFZmYWUP2VHjKn5L74OP/Si6olF8+K+4RpfqIalyFfFH0u13Gul47mDUF/bxLKu19I+WHhVd39WMes0PimyuduGuV5HlMz0Y5Uc83TcS97N6X9196nvD/GZ0+nt2llf3r+z5sp+P/ChaeeCylwPjO5gZfnq7fxSU0UNa5T6FO1/EnQ+XUa5TQ5nxVmivO2uNszmixvjNVD83GeWvQPve6X/lYY5zyHzGjPeT2d3v3fW+XMNHfhRlZA8rXyr0uT0d3B8Jxpl2FHowylfEvZ2p4dhd78u/53E30TM9ax+rNfdy3eXMHp7vIvt9PcFoProzFL6O92h8ODntU+j0dFfvd9T4cg+PfxR96gN1FveQOG2F1TrcxzigzvgKspcvNb7AeWS4vGrMOTqeN7Ojj/ohEWen0a+x27PK2f27cPOlcE4zP3PO5zxdRtevWNnTodNDnRnHKD9itNflnTaDu2c+G8zfQdbjTB+xuu8qHv8oOl74S7mCaoCrXDDKH806s+yuN8NoLna89PkizzTNKfQzznxv4Wx/q/3dnL7kZ9GZ17Orz5h6kOkzuPnRtc5DdyboY+w0vR7hPTiPwjzjLtU97YRz4BjllajXqRuMfFk+099C1rtMz5j1dzlT9xUfRcfJH+KN7Bzq6iU++5BezdX3Ur1QnXYFnT8gB/7o7GR3vRVcnxk71JOtO9DvnhG35pn5LJ5B++N65bSKzgzNeDreI7nPbK/THF1fcKYPFTvqsgbjYKRn+Z/O7Cx0/V1fxmUfRSs3trLnjcwMufM67TfDuWBcae6YIfxuH7XuNTIPdcZPk81lpl/F3ddbZaV3OkOu/9S7ni7ZXtbI9CznfLsYzQPzjM8wW+vMh9DKnrdy5TycZfqj6OofxtV32puohpU5xrPoQ3W2lqNbl55sn9NmGPWeeb7UHeoZHeHXfVkt6ozpc9dSXeOzdPugvs6aGs9n0XqcsdlrOL/TnoA9dzDP2GkRO90djpHu8l3tKu7s6+y1Mj/n+w1c0bMrau5k+qPoDt7+S+uSDXg1/J2XfqbPEPeQ1ercB6GP8W6ql/KVVH9AlNEfm7cw6lM2C9m+0N2MZXsU3a9nx2xtR3Udp+2iOxvqy/ydWk/M49XXm+2P8zttFc57lx1zfAVn74W9Z/xGXvlRdDR/eR3Pm6kehOpFfQb30DLuUu2rcit0Xugup/s6NUh43R5q3fqjfMWZvYTzpzO3e/54rTNwP++X+RnO7K3gbGgfueZBWIce7mV+laoer3kHWa8yvSLm5uz8BKzBuIPO9E/g6tnYVX/5o2jXDVTccY076A5117dKVb/KHUU+0zM6PV15ueoLu3p5B/oSr47w6p6sDnXGrOn2Zcx4R2jPZvsX8IXd+WPCvO7JzhUjj6t/Jd3+OF+nv8wzdprOW5zVk8XuYB09303Wz0yfZVcdhTU7s9nxfAJXz8nO+ssfRcfmG8m44xpPc/XA6x+HGbiHcZDpXfjiVd3hvGSUP0N2v2TF1/Gvston3edqUGPscDWrfcwxJq5+Fr8dnYlsNjpzo/mRt8PonipW9gQr/fu78A682n8s7rmTbp8qX5V7I6c+ikbs/mXsrreb0YCP8juJl8DMNZ2XGuPduB47LahyGZ0/IEHnxa+eTu1R/pisNfIo0b+VPs7soZdxB97r7DwT3X+mzgyz/TnQe7dXdefhXuYD1V0dMsoHo+teBXs6mpdR/hBPx+fI9IzufI7yP41sljL9DFs+ivhwZbkz7KpzFaMhnRl2erp7g65vlavrK53Zol7BeqND91BnTpmNZ+F9XIWbPZ1RzsIo3k11LyP4M+2As5Khs8QjgznGTuP9dK6RHfQQejKq3Cq7+reTbC4ZU3P5T2Nnj1krYuq72PJRdAweCMY/mc5A68PS8R9JXWoz9T6N0Qzx5a0xczvo1ON16e/UuJuYoc4sdfP0MVa4J/NSV7/m6HN0PDNkPeU8dOCebO9K3RVG+7I8f4472dXfmC09Zqnm2mlk9bpPsKPXrMH4CrZ9FI2444cJ7rzWKqPBrh4e5cwDuhO9/pX3MvrjcAxys+j1srr0ZL4jubfRvkx/C90ZdJ5Mqxjlj6Suw3m6e2eY6WFnHjhzLq8eh/pcnRHda+wg+tHty8g3yndhHcZf1tk1O7Pc9lF0PPhDfgrdB6rrq4gXP2sxzrSMGe8qOkdnZop/ENwRPu7RmJ5RzBpKpisdzyydvnX+MFU5xfmcFnRmlXHg9nZY2TMLZ6HqLXOMnabzyWs51Ed/tjfTg1H+LLN9op/xWbJ6Tndaxoz307l6Zipu/Sg6Hv5h70Jfwp1BzjxOn6nr0HvLGOWPpudqqhc+X+zuOEOnBq9Ff1ajq52Fs9TpqXoyf2fGjolaVbybq+sr2tNqFqoZCrL9jsxX6Xr9mWvtpNubrk+JmV3dS6gxVs3lDuQzz0/kidlSbv8oOk7+0G6v097AzCDPeFeprsHcKL4avohnmPVX8I+Bg57MdyT3lmmjWl3O9m71pTx66Y/o7OM1RnuqfJVbxfXQacqo95pzPu5nnh5X4wzVdZUd12bPGJNR/mh6rqY7z3dzplcdrq7f4ZGPomPjD7+rzpXEHxU34E7bSXXtoMop9DHehfY0W2d0PERfzjwiT6/G9HRiMsqvoP3hOnvpqs6cwv2Zd5TvkO3NajMO6M98HTgHXUZ7sry7HmOn6Xy6GqTKu5xq2brDjH9H/0aM5p84r7tPp2nO6W9mpm8jdtY6w2MfRceLfglPUD0ATncaiZp6zNDx08O4g76guzMQvpV9o2OFzl5eY3RdpynZPkf18h2he7ifsWo8k+5cunx1Txl6Le5hvErVT+I8nZlQ3XkO46twPt3Pe8pq06/a1bj+Oa2i49eZ5dHB+Uaay78Z9pxxh5U9V/HoR9Hxsl9Gxa5BHT1Q7uHg2VHlZuD1XV2nPQFf2HyJd2Zr5GNtkl2/g/N1tZ2M+lnNn9MczldpWc7l6eXcMu/oeEZUc+B66LQj2a9ozvm4n3l6XL5i1n8Ho/5lMzHaV8G9jJ9g1z3c2eM7r9Xh8Y+i44W/lCfgMDMO+DDzYd/BbL1Z/xW4GRq98DXHPxK6N1sr1Ol3e47F+z5D1asqp1Qz151P5keeDM05n8s7bRfdvmW+Su/MFDW3xx3qp8ZcrPWssB41F19BNluMz5DVd9qV7KjPnjC+gjuuMcsrPoqOl/5yskHL9LO4upXmcl2yF0aQ6Z8CX8zuqHA+jZnLPC7HvU6bpbO36ilzLnaaOzPPmHpQXYMavazt9ilu71WwN1nv6dO8xg71VfUynN/dG9f065o++p/A9dpps7iZJMx1Z3XErjp384Z5cPzro4iDvWPdZWXPT2A0zKN8hj6oPOircHmn3Ym+aLkezZH63J7ufl1zj9NIJ595Mn0n2axQC13zznM05ybbP9KY13uixvVVcK46fctmitCX7VGfy5+F91Bx1T2MyGYq0LnNjlk6+0b5ijN7n2ZlBnTPFevgH5fINtF3Bd1rdHwdzypPDCSvyVipco5Z/xX8GbwwZ3NRT+s6X+Cuz/3MVXFoPJiv4tBCdzVWyV7a1Cqf04NOTmus+p1GMl3peLqwT92e0edmx625r8qx5ujQfW79NK5vTiM6Nx06XjeH3Md4BtY/U2sG129qjDO6vqvhfUT8v/+miIYn4b2MYqcxvgI+VDsGNKuRXSfugUcHerlP88ztpurXH/NHINYO9VB3hN8dkVdftjfzEHroZxxQZ7yDbp/pYxx0ZzLLZ3rF2ZmtnolduBlgP13MPQ718bwC67l1xJ9Id0aP5vuWcaDezFNxZu8OOv2lZxS/Bb2vf/3ns7fw1l+eY+bB6FA9PJk+y8r+lT0ddvWaL2kSeR4d1NvZx2u4PRozT28G76uD62NXI9WsOtQz8mtdXqNbhznG1Eb5HbBPLuYs0EM4Y26PepirGO2pcnejs6IaY2pBpo8Y7evmR7P4ybxlRka89qPoeNnDNkN3sPlw0su4Imp1DxKay3XI6q7SeRHHOfNpzuUzqn0aM0+vg54qZv2rqPqmOe3xaA/zrFPh9odOXN3R/iqnMbUuVQ/ZzyrmXqc7j4upqT46HJl+DHJ3strD2Nc5uM/F1DPU79ZdZq97NdUcvZFXfxR9IjqIHM6VAQ9crTiv1gy4fxQH1bWZy3xnyB607kPIl7/bV2mhuzX3KMy7mDjtDK4f2rOqd/ToOfNyTVgr05hjTD1wurvfM7CPhD2uYod6Mj81XoO56nBUuUOuMarzBnb0fjRDVe5o5IOuL5j1f/k/Lv0o2v1Q7Ky1yuqw6T4+SKOakc/OV5DVzvTD/FykynWpZsDNG1/01cuaOtduj8LavB73j+KncH0609ss15ljzdHPe3J16FF9F663hLOka+6lpjFzlUdj+lwNol63x2l3M9tHN0O7iFnr1rziHn4qd87ZpR9FwY4fqFuDPsZnuHJ4Ry/3XeiDywe4e92ubzfRS+1p9sKOXBdX08Uu56BP9SoORtdw2hW4XuvLnPnZHOdPD4U+5lhvVONquv0n7Hvlj3x4uOaZxyrcP6pZ5c5Q9ZM5NyPM6TEL9zAOsnt4M+wdY6cxPsPOWh2WP4r4ULj11fBajD+ZzsM5ypPMr7peN/PPsFLjbB+rl3DkeDiY09itXR31qOa8hJ7qOiuMZszlstngDF1F5zqZ/hRuBhizr7rH7eeaMT2O2FcdhBrjpxjN8iy7arnnZVftJ+n0veMZsVpD92XrjPSjKCs0u1bN6V/+Ez4w+rDzrPnOsYJeM6tBnfFZ3Ox0tIj1IMxXXiXy9DIOTc+q0+/iDJdzWpD1sKPpHFAf5VR3noosX9UMqvvaAXvl6OTpYaya+t1eEp6O11HtZxya09/CTO91fkeHgzrju3jqumdxc5TNP9fqd+uK9KPoKro39lvoDmz4uv5ZZuvO+q8iewln2uxDEnvcvkrr7FEy/S705c7eOn3mj4HT3PWcJ/PSo+tK4/6gm1+F/WUcVPOT7TnMnHLtfNxDL3G1Na72PknWO52rs3B+dtXdxdvuZ8STs3T7R9FTrP6Sdz44I2auow/f2cOR6XfDvmUvX76kCfP06IudL3k9c59Cf6ypZ3WcdhT+K9C+ZzNA3e2hJ8hmztWYgdcd1XC+0Z4uo165vJsT6ppzWqwZOw/h9d0RuQ68j9CuRucrmzUSvjNH1NGz1n+K2Wu7vt3NHdevrvEfH0WVcSdnfvHc52qN4jfC4eUDxnymnaFTr+N5Cp2FWLv56OQD1tM1j1V07456u8lmUGP94+AY5QO9Fq/rrkctg97Q9HyWUc/Y11GvR7rudWvup6dilD/Mz3MHqz1z/X8K/gw8c61k+k/h7nlyPPpvit7wC7iabNCztcKHJfPtwNXOrst4xKw/Q+elmh363NHB+ZwWaG1ex+kur7CGY5R3/G38gWDv9VztnfWPcHup6TWqnMs7Op4M1+cZ3GxE7Gp3rhMe+lTn4eheU3OV70nO9HhENYuE3plZreZ+F+zfKM60Dqv7zsLrPvpR9EZ2Dlc24NUg66A7qr2rVLVWc2cZvXgP88eBOM0RdXhoLoN+1dxaPZmWkdVysVL1iTnGGZzDznzTEwdzGeqPOGM1dwWuN5XGmXCzRJ/T6FeNh8N5Mu+RXG8n2awwrpiduVlYk9fieUTcb3XfmX4XO3q+o8Yu/nn6Zq5+kO5Ah9ENaLZeZUcNBx++Cv6cnT2r8KXMeXE5ejTPo8ozR59CPfOQrhYwx7gi69OZGT3r1xniPPHMdcWsb+UaQdYDnYUzWlY/UE+27hJ73D7eX+UNeD8zzPQh65+r4Xp+lqxWpmd0/Jkn09/M7ExczWv+TdEbfjHdgcoeuO7+jNn9f/FPEasHa+pZdWpPUs2LvoDdS5txaA7uz+IKt0fPmaY5pzvYI9dPatzjcvS4+XEe6tS4J8tR6+YU5v/K/TvtClwf2Xudl8xPb2fdPRSNWW8VrdOh0w/nyXpJLXxnD9b8MqY7A3fymo+iT4NDrzFzFTPeq+A9jGIyyivuBaJUDwlz+nLmi53egC/57kG/xlxHrKgvztkenu+CfWGcaUR7zLNj5Mn0Y7A37kNzlZ9U+zuwfzO9Zhwa52d2rZrCvHoYZ3B/tS/0LE9mfu8O9lH1nbAe4y//R7f3d1N+FOlQc62e0bpL9RC9kV0Dzzp8cWfnVeIFoS+KszWDM/Vc751GnCdmiUeWJ9TpnVlrnHkyqtwI9sDFOgMZ7CnP6qOmMN9ZK/TQF1qWczHPFZnH9chpAXufxU47s1acduB62ZH57ibrMzWFMxTnbG5WcDPFuvTw/NNZmRfds2vt+NdH0WjDHfAeGP8U3EPTPV8Ba/81H03unkmmr1K9dPWl7HzMOQ/RvNs3u45Yz4T5zKd0PEfSD2rRa9fzEbqHuFqZ94Df7WWceYJR/kjqreB6F3PgcorLc62elXXnqNB6er4T158zmqPrU7iHcfDXzDXP6v103Iw47S38+fPn3x9Fx8tvOviEe3RkD8AsWmf2cFBnrJrmnO8s1Uu68xLvvrRZa3TontW1OwfU3Zl7HK7PWUw90Dxnh2u3T9c8az7LEfWS6poKtSqu6lRU/dGc87k85+fsutI05/LURtfQWpHfwWxP3OwwJrGne8QexyhPur5ZZu/jCnbNwJXYj6I34B6qINOvZOcg7ar19jqMzzLqO1/Q1LtHVovazJp0PAd8Z+FL3FHljiKveuca9FPnWb3UMlyN7l6yus/B3rO3Lk9tZa1UGq+rXrfvKPSjuIcgy1W/c+bcXDgtdD2/EXePO+53R41Vsj6/jdd+FAVP/SJ3Do8bcOZItedK3EPY0c7e52qfqxe3052fcE+cd60d4Rn5glG+Q9Yz9rYLvdmsKIwzjejM0c+YxF76WI/5K2HfdRZ2wPmiNrqW8zjtDXT6dmWPd9fUertr7ySbhbfOScY/R/HDnOWqulezY/D40HVqZh59gfO8A9YcPYROO3Cfqp2BM9R5kTtP5j0Sf+h6Xl2zNs8k04NRfhbXt9CzfCen61UP6zuPWzuyvF4j8ygdz5H02Wkuxzy1mXXsrQ7dw4NQj3XmreIzcDZCI5wR59kFr1WdZ+AexmSUv4qd/VXurPu//6bIJXdwVV1y13UqOIiMK9wDw/3OU/FX/qhUR3jfSvWCDuhRL9c8Ml3zO3B1d19jhPacqL7qcZqS7c/Wlaa4fDbbo2sdTU+F9jfOmZbFbp+LR2uSaTw6OfXw7HxnWOnDiL/mfZgdFZknNHd2WoeRb7be29k9R0FW95b/fJZdfJZddVbpDFn2cBD10M84I66VHSu4B8rVdNoT8OXLGXEvc2rc43D7u+uIeWa+o80w6o/rdUXmc7PA2prnWen6V/e6fUej9ojVHgVuv5sVzsTMWq+hmupK5nex4rQzjPrh8jEDnIkzfdYaPIjTHHp/7lwx472C3X123HENsvxRxAfGrZVM73J2/y52DODTw7zrgVa4h3GXbp/dC1lf2HpWnHbgJa+1s3h1TTRPnWTeEa7PbgapcU9Q5at6jo6H6B7uH9Whr+ufgT3SeDQX1DK/i7P1SOvoFR3PYX6mGdgHF1PrEnvP1LiCzr105/huVnpMshqcz9V1RvpRlBWaXSuZ/snMDOXqQ8drVDX4gLsjQ3OVb0TnPiuqOfmDF7zqCnO6j1p1aC1qM2tlpI3yq1T9yHqmM6Nnako2Z25/pvPMfAb3dWo7T0aVz/p9mLlQPcuHxrib01pcq8bD6dUe6pnX6U+S9TLmwB0jMk+mO2a8b+ZMf7O9nLfVdUX6UXQV3Rv7JHYNsb6c9ew0d02ndXF79UXA8yzZvmxo9aVZvUCdjzG9Tstwtc6uM+0OoqejflIf+St0D/czpuauS83VOBr5WWbqRH9ne+vmI9N5jdHaHQ6nj/YoXd8K7AHj0DgvqnE9Q+anzms6T5fVfZ/KVbPT5faPouMFP7Ry9cCNHj7q9DNP3MM3S/fBze6JMRnlq3nQXKz50mWsWvUyz7Ts0PxoPaqneXqZU20n2s/V3gZuBqu5ctd0HqdnWpDlqHdj97OdJZsXFzt0L2txzZnKPJUvNM3RS5y2A86F9mZ3nxwxD3qErmcHPZX3aOTJrP9tXDUzMzzyUXS85IcPrhqkmcEfeRwreyp4v6P7cnnGu6lexu5Frmg+O+iLWPXROqObG9Xj/e3C9T/Lqe40t46YWuhuTY05rcdc4HRej55RzRmqPrkZYj6bhWpmuId7A143YqfF2uku3s1sL1xfg0y/Al6Ls9WZ4aDKV7lP4MrZmeE/Poruvqmz1zu7f0Q2ZJ0BrnJBxxPMeB3x4GVHheY7P/sb0Jc4X+p8eavm9O6a+3g4T6z1nJHVqWCfO/0+Bj3P6rg9GjuNjGq4tcZ6ncyfrSOmtoNqXhQ3M9l6Zo/zaR3WVM35NU+oM55hpheu74xdf0NzxwyzflLtX7mfOznT42PD/lnc9UJ77N8UBe7mdsHa7sFe5eyAjvbzQY6zrrt0vB1PMOPdCV/Sqrtc5lEyjfUYcx2xrhlTpyfzqd7lTI9mZ8zNqsZOYz7zVLnOutp7FdpDp8eavWae52yte5hzR9cfGsnq0FOd6TtL1u8KN1tklA86szdLtj/TP5VdM3AGvYfHP4qOi34pV9Q8i3sI+TBV+ZGWHSNflqde4faswJdtnN1LlS/j7lqPFY3riLmPvpEW6PoqshlgDzWfnY/jOP7rv/7rXzlXy61Vc7U1l9XgvuwcUGf+LK7XDuZ0fvTQnFu7GppTT1ZD45GmsVt3YoV1R6z0i/NDvZvnQa87Z5pjlD+anjOwF4x3cVXdM7zio+iY+OV0fW/GDfTMw6N5fTA1zo6KLN+9n5FGshfq7Asyg3W4dnnVXI5r9bn99LkczwprdOj87sloPrIZiI8hRb2s2/l40pruutTcPRzGx2sxrx56z6Kz4WZiNBedNetlOdW4dnFo3O9q3U3WI+qMKzgD2aH+CvXzzHWlkY5nJ7v7vLveLi77KFr5gVf2kNka+nBnzAxf1zt6KByVr8p1cPud1mF1nyPrjep8QbscD0KN9RjP+LJ1wPwusj78NS9pzR3m44VUHy5aP6j81APeg7tv3eM0R5bP9CtwPdfeU6M/W+v+zuH8Ls5Qf5y7e84y6pebw6DKdalmj2dH5x5G+RGd+8jI+pTpwSh/ND0VZ/dXXPZRtMqZH/bM3h1kA8g4yPQDD8zo4avqdHD7eR2y47qzjF7A1F1OUR8P5jXW/SMf13rmWjV3LefdRTVjXc3NhPNrHJr7AFKfMroONZ6ZP4v2a9SjzEfN1eyuFafTX12Hnogd3ENN4TW6uN4HmX5gXzVbs7i6PHOd0fGQzp6OZ5aV3gWre1f3zbL1o2jXTbMO4zvpDNQuT+AeJp65XuVsvdGeUV7J+tx5IQfM8+UbsdbL8k7XmH763Jqx4jTS8XRxM+Wgj2euGbucy6vOM9cRM69HaHrO9mVUubO4XnJGnNZZq8Y40LWD+0NjLqtDb0aV68B+Epfr9n8n1XXcnDL3ibC3jHewu+bpj6LdNxTM1J3x7qYa5qPQHaOHO8h8oc8csU9rVGvViNbcQbev7uVcxTyrh+ustsvNrN3hqHIzsOcO9pheV6P6NzwuRx/rac5poxrc243dfZyF/c36SA8P9cysXX3nq2LWiDXh/owqt4uqj1mfQ585SGjMaVztH5HtWa13N3f0fsc1Tn8UXUnnB+x4nqAa0iqX4fbwYTtDVsutz15LYf/44tW8W/PMNePMrzrX9LrcaM06QebL/LO4F2YWZ15qoXMd5+p/P5TVOwZenoPM43xVHMRezTMeUfWN/VWvW6tnZR24dXgZq0/j0Ii7HnNcXwVnYUTX5+CMZFpF5uHMufpvoNvTru8NLH0U3fkDnr3W2f1n4AAzDi3TmeeZeWrdg3uJ00jHk8EeMVb0RU2fy2Uv5fDycF5q9HfWjLmehbVX0P6P+kffaHZcjpp+PHX/rRL9AWur5moydvs7ZD3IdAdngXNBbXYdZDoJH+u5WlmdKqfQwzij2y/2XOFczBxaI6t/DPKhM8/47Yx6Nsq/jaWPoi5X/jKurH0lfKDcutJGrOwh1YOsdDwk61umB1WeuYj1xc2DXqXSWGN2rbWeIHsRE/VxHrhX81VOtSzvvE6r1vQT5hnvhjNDmFO/m6Hdax4K49Coa03itDtgX90s7YKzx+t0r93x3M1T/VPuvIdLP4qOi36YK2peDYedceAeHn3Q+NCF5vbNsLp35bqz/XMvXGqsyRc389TCnx30aNxZnzlYcxXXJ2oaV7mI3Twyp/6A/6bnaFybNd1/pqPm7lHzFaP8CNcz9lbz7LfTVtcR6zqLXc75mOc5Y5S/AtdLp3XR2eJ8aV13ja62iruPO7iqr1fVzbj8o+h44IfqXq/rO5oDlnkynVQ+l3PaCqOHaPSQZ8x4D9OP7AWcvXydP3Rdu0NzXHMf8521i0mWy/QZRr1gjxnz7NaB/qcw5vkxxFr0MMf1YWoexsP9zAej3G7cfOj8qH5mzdpBFXPN/YwV6owzbQY3X6FX6J6RdxXeF6/nruu0w+is7Rjlr+JsT8nueh1u+Sg6Nv1wO2qcZTRsHNiOX89dRv64j87RgT9Td98q3V7zxexirunRnFvrodrK2tVkPjxXwf7Fmj2tPKxxJB8zVa6qfyz874y4dvfIWOF+amdgr7P54Cw4fWUdhM48PZ0145FH89R3MOo3c4HOSnWMmJ2Zjq/jCWa8q3T71vVVnK2xun/7R1F1I1VuxJm9u+k+JA63L9PcdTRmLsj0DH2Y3Zq+K8n6TJ0v3G6emjuynOqzax4Z3HslWT+rGWNMjflRTg/m3Dpip7l1xNSC7PpnYN8YOzgXbl52r7Mj8wS61pjnLD9LpzfOU83EClWNyLlr8kwyfYTb57SnWO33Mdhb5XbQ/ijig1FR+arcGa6qq3DgGJMsn+mBy7uHLdY8Zgg/a7r1Tqp+MTeKA6fr3Gq+WtOv59U1Y+ZCI+HL9rk9Smcmqn7rfHBW1Mv/hOX8LpfpPHMdMe+PWuXfDfvUgXuymNrKOou57niyeBVek8z2q5ob1XQmOnDORvuZ4/wxP8r9Bqo5qHK7aH8U7eSOH4w8cU2iQ+4G3j1gqjF3lt31lE7tUU+YZxwaX9p6dD2hc+1qnFkzjrXmd+BmqYJ+95+7XNz5GNJzpQWMD/O/U4q15lSbrX8YnfEsrq/ab84AfbrO/DNrakF3PYqVUbzC2X4cyfxwvQJrVrUZO+hxdXey0p+VPSOuqDmC13zko+gwN3IlvBYf6CvoDG/lGeX0IVk9rmB37eiTO7u1ehzOm9XSHLXZNQ/mAufbCfvDXnVz3Q+ROFcfNQH18Lp74plrB/PuXugJzeldtN+qZT12M5DFM+sgy2drQo21GRPmnSfjTB9G6LzNHrFfzyTTj2TWSZbL9C4zv/8rWbmPlT0jHvsoOjb8QGf372D0IHTI9lLnQ8P8DrL61bV3/A4IX5xZTnF74qWrR8ejXvpW1oQ5xlfAnmX9oq6xfgxV/5ZINbdmXP2PqoNRLZfnmetOvINsJrimL9My/2gduLV6ne5qOX8WK7zWDFkvz6x3w9qMg0zvcnb/iG6Pur6Mlf0rezr8s6PwmQHnwzai6zsmvWcYDeZf84IPQs/yR7Kf8Vn0Gtk64tF6Bu3/qF8uHxrPsWZM3eXpjfPZtR4Z9HCtvhmy/lTzp73nB5DCedHzkXzwaK6aI1e7+rdUmmct1bRulp/hTG/YY+53ecad9ejQ6+laY83Tp2Q5+lbhzDDurmf7XMFavC/SuYcq91a6PXazdxUz1yj/TVFVqMqtsrPmzlpncENdPQzdB4nrM8zWueIeHK6H+kLWNc982Ohxa+6rtNl1xIHLZx6uqbncDJxHnUGu9UyNs1B9vLic07LzkXz8ZP+Zzu0nHY/D/f5dT7nOYt1Ljfks59ZOy/J6MK9nt9ZYyfRVXJ9U66x30a2vs9iFfsZ3sLt3V7DrHsuPomNwodVcBR/I6iF7I9XAjh4I5hgHfADPHlndJ2CvI1adHtXo0/nhTPFwHtZeXXfiDtkepxHtd9brbK2wDn3u40bj0T3wY8ddw+XoqfLZmlQ5h+tDNhOxZswcNeazHH2hhc4192V5xvS9FdfzmI+zB+tW6yrWMz1KlVuhU2+lvzozPGap9lS5WYYfRZ/AyoPZGYJZZmrSyzjI9J1kDy8fVK6vIuuh09l7PnRO0330urzzraxHcSe3g6yvjLlWj1vrnurDhmclu26cncZ9Tnd7s/wqVX/YR64Zn93LNf3uCLhP99Ljck53+TtgT7M5OEtVK7teFlPPcHO8wuzerN93Ul3/7P3966PoTDGys9ZhHsRMGzE7BLO4FyzjIPNy4FXLcquHq8m10+7C9ZgvXHqcxrXTmGet1fUo5jpit56l6hdzWZ91DtxMBO7fElV7439T5HRFPdkHF/Mj3P0HWm+WrJ+ac2vuc1q2ZswctVgH9LpDvdk6q30nnCWuic7P6qG13Lmi6+n47ibr8+6e76w3qvWvj6JjsGk2xwdqlSt/4Vcx82AcTd9szVncA96Nd1L1OJuFbE/oOouxHh2j/d31KOa6y8jr5iXTInYfNtma58N8lGht1bjm+UhqubXT3M/Be3HX3AH7nMG+c+20bM041tQynWunuRpcP4nrZ9bbTD9LVZf3V3k/hWwOzsxEzBpn7mr+/PnjP4pm6N4wf8i7f9in4NDHS5l6wIel83DvQO/J3V8W87yT7gNGX/fI9mfns+ssdtCr69HeCvbL9drh5pB73QeMu17AjxbeB/N6znL0Ma/aSB8x0wfXR+aZy7RqzdjVoVc99KtWHQrju+j27Uhm4iycI11n86YwzzPXZ9lZ6yzVPI2o/Cv1Tn8UkdkbmPXv5O6h4PUYE/dQKJk+S1VHH+YMvU89dqDzoQPOYc8eKPVTY55rHpmvs87i0HStXueptIpRT7Rv7KlqXGcfKHGm1vUH7j+XZTnCepnPMeNlf7ke+XgeadXaxdRcPnTmHNTpZ0zfTkZ9qmZL81fQrc3Z5rlL179a/wpm52HWP0vro2j2Jlb82RF59f40sodVYd7FZw5Xk4zyV8Gecx6YD6r54XzxoDfb01lncWgux/gudBbc2n3QUI8c50Vjt+Y1eR7Vc156qvXoGkrWQ2pca+z6y5ybiWrNuPJQj3WluToaV3R9FdEj9jfrU8B+c5/WnTlcTdUc3HeGnbV2wfnQtTtmmPWv8M8xOdjHTTcWVNfiL/wuYvh4XoF7R0OuD1+snS9QT3bQr2eS6Vcz6i8ftPB3NOr0ZLnOmnFQ5UiV66A90/66deZRsg+jiLlfPfw3O8xTc2dqvB/mnYdraozPwnlwus4Dz901D1eHeXpGmtuvON1pVxI9rPqovR8dGcwzJvRS6zC6RsXqvhFVf6tZ2c3sdZy/9W+KRrjCTjvLFTVXcQNNjefMV+n0zDCz13mdFlS5oOOZIXs5E82N9sSamuou7qwZZ2uN9Uycd4Yz/fiLF7F+3HBu6VV/5PWc6dUHFM9B5XVrxjvRfkWcrdlXd15Z66H16HGx+lUj7jrMk8w7Q9avTA9GecdoT5XnnIVWxaTKVzml6+tytn87mL0HN9sk/SjKNmTM+me5uv7ZgYn9PGf5EfTpg6W50N2xgts7indRvVgDl3da6FxnGg/qGnfWjLkOmKti7pmBM8O19t3luT4W/pNZ5XcfQOpT3HX+67/+q/xQ0zX3ZhrpeFxv2Dv2OFu7Gdi1DrI8D+7jnoyzeWX0+6/yVe+7xFy5Qz1u7WKH3mfHn3H2Pt4IZ4XxiK4//Sgi3YJ3Ud1PlSNu+KqBOkNVdzZHzxnid8CaqjGXaauwZ+6lPMLtcWunVWvGozXjbK31FeqMg1GdY9Ajl3P91o8ZftgcZk44S9zj/JrTc/VvmLhXc5nm9ro9ZMbreqIx86O1+mfX3TrOq5o7Mjqeq2B/GCtufs7g5ow4T3bm2sXBSM/ys1Q95UxV3jtYvX77o+gwDwqhxoeoc3To+mbJhvHMQI32/i3+QIS2i6jPg2R6RuWtcmfg3GQH/a5GrDPdxdmacbZm3GHGexS/+2zm6Nd4tHYz0/2giXP1b4tczvkYZ37uCbKaI1xf2C+us5hrp3XXeszorp6D+zLfWWb6wR5m86BadczC2g53HxmZl3GQ+Z8gmwfOTOfgfuK0FaY+isium1BcTaddRTXQjK/E3Qevz7ii8mb1HfQwdnQ8I6oHZITucXWoOT2rs+ILqDuvxjsY9UJnIY4sxzV9x+DfElHP8u4eNM7yvLaDP4s7jxj1yPU81i52B/27104LqNN/NVUfslymH4N5cpz1cJ4q76exMgMre0bsrHnqo+gqqgfPxdSuYNdA/zV/bDKqh1fvp3NUXleXsfMqVa7iqv65Gao03ZfpLlfVy9YRK8wxVhh3YH80Zm/pddrfv3//9b/h0Tr8INEcz1xHTN/o3zy5fKwZOw815rqwf6Fla3eoj9qOdcR6Vpx3dKhXYX437C9xveU6Ozqe8GV+5jI6njfC3rrYzcnbiHv7j4+ilRvmHsa/FQ4440wbMfugBSNvlWeO8RPwQeOhPvW7/ZWXPrdm3FnzoM+hucpHsn5l+iE5zhv3VDHnlLUO/A+k1UOf1q3+Uxpjl1Oq3Arsi+szfdyjmpuNHWt3ZN4ubq/ud1rou2APGavmcjtYqevuKauT6b8FzgtjpzEmmr/k3xSNbuAn44aaZ+Wv+SNQ+Su0Fo9ZRntcXcY70Bdp9lKlTzXNOZ15t2bczbk1z9RGefVVuF64uXJrd3ZrxdXR2Gmk+rdLqrl1pmV+1u6sSbcXgfaTcayzI/PvWKtW6aFdQVa7+v0zV/Ut+k19hO5zB3HaIbrum6lBjfGdZL1SOp4VOnXPesqPIvcgML4TXtvFTrsLHXZqHdw+nt06e7i6uHrUXXw13d7Rp3PAmaDujmwfc4w7a40dLp95V8h6yJ47n8uN5kVzGh/F/1Ub67FO5mfM+6TG3EgLNFf1xuU4B6Exrzq1HWuNNV/pzPG4k6o/wcijc7VK1KjquNnL1k5z+adhvxln2idQfhS9iR2/YPeQP8HoIcmY9Wfog6z30qnZ8XSp+lDlDvPiJi6v/XezkOU7cWed7aWuOG0X7KWL3awe//NRwpybJTdX/KDRfPWfxnhWv/Pyum5/Rdd3oN8RO505aoxdnR1rp8VayXQHa3f23AV7qXHMipuZGUbzR3gP1BgzF2T6Lt7Sx7vuY8tH0V03uwPeK+NZsmFW3GDzXOH2c32GUZ2/xQNOnfEKo564/OhlHJrm3R63HsXdtasRGvPUNeb+WThDjDOY071x5v/4mv6AHzOZprHTAv1II9Ve6juY6U3We41d33esnRbr7Mx9I2b9K7CHjGfZub9Tq+MJ6GX82+jMVcejLH0UuYtQY/w29EE9e6+dwaw8zGUPVbZWrXNwD9d6pn+VTp3q5Vm9XJ2maJ+rdaVl+ZV1B3oZz+B+904LOBMa83/3E3BPnEf7OXNH8nHEmgqvrZ7KzzPXO8nmZ3Sod9c64iwfa4X7VHO5q2A/R/1i3u2nR/XRwT3dNc/V+gnu7Okq7v6oMe7Q+ihaKXyYh+buB2iGO+5tNOj6oM2sVZvB7dHaM6g/25vpI1xPXL84Y5qv1rrf5WdqjdadmjNU+9iT7PfvcowV1g34n76U6kNHqerpfbp7DioPYyX7uWbI+kwPfcxznc3KaK2H07N9XCvcP9KDKncW1+ugmregylXEdd3sqM51xaxvB+z73ejsZMcuRrX+9VE02rCLO65zxzUqdg5tF31IRwf3daGXcaatMOph9dCEpnmunZbl//z5c/zzzz//0mfX7mDexaG59SxuBoLQ9azev/h/E6j64NF97t8S6REavbxP+nlvGd1ctt4Ne5n1nlpn7Y6YXR5ZDa4jVhgHmR6wbmhncb1zPeTcqNY5MqrcGa6qy985409i173/66PoaBZ3HqdVxIPBg1BzvlE8w5m9wWiINX923XlYV4ma2fkJOv0Jj84K1/Qz3zlc3Wyt5wrdd5ZRn7J+MnZa98Ml+xhyaz0fk9egV/2szTN9I20WzozTOTc8NNdZj45sX7ZWTXHaUeh3wJ4xPkvMlx6hq2d1XWm7ubJPrM1ZymZqBPcwPsM/Z4qd2Vtx9he2iyuufeUDdBXufjvxDO53XWluNlZmRv2szRzXHQ/9jJmr6HhGVD3S2WTMs3oYq5f7XM7FoQX84HFn5hmPdObPUPVqpueH8Y/WI7/TOuuA9TMtdOK0Dt3+ZD722c3cGbTW6trdj9PO4Ho6otNbxkE2G2+B92X/TZGDGzO6vrez6+eohv1IHoyV9Rn+Fn+8nsL9/jOND1ymca0e3ZMdmS+rVfmVbn4X7DF7zTiIOXF5zhBrZznWdPU15v/OiLjr6TnW3M840xzsjfZ+BD2cBa3l4ljzP405TyfPdcR6Vjoa690Je+jmYTectZm1kunHINfhqn5cVVe5+hrtj6Jj4ma6vp9GNqiZPkvnQQr+4g9Odeged74bNz/UGAfU9YU8s6Y28lHnOoN51mR+layX7D99Iy1bR6zoB43bU611v2q8puay+E7YS9dP5t1BH2N3qCfWTou1ejJc3mlXw9mg3iXmZ/Yg1LN1h5F/lP/JzMzajFdJP4q6Bbu+3wIfDoV69uBk64hDi7U7Zrjaf5Zsxqjz5Z69/EdrapnP7aGeeTQ/y+w+zssI+hhTc2v1u/9RNj2uRqz1TI1r+jR2GvMjzeVm4QyM+tmZpczn9jCfwX3uUK/bewfsaWhVfBadN86TzgqvS69buzjI9C539eROuj9TdybTj6JPovODdjx3kT042UOy+iBE7dGhfneuYI3duL7xZazDzrXTqnVXc3nqmac6RmjdVVzPGDvNxdWsqOb+LY+DNellXjX1Mg7NxdxHjbkr4BxwJkZrxm7tNB7Oo2eieuYJtK5qq7Afo7jSRseIWU9nTarc0cj/Js7M1fRHkbtYV9sB6zJ2GuM7yAY004PsAcn28eHVY4bw85xpLiaj/Cz6Uu2+YLM9bt3VsnXlH8H92THiTO+dpuestvMG2ccQa3Gv5rmuYF3qrOPugWS60ukRezlz6H7WynL0UeusqdGnOtdX0elH4ObBaSN0ftyhPkfmydZKpjtmvJ9Gd7a6vozpj6IMdyN8qD6RM/fOB2YnV9UlcZ3snPlWyV6wVR84Y4ypZWuNs4N7ZtbV8TQxq+4lzZ46D9fuY4jn8PF/PK33oWv3n+HCQ7L7UkYevfYs7O+ZHrtZ0jVj5qhxHTF1zQduT3e9g04/Mg91xmfZWU/nX8+rsN9duMfFTttF1M9qZnoXt/+fLHEU+go7a30aM4OtL+JqredPROch1k5zZD4+PIypVWv6Mx/XmT87rmI0GzpPqumZ64j/4v/xRvVkHy6Bm2fmHN1rsK77mTrM+pXVvnI23BzRl+XooxbrgLrmQjuzZr0nOdPbLjqH1dqR6b+N0cyM8orzOu3Qf1OUGTJ9hbc9HGeYHVz6+XC4fLWm/wy8Fz1fAV+YV6Cz5q7BfHa4fLa/WlewPq91Bjc7CjXGoTk9+zjhNTlfbl3tVzTm9TOv06kFVe4s7C2Pyqd65hnlmO+sqQWzayXTdzPq485ea61sft3aaVlcMeN9OzpzGVk+02dY+s9n2YUznfBh+zTcQHfhg5itncZ81Dp7sL6enXYnnBM3M84Tsc4aD3oj5j7GTu94qOv1Mkb5iqqPCjXOhWpOd2vCe9A1P264pl/Ps5q7R+fdheu5wrlwPtXpdfpoj9OoK8yrPlqfwc3LCq6/XJ89WNdpFW7fLnb140o4h0+y9FG0k5lfQsfb8bwdPhizD9gso5p8YPVF4O5N8zvhi5e9dlrgdPWP1tQyX0Dd1aCW4epW6xGudw7ms30udrOQrYPuB9GI2OP81GdrzzDqSZZ3s8E1vdzn8i6udGrZ9SuyPGuTTj/YOzdb2TlgvIqbKa4zOp4j8TmNdDwzVH0LOp5gxnsHyx9F2Q+S6RWdPc5DjXGmzdAZqI5nFtbMHrS/5p9aZo+34XqWvYidN2DuT/IiDl2P0HkerZ1fcxnu+i7P9Vmy/ofOPHXN6zzNrjsfRKN1FVOPnJ534vqT9djpldZZ6/8L15knW2daxHoeaVl+Bfa1OpNMP8yMrBwO1Z032zeC+xjfAXvJONNIx3OW2Wv8x0dRtnmXXrGy5w3EQK4MpntAnPYUo+u/5V7/yEubcawrjXvpYez0bK1xwJyjyimjOsT1SV/YrqfcQ133rax3fBDFOuLQMnRfxSjfxfWIvXOzErHmeFQexlqbecVpJKtFTXHanXT7fiWc09BGa8bMXcVVPVupm81lpTsy/eBH0dNUN7qDq+sf5kU9YsavvpF3Ft5HVZ85xnfgHgJqjDONulszztY8ZnLEaTuY6W82m1HDzUx3TXZ+EAUul90Dvcx3cD3raofROSf0UF85susQerme1a7E9ZDaSn8rVmaHs+l052HcYWXPVdw9DzO86qPoePkva0T3AewOJx8MPXN9hlFNp2XMeBX23b2cM22E7tNzddCrZ1ezyqmuZPucdhbXl1HfAzd7SlYn8ztG+/6aPwyrup6pZ/EM7B9hnj2mpmdqM3pnrRoPegKnPYWbozO97DK6bvceur5Zrqq7whvmJOPPnz/v+yg6Xv5LO0PnIXF6pf2Vf7pdPSoi3/E6VvcF+kKu4Eva7dGXN/1ch8fFTtf9miPcHxrpeGbRPmhfVcvmIjT+n76fWbv/M3x3XUemH0lOfyaXPwv75aCHfuYVnanVtdbhOmJqFc7rtJ1EH10PnebQOdh1aF1CnXGQ6U9wVR+vqruTf30UZTe9S+8y2s8840w7Cv0qdNg7g595qDM+w+jBDpif/dlGdHoTHneOgzFzmu+sGWdrF/MIsnUG93c405O/8sInmstmQNfufzcUa/dxFLnQOvpf+Wij7taVprBel6q/Vcx54exQW9HdUcF6qlNTMr3L6Pce+arHjDPtCngdxm/G9Vmhxpiay8+Q7d+lB//6KDqKTZl+FdkDx9iReTKd3Dm87qXLOMj0M3ReLMTdMxnlSdUb5s7OhjvIP//89+PhrqV7srN6ifNm93GWUR9cL50WRE49/NCZWTtt9KFU6Up4qp/nwM+0A/a1yukscF15Z/VqPTrcPr0OqfKMV9HeOlTP+st4F64uNXc/b6bTt8yT6W/EfhS9iSd/mXcPrLuee/C5PnuQ0DTvfEGV24m+lAn18FaH+qo9WY56FlN3OUVj5jKti/Yq67/yN/m3L47so+audfazufuOvPu5GJ+BveQ8ZD6FOVejs47YrRlz7WLuD/Tau9jZE8JZWD20Vqz1rDjtLezs23FBvau59KPo034Zd5M9GHyYMt9ddK7f8eyCc8WXMPMZ7kWfrRkz1/EFbq0aY+a6sCfZPLkXOsk8jN3HymjNeLTm9clMvoI+xjNU/cxy2QzFWo9Z3R3ZdTp0vJnHaUr39+5mtLv3Kngf2f1k+pf/ZjQjO4hrbPkoqm64yv023Ms6exgy/cBDvwv3IuE1wuN+jsozC2eGcWiZPsqrR3We9XB+XqPyOZ3rrB7XO6j6k+WyvrPfus4+bHTttGydXUe1aga533mOpLaDfWE8gj3ODvqpdfTReoTzdmPqZJSvYE8d9FS9P0NVM8uFrufR+jdRzUaWO6OnH0WdzUqmH4PcU+y+Jz50ca4GXnGa4vJOWyG794pRfgd/zEuYhCfzaV49sVadPu51R1WDuvMG2bqi6zsafY257MxpHN2PFY2zj53O2v0MvF9eO2Au863APrj+RuzmIItHOWqudrbXrUeH87KOY5TvUvW/y6x/BtZmHLifg7nO2mlnYJ/fQHU/WW5WJ+lH0W66NzTC1aHGONNWcIOoZ2pcV+h+h6sZe84cq5zZO4L96ryAR5546LsH92Qx62vMPSPN7VXN6WTUl04+PKMZ0Vz2MXMMfIwDd113X6NZVn3kY47xLNoz1Rizz1zrrHAOuM487mCNHbD+TrJ+UNfZoKbxzoO1K0b5Ebo/1mdrvo0r5qfDbR9Fx0U/ZKdmxzODG8grGD1szJ0he7Cyh97R8ZyBfdRYX/BxdusM1nJ7qLs9GlPn2Wkut0KnFyNPZ9b+Jv/fdOjazRY/mqjxejqD2ZrEvYUn8x3Nn3XEbO9iNty+ztwwpq71na77Mn926B637rKyJ3B9cn2kj/EuqutR2wXrjq5V5VY407+Kq+p2uPWj6Hj4h93JjuGKAR7Vyh623Wh9d63qPpw2YnUW9MXMlzRxOb7gXS3mO0dWQ9EctSw+Q9aXqpeKzqfOK/dpnH0YcZ/zHebe3D1oXnHXyZjxnoU9reaAc5XNFPM86HHXY5zlqFXrLjNe4nrmNEXzI+9ZqhlVur6nYI8YO43xLGf3n+Wf6gay3KxOur7fSvWAXPESZ73qQe1qT+BezPHC1oMwr3V4qJ6t6Y8c16q5OMj0LqP+xExVs+V0ahrrmv9WiLkzH0Q8B6Ofpco5Mn0WzoXqeiacK/VXa6d11y7ewRU1FfZK4yp3BjejDpfjDDvPb6E7F5lvVj9MbvhvirhhF1fVneWJ+6genNkHI2qdOZ4m64G+qKll6B7nVZ3rzOPObs2Y60rjOfNxPUO319lccG7U586jjx3CHOtzj8ZxPedTspzbx/tZxfXQ6e6c9X2Uc1ql8yDc0107bSdVjzgfGdH7M0eH8FX+KvfTuWI+Vhh+FF3J07+E7PpXDOZqTfcgrdYa4eq662su07voy9O9kHe9XFmb1x2tVcsOt8/VCJymOaeTjkfRnsX6TB9Zj+h/Pqv+UxprsC5ra8x/E6XotViPOvedxfVfc+rp5tWnOXfQw1h14nKuzmidaTtxvXJakPV8B1pX5y0jy1Wz2WF131NcMRerPPpRdEz+MuhlnGm7WBk095A49AHIzorTzjB7rZHm8l06PYwXbfbi5THayxx9rEOPam7Nw+HqKNXeEZ1+xAzOHLqPa60b8AOGe52m+/8m/8NuXjOj4+P1Orh5cMzkq3O2DkJn3l2bPpf7VKr+VbkVXD0328yNcHvfBmeEccWM9w62fxSt/ICdPbs8ytVD5obZXXNVY7wK67j77jL6Wbu4XjrtMH8MKpiPWF/+uqZP105za/ozjXsDpx2FnrHSm7/mI0fRWXFr+o7kQ8bt7dbRXPxvlOKIGq7WlbjeUHNzwjnIztnaabFWjztcLqvl1m/m6t6zPmNqzDMmo/yddHve8XU8ZGXPDMsfRdWNVbkM9yDewe5hm6m34uUevvRXDkelZ7mr2DETOlvZrI00rtVDf3UN9ammcD89jDONaO9m+siZ0TnQc7bWM7VsTdy/ITo2zeSOGjNUc8A+0qt5t2ZNFzuynNbo1DnMz/AW2OPo+9mDNfU8ous7HpjTjLP9Hc1PxsqeWZY/io4Lb5AP4ZnrnNkbzA5hx88Hh+cRXV+XTr2OZyd8CTutWnN+dK3Qo3q2ro5sT0V2D45RPsj6pS/WzNOFdVjb5Zlj7M7ugyj20J+hfh7qcesO7Iv2061HZPMTNbLD7dF9uqbH+UawFq9RrXfi+qlQZ3yWbr3wdf1n2HGNM73S2eCc7GZX3VMfRU+y6xfQZTRczDPu4PY4LahyM2idlZorezKqvmqus1ZtpHfWWdxZs6bzufgO/pqPBD1GhEfP2ZpnXodnrt1/Mqtg/S6z/oD96/aas8GZyfxuzZiwNveoz5HpR5Lr3M8MnX5W+c7+Wap6LldpLucY/RxV7gpm+7iTndf+yI+inb+AHXSHuesL6Hf74sE4c2gtR6bvovuSniWr07mW89A7WtOf5eghbk8X9niE+mPN/52OHrqP52zNs9MC92+IDvNzuf/9kOapZ0d4z9Dpk+tppmmt7todjupaqrEOY5LpitvPuGLUp1H+mJiL0aH1qvOIrn+Uf4qZ/r2Vyz6KfsIv5wrcMDstgw/h1Vz1kLoXYgf3Yq7WyhnPyppa4HJuTS3DXaNDzJIemuvADxVFa3Kdobnsg+hIfPxZ3M81YsY7YraPGazjDvW5NesQ+p2nYnQ/I2a8M2g/d/a2YuY61TOxMr8/mZX5WJmryz6KfgNnhzXbn+l3ctc9VANbvVw1ztaqzXiydcQza9UyT7beyZX9rD6MVNMXvHvZa1x9EFW5N8Peur5z9pjn2mluHXUZE+51HtLxHBO+3Yzm807c9Z02y2yNWf9v4uM+irIHK9OvZjRc+kfAQZ3++ONB3xW4az8NX9IzdF7q2R8Ktze7l846YmoZVZ1VdJbO9Db2Zx8n1R8hvbauu/sz338l//mM+3cS/ZztTzZnXOs5W7s9bk2fW2fo/spf6TPXm6XTY3oYX4HOuTvPku2brdv1jcj6mOmfQuujqPohR7krHoIr6Q5M5eOQVt5V/m5+4c/WuupnW52Xzos382RrjXccrF+tqVX3spOYKz3+C/9v/oRPz4f5SAk4I6xT7evU17WDPw+PHezoA/vOuLOujmxPta7gnhkqf/f6h5mtDjp7O3H1nJZR3VeV+8nELFTzUOVWaH0U7WDHjWc1Mn0Hq0NY7XMvY41HD0Cmr+Ku7eB9VV5HZ8AD9eget6ZXoafyc6+S+WbXTsvWVxNzqAfhbDifxtnHCj9cXJ2zH0T8WdyRoTmuuS/rUcyWHqrHWv1urWSemXVWO+D9uTVx14k1a2TQewb26G7i+jxXOI/TPpEdPd1RY4XbPooOPASzP3Dmz/RPwj1I3YdDX/irR0XkR74zdGdC8yPvYfydPUE2q7vWs5zZO8PMbBzwB9lHCz+MjmS+ztQb4e43u4cduP7rTHXmy2m71tTcfcVajw7cfwezPdR5OHOwpp5jrV6er+TKa1R9rXKE8zWzdze3fhQR/hKyXwjjT+XMcGYP1S74AHdzV+JmIUO9o32cs2rNPSvriKm7te7Zic6PzhFzI7IakQuyjxf+mx09Uz8ae7M4yPQj+VnUz3VVawXXf67Vq+uZQ/c7bRVeZ1RvlD+L6yGpcmfZWbv6WZymjPK7qfrKHOelMzd38+hHUcVbf2GrA+eGfLVWcHZ/4O7p7wV/BJSsv4yJ7nE1uN/5Rzm3Hl2L/mpvVYd5rlfI+tjpcXjUm2l6PooPGvdxE+fMV/1bobiP/zL/f57N0PF3PGTUQ6cdyfxorfBka16Tmlurn2sl0w9TT4lclt8F58rNA+MznK3l9jttxMqeq7mr57vY8lH0KT/sLtzgxUPnchUjfyd/9tBab4APkJsvp5GshtbvrgnrdVn1VvdScaansTebF82pn/kg+8gJT5bnBxGvmV03jtWPpDO4PrnZIpFz/Z5da8yaK2s9Al27uIurfRU6BzonZ44Z6I+Y+ohZ/xnu6Mtb2PJRdDz0S6uuedcDtgoH2j0Y1Hj+6WgPR/2c8ZJsL+tEXB3cxxpK10d4vYyVOYmXfOdDIj5W1Of2uJk+ig8f98GksbsWfYR7Mv8oX8G+jHrU9arOtdvPWlkuW3cZ+Uc1q9zVrPS3g5vHbJ6c5nA+p11N1a8qdxWj+Vpl20fR3Vzxy7iCmeHteDueT2N1uN0ealo7W6uX6/Ax5prxbE5hfoWZOeGHQLZXPfHhMvowcrVUi//c5XKEdbmOD7kM5hjfgestZ0HnJNOZo4+H25fBeg56quswrzjt09G5ymYs0ytW9tzJT+nlR34UfcovX1/aM8z6d+D+yNxNp698wWYv28O8uOPs1g7mXD0XZ7nOtVkn850lesw+Mw50NvifszofRrrWs0KN+0ZrjfVQqDOvuP2hz+D65zRF58Tp1drBfLZXr8u1HhlV7kjyTvvNuPly2jGhZ7O8m5/Qy4/8KHqKbKgynfBFzLPTeI71ziFnbZLls/UZOi/eEdl+vuR1zVj3OG2U0zx1hfszMp/TRmS9oh4zlv2nNI1XPowUXmNlHajOnMI6I1/FqA+uf5wP5gPmuHYHcwrz1Tojq30lrtfMrax3wRni/bo5cx49d5n1fzpXzt0/R/MCuzxn6Vyj4xnRGbLMk+mHyTF2VB590FaPCs3PrnfReflmL2mnEe5l7LQq5/IK49CcvhvXHzcHEVMPnK57uuvs2vSN1o6u5wrcTGRwVtyMMXb1uXb7tXa25p4Oo+udhX10vdVe6jr7z7NRY9fhyHSl4wmc12lX0elnx3MHnftwnv/9N0UuSTqen0Rn2DJPpmd0/OqpHsRd8Hoz6yvg/MVLl7rm3Fo9Stc3Itub3asSnsyX6R26/Qlf5td/q8P+z/4bI1eD+WrNfVlOzyTTu8z0pOt1c+hiasxrHVeTfoV79XC+Lq7GKuydxp31TliXMRnlj6bny7/pzFfm+Y//fJaZvvTJhjjTD/PS5jnTIt51aM23U82qe+nri5jrDL68I87qOL9bq+b0u2DfQ9Oz6n/N/55oVGO0VjJPttZ9zI08zAeZnuH65zTF9d1pAXOdWcvi7BqH5OhnvUp3dDx3of3VWdhxZNfRODtzrXGm/0Y6MzfKHwPP9P+mqCp2NPJn6NSuPFXubkaD7fLZQ7ILfbh55rrLyp6K7KGYeUkfyYufa42dpjGp8tw/y8pe9oGxasx19E4uWzPurAN+pB3iY87tD7J72YGbIc1lcJ/zcs5cnO3TutmaOE2prvkWdvdXqWpnOun6nqLT247nrUx/FHW44hfSqdnx3I17SBhX0Mv4SviHosuMt2L0cq1ewKprnl6Xy2oeZr/D1Yy125/9HJk+y6gfOqP0Rm6kc81ztXZatQ70oyf7P8fn/4D7MD9jkOlXofOmWtVzp1NjHOj1qmsrzFPr4mpfTafnV8B5m2F13910etnxvJFLPoq+rOMe5KcfFN7TmYf+DrKXf/VSj7zzVzH99Li1i98Ae+r6zP5Xa553rAN+EB3m3gKto5rzZsx4V+DsEM4S/Vkcmp6pOejreCvPXbBH7PnduDlz87iDnfXe0k+lcz87PP/6KBptODZ6unRqdTw74QAydpqLuw8I84zPcqbemb270Bdz9kCrzrWDtRirxpiag7474AtadcZOy87Zmme3ZpytA/dBFISfexir5nKrjPo503POEmPVGFNzXtXopy/o+p6k28+urwvrVfNFrRM77Uk6ve943sa/PoqO5g+yy/OT6Axp5akeIqXr68J67gHM6PrOwBf37MuYXldHPfRn0MeYGn8OR8dD1Fv1jnrmrTTOCGO35plrp3EdZB9E2cdRR8ti5kas9kzp9p95xtSYdzl33a4v6Hh20+lraDpbej5LVm8Uq+ZyGV1v13clO+agU6Pj6WA/inYSD8eZ49Nwg0iNccAHpOM7e7i6I+0wOuO70BnhOuJsjuihP5vDUUyN+SrncPegVLkR2rdYU9NZ4dw4nWueZ9ZB9hGk//lMcTVC7zLjnSWbLYfzUGMcGvWINUePaq5GhXpn9p3F9crNgIvPHq4u6foCra1aFStV7m5ihlaPER1Pl/SjqHORjucOOvfhPN1f+MzgB13f0fBm12Z8hqxWppOu7yzaL+1fttY99DgqT+RY28XUHLxW5nN0vTN9cXPG/fosaF7P2Zpn1slyR/IRxPXfxv/FGe9D9Sy3gpuDEfSyBmPVGHdqEV6Ha42z4066feIMqL4L1mKcsXPmgp21Rj19ou+rdO71z58/+UfRpzD6IY+mZwYOHeMuM/uyhyce+LOHg3rlDTQ/8t5JzIA+GDoXnBHnIcwxphbXdj6H87JeRfX7Zy5iPVeaUvm45lkP+oPsI4gfQIHTtSbrM6bm8jOM+nSY+RztYZ6x0/Qa1F3O+d4CZ6kL/ZzB1YM19TyCPsYznNn724n5/viPok9EB9cNcfawBZl+luphdlrFrN/Bl7VbR5yt3Ys8q0Vv5mHdLKaWQR+9jHdRzdiBvPPx39LwXK31P3kxp2QfQfzw+S/8n+RX98b1bly/dCZcj0c4P+u42k7LcozpXYX3s0I1I8T1W+nUWCW7dqYT5xvt+f/tneuS5DauhDn7/s98en84cBbOycSFt1JV1xehEJEAQfYAkuger3cHs3V5ApW9ZzHeHx6KskSjGHOKytoYox76XWADV5tbPaheZ/7d4FpsTbXX2/haVscMjPFxZke6h9moVZiZExHVi9Wajb1mh47o8IG9i3vANdCvDkHsQMTwByXMjRrznwZ7A2uOPYa2ikMNdfQzMM6Pq9duohphnVks006A66g9oZ2h8pymUstKzLsQHorGh/2wr4K9fGcbe3W+guVjmifzG9W4Cp1+9LE2Zhra2Utd+ZRu+A+Gv7zvBNh/rB4Yg3gNDyGzByOzUR/BIShbG3NF65vmx+ivslI71geYL7MNpY+kv73t41DrwNY5SVQ75mPaCiwf0wZ5TjKyuMz/qVT6qxLjSQ9FFbqL/haqjariMl35u2T5fsjHxoM62kpj2Iu080KuxvqXPXvxe3APTPN7VbrXnoSqB+pYd3VAwcMJ3tmY2SPIGx2IbIy5DFwTNQbzM+0WrKdYXzHd7KoPY17Fyp93NLfaA1WyfEr3VGIYs/MUrPZMezWVPc3E/AcFxq6YnWTrsYd/Nz/wst+ByoO6X3f1QpimsNjOnBPgCx3HrBcwxms+BuejHelMq6LWnM1XxdfU15WN2cHE+1kuzOs1la9zIMI8bN/RXjyR7zSs95jubaWbzfLgOIvB6xRYK0XkZ7VHrAdWLsuDeTu64f0qZgTzT5HV+nQ/3GLLb4pu89Q/+GqTsjimIRiD9g6yB/YJVF/KzIea2T4XxniUT+kG7jm7/LxT2EuXvYTV2GwPO6CMZn5/H0FOtNl63s7Gu7F6qXrOsprD78vuKieLUbGeKCZa7ya3+oD1dKQbSkeqcV/6vOWh6B1hTYwa2qbdeJCreatxu6m+VKOXOHvZox9j8GLxVb2DnzuTh9UpeiEzzZP1oTqoDJiL4/8j/4etUa7IZ6h94p4V1TisC9oe7KPqxXIwLdLNh3cfi2OWU8Fi0WaweR2qdUJmeqKDyqn0L89i66FopcGr3FjjJuxBwY/EadRa0T6UPgO+vCM6MT4vzvMa8ymd5avoTwHrpj4Q0dhszBUdWHAezjWiHJGP7UmtgeDe/H2GHfX2PYg9VOk33IPPp3T0GRij4kaS5wSsTkyL6MYzVnJEzwNSjft0TvbY1kPRl3+42bj2QK1cXaI5ke8UlZe1gTGVjwXzG1Xd7xGvDt34CF+r6ths1T/ZwQXne6K5kQ9zqXXU2M9TKF+lhpUYw8fiHLSNTMe7jZnObKahHYGxfu1TYD8gqFk/zF5fnsdsj20/FM1u5NOwB6X7wKh4pe9EraF0pBpXxV6e2FPRC52BsTjH6+jzfhurfXl/B/x5ogthmkfVxOvVse9pNjaqBxg/juZEPszBcqux2Rl+bhVWF6wluxDUVKzKY2PUmO5t1Bm41hPAGs3UboVKb+3Sn8rTeqJL+VDU+UE7sR1O5T0NezDV2NPVd3E6/y5YP9iLGl/uGItxqGMMi2UxTGPzGJ24St7ZOka9+QMHDNSM6kHmR/wvziyG+QzM43Ucsz2yOBxXYbWo1Ejh56k8UY8pH9pdHW2vKf3VrNZ2B2rdrv7budFT5UPREA/+LTprd2JvU2n2XTEznMr7BKIHCnW0Df8BwBimeV1dN2CHAtQrYySKwQNNltMfiPzd+wzM5XU29mAMi1N6xmo9fU9gf2R9U/V729+r+quZrY2xMjfiVN4nUe2BatwTaR2KvtxDvfDRvsnt9TzZy97fqzraSle2R/mU7vE/l/85ve5ju2DdsI9MY/hYnIc9iv4Bh5konh16DObDHGxf1XFXU2ANGVhbvBhdnYGxfq8d3dudPau8RpTrFJ3admG5sU8/ndv13MWxQ9HOP5CduZ5E9QGpxs3CHtbMVloV9VJlWgU1R72McX1/97EYx3QVz/SIyF+ZXyE6GGQ+037g/9AV7zgXD0ZRLvuNEptrMWot/M2U4WNwv4jSZ8BeyOqH/YI+liPKH+n+bmOMi3RFFK/0HWAvmBbZSlsFc6LNqMS8IztrvjNXxLFD0SAP7Oz1yWQv6Qx7GaxcEZl/FGO6zNY96ptIxzvGRT3Z0ZmW0YntwmpnGusPr0V3nBcdbpg9yMEG18b4QeawcaS9Cqux7w3fK6qHsDcqPgR1G2Ms2gbO77Iydwe+t1YuhtIHeW4Yke9dwJ6cvW5x9FD05X9UmrsSMxpxXTAv2kxD+wbqYcEHx+yq7vG5o7gVfH68MioxGb522Qua6T/uY2BjtD3qYJTF4t6q8cxmc5l2mm69R1BzpQ/SY0z3VHT0GairXKdhtUQN7V1U80Zx2O+of8lZ6bnvoWgz+AJmKB2p5Ppk7AWcNXg1htnqbmNm46V8HqaZrpiZUwF7Sdl2ZwcOf8c+9b6Zvw7DvwZjMR7/12+ebJ7S2LgKqxn2BfoN06M4lQd15vO2v6PubdQMtg4j8++A1Ylpp8E1Z3pJxSn9y9+s9lz7ULS64Gmevr+MJzT/7T1UXq4dfD6fF9fBGLzbGOehjr7IjzqbO4I+Rj3K4fGHA3/3+EMC3rNxZJtm4EHH+zHOg3MGiRni51B+1P29Cvvzz+prmB/v3h/l8X6MYTpbR8WgjXGMSswrYT13m9V1X/0zPLm+O2gfip7Muxar2tj4MFTnZWQfkVtE9WMveAbzo8Y+AOo+g5ob6co3gp8d7QxWW9ZHmZYdRNC2Mf7GiV0GroH+kcTgXlDLfGhnsPpE+PjqvEpcFMPWw32zGLTRx4hibE1c+xasN17Fq9ef5RV1u8VHHYqezswDgHOYvXoxvK5iXknloYxe5Myn7jZWL/LIZ6COtmk+R5RvN6wXlGZEhxJmj8KcMRHzA/8LNozFPSmNEflmUDVldY/iOjGm+buP87YH/Qpc60lU6mw9s3phPm93mZnzZQ/fQ9EGOg1ssTgneojQfhXRHneSvVz9C56NPfjSRxvx+eyOY7xwPrvQj2NE6SPxMVitmGagD1/ypg3yV2E2RtuDh57h4vCww3JhDBsPcnBCmDaEzrQqrA+8L7PZFcUgpnkfxqoxs01juqIbP8tKnU6A+4n6Fe0vfXb02PdQ9CCiB+bV3Nobe4FnsNjqSz5aj2kRWbz6MCjdqO6XwWplGqsp07xdjcM7xrADC8up7kpDG2PYXjAGUfoQdcjqaaha2nzUPV0f9hDaOM5yRJrBfEw7RVS3G+D6rC+/PI+pQ9HNxq7yxD3t5MYDlX0ckNN7wpd3BT+HveyZjfGYQ+3D+5h/iPloox75dqPqxmqr7ja2i8X8iP8lGv6mCeewu41xPfQjqOF8FtMlq7OqMdMifST9p3zMZmOzTfN3jIvoxt9itc6KE3mrOatxu3hiXXcwdSgaD272J5C9nA18kSOR7xS4JtpVdn1kxsILeZAHN/oIoIZ+W1/tA/1Rrt2wNRFVi6hfsY7Ys2ys7hjHwDl4xzHa/t8tsjuLR+1VYL3QNlRfebIY1FR/RnEZau0TRD3h8fWO4k6SrVv9Wb5odvXdf8bGZK/iFfvPGjdqcrQ9kc9TjavC8jEtAl883fkRqsZKj2AfgJk8nmi+WsM+IKhnvhVm+5LB6ow5MAbvHvytEd6ZhnelMfs2WMuZvjAin+FjKvEjiVP52BymreJ7w2tsHKH6Yxdsnx70dX6GzH+bE3WeYec+/v83RbNJZ+ft4tXr3wQfCLQz/MNqY9QYT3to8aNR7YHqixxze9sT+QzlU/otojqxnqjebZzl+CH/kUYP5mCaujMqMZ5qnEfVNOuRiGiu778sxtuKKA5t00zHdXaD9UDbw3oE8X51dZiZ8ymcrHuF2fVVz07/9ZlHJf80oqaPfCPxo8/bkc/s6qVyMLKYzH8b9oL2to9DlB9jfV70GSrG74XNZXN2MlMvm4N3G6OO/uhu2L9X5PG5vcZsf1cazq0yOy8iqrPSR9BXnqpPjX0sG5uN2m26dcG+qOL7J7sivB9j0fZkuWd9n8SJXtxyKPryDzsbcfZBOomti+uj/UrUQ8Je9EzDcQc1j61nNmqm3ybqN4T5WW9g3E/hI8L8mBvvnsiXUdnfKbD/VG8Ys74M3Me78YraeXB9b6MP6fizWE8n9p041Z//OhStLrI6/4msNFQ0N/LdJPoQeJ35R6A/hawn1UfAxtEHyvuYfxTWH4WYbI0uWDNVY9TV3WC69RBq+Nsh0zFn9d8zwnkG6jgf7zep1DXyV3wYw8YYg0S+J6Bqhz2CWN8w39PI9pj5P43Vnozmb/9NUfaA7aS7Tjc+42Yj4lpoz6ByKD1iZs5t2AcBxxWiHjefiol8iIrBNVRcBayb2dU7Hm7QrzSz8fJkudk8jPE6+vC+k5na+HicF9UbfehH1DpqHGmvRNVN6QP6APVVMAfaSOavsiuP52m19pze21+HotMLfhqdhsRYb6NPaR702wNfvRimZ/4ngC939VKvjKtEczJfx+9/Lg/aTyA7vNgY+87b2FeVnGhjDN534uulxj7W3xmV2nsinyKaw9b3Y5zLYnB8A1VjtCv4Hq1efm4E+tF+IrdrWWHHnrIcfx2KRmFShR05PhF8GDoPFoM9oLOoHF39BupF/Kf5T8HYp8xnOTHW6PjU2GtMfxVWY3U3Kn/NZTbTPJUDEctjKP2VVHsJiWIrPr9uhU5/RmN/x/Epsrpn/hmiPqzCcqD9dG7Ud8calRz0UDSKkzP8g/I0Tu0LmxntDIyvfgh2otZQ+mnYi7dK9cWs4tQc2wv6vY4+w/uiuJNgLc3GO9PwPhNnF2rZgYjN9TrTUH8S1V6IeiryjaCfVfxoxCkqP9MNsPbYT6fA3la+ChiP9i5eWaeMHXur5pCHotFI8g7s+llmGnJmjmd1fgX1slC6J/LNgvVafUlXqK6hfEofD/g4GJWXNd4jMAbn4p3FRTHe73Vlq1zMx2KQSswsWb8xbQT6aPjUOKIap7D5q3m6YA079Z+B9WOHmTmelfm3a1Nhx546OcJD0WgmU/xJ/knmNLvXxaZDOwPj0c7oxkf8kI+OofSRvFiYNkO3ZzovehWrxmajFukj8XkqMTNkL+io/gjWHO9R3E/wWyAcm13V/B1BP95x/AR8z6j+MT3yodYZI5Gvw648HlW/So1ZT61QWdODMWhnYDzaSOZHTtSrQtTfXbo50kPRmEgasTPXp8IeVLQ7WD52VcA4tI1OzoiVHvEPkr+rsZ+nxn5Opme+CNwbG8+wWhM/38bqruIM9u8eYf6KZrq/25jFKtg+1Fyl3yDqKeVj+srY25j3Ft0aoW2g7vuGXbvo5urGfwI7+2omV+lQNCaTK3bk2pHjFKyRmebJ/Eg1vho3mrEnmKmpf0kzlG6oD0A0L/Ip2MfEjz1qHyz2BNYH/oOAveFt9fHI5lZ0ppmNa6rfRPk4zINk/ipWW1Xr7tij9NHwdcce1PFnq9CN93RrpGqP9ipZvq4fbaWdZqVWXXauNZurfCgaC4swVnJ15nZidzLTvNkc/3I3G314VWCxaDOiGPNFMYyZelXnqBd+Z76KNR/6vabWxDk7iP7cI18G9hz6lMZ0vGMc07yOGht7KjEnwFr7fuiMPUofwod7mB0jkW83Uc0iX0R1nu9F1peYB20E53+520sRrUPReNDGT4LNijajG9ONV3QfLnyocS5qzF+lE9uFvfh3oD4Gai22D9NQvwX7c2cawuru70rDeagp3Wz7KzXUKzlMz8Ye05XfyPw3UT2ldO/bRZaru141tlqHal0R31esvyJYPNqoMT9yM2YX3fp7ZucxVvYxZg5FY+MPsCuPYmd+bC60lYZUYk7RXRvjowcb7VNUaooPxeoY1zRN6YjXK+NdZDVh9WRzmBahcindo2JmdRxXmZkzWz8/D8cs54y+OmZ5FZ3YKqweTDPQp/pEUY1D2DymId2YbvwqJ2p6gh37nDoUvQM7/nBWyZqS+VFDW2mMn8l/+mF0H8iTqNqizl7uM2PPil4Z3yKqoeoX07xf9Zf3V/TIV9WjfeAYNeY7idXc39nYxyAreneMGvMhlZhVZupWnWP9xPqqSmXezZgVbtTzCbz8UIQvgN9GpZFZDGr4AKO/gp8zM19RyVXpgZkY9ZKfGbPcHb2CzVHzV3IrVH1Mr/QF6zk2X+lMm9FNY/oI4m+AtevUEOd2de9fGXsbdU8l5gSsp5iNvgrWV6q/mIZUYn4bt3sk4+WHoi9/wx6cSGO+J7Ha9NH8XS9f9RH4Qz44pjMyXfk9GIP2TrCHVC+hHn0YvK7yer0S09GZxvZ0AuyXTt0RNWdFr4y7VOZGMZHvyWQ9hRraX57J2x2K3vUB6lJ9gKpxJ7mxh6juzDfzwp+ZU8FyYU60GZWYKjN1wjloe36Cg5K/e91f6PP3qu7HeEeUvpuZGqqeMbr6SHyzVHJGMZHvNDvqz3r3S51X1l/xmEPRE/9wToEPEdqG0neTPdizPiOK+SN+E2PM+gb4/UeGjStYPM5RuvkQ3EMFlb9LVGvUle3vLF+k4X9LyONjVA5/Rx3HHtTRNpjONCOqSeRjqD5C22CxSkc7wvcmGzMin1GJuU1U252odZjOtHeiWudq3G0ecygahT+kzG9U496NEw8Ly8m0FVQ9lD4S30j8kc/wMdHYX4xIVz5GN/4kP8GBRMH8LM8g/1Xr4eaz+BH4I72ieVQupe2gWvcoLtLVPK/Z2MeyOSPQjcw/RAzTqkS1QR/aSltlNufsvHdjpd6nedShaAR/WEr/bZx+aLL8kT/yzdCteTd+FbWe0ofwMe0pqJr6AwSLUbrh56OGsEPUEDnQh2NG5p8hqyn60VaoOKUzfGxnnofNYxpSiVmF9YqH+Zg2SzVXNe4dieoc+Z7A4w5F4w3+0G7CHhymZdiLAq8nEtU/8g3ir34A/hT/qsB8kZ8RzfFUYoxsLxGdHrAYvHtUvp/iX4cxn81lROt5Df2GX9vfb+D7LEL5Iz3rB+9TYwR92f6VvpMT9fI9hX1Uhc2paopO7Gk6tWWxTHsajzwUjcYD+3SqDR3FMZ/S1DXLytwuWZ2jlzHTGFlfKf+f4GNjvsjPbNQ9Ub7ddGpssVFfMd3PQ51p/u51Fm8+dYA6TbVWnRh/Z/MqPYc+b6PPU43LiOZGvt2onqngew4vpKr9Jnb10k0eeygab/SHmFF9MKI45vMa83fBHGjfoFtzFR89jJGPUYlRqLlKH4nvFPiyx7ECe7ASOxOT2Qz8GTB3NcdOKrWtxHiiePNFPa98GKdgcUx7Ejvq6nOwfEz7jfwhB/Qn8+hD0XiDh2sW9cAofQiff9mfBtdB+xTZQ6V80Qs+85mGPsNimD/yIVlMNU9Et07sZY93pZld+SszRPm8rnKyfTEbfbeIaljplywm87Ex2ujzqPxM82T+XWAPmHYatq7pCuVTOlKN+zLH4w9FX/7NqQciy5v5O3RelFGs8nX1IXyo2YcBdSPzKbK8g3y8otjTsEMG6w/lYzrTIt001NGeZVeerE6RX/mifsl8yo58nq4+gn6taqfYVWPkVN4vd/i4Q9HNh4rReSBUrNKNzN9ld74KJ+pUyTnzIagQzY3W7NKdHx0quvgcbMzWUD6vV2IimJ9pp7GPP9YIbaUxqnGMqO/Q7qLmK73CytxZdvdJli/zf3k9H3UoesVDtYp6SJRuZH7EPj7sYihdYfFRzohu7VQ86mijhn5m24VEvuH83vZ3xqxvBl8z1Kpj1Fj92Tpm+wt9/u71H/HXc8yeGeN9lqxezM96Cm0Pi/cwH9qeSjzahtKNzH8aVU/rG3VVyWIjf+T7cpePOhQ9mZmmz+ZkfqMa9yTUC7Srj8Q3iB9thH04DPMpv5HFRL4VsBfM9h+A7tjwtvKhzlCxSjeN7W1mvANWP9OYbwS6UemZyD/IGn4O+pRmKJ/Sd+DrU6lVJSajkiOLifzK5/vySVT67J35HooeTvZQ2IMTXV1wDtpK24168LxeiWE2gn60DaWPxNdhVx7EarazdpjL9xzzMd2j+rYy9x1gtWXaCPQMNU/pCoxH21jV0c7wPVDph0pMhu9LdkVk/i/P4nsoegDZQ5P5d1JZqxKzgnpJrureRh/TzP6T/PYn8hnK79dAmLaL2Rr6eTZmuZTPPiL211+oY7z5/B19OE+NX0lWS+Vnuu9H5h+kZ1mc96E/s41VHe1VnlJvI9tP5v9yn485FO1+uG6TPRyZ/12p1E3FVHQVM4QPNbSNP8lHaZAYtH2cvyNKv4k6aPgDC/ao9yn8PBanfGqeGj8VVVuvs57xqL4yUMeeRFDDvTC6+qtgfXqKbJ3Mz5iZc4qn1XYXH3Mo+gSyhs/870r0gjYinzETg7bSPJl/FGMq7MrThR061IFDjb2NOmqRH33ZvEF09vPcQNWvq4/EZ1RiInA+2kZXR6px70rWY5m/y+58v5npQ9GnN/UKJxv0Zm7/IbnFal9V5mMM2pHGdE8lxqjG3YLVG22Fj8M5/jCCa6CNGvOxcQSbU527A1Xnrj6K/cViTPM6i4lso6uPxHeDm/W+udZ4wXqfzvShaDyg0Y1b+8DmYy90g+lMQ3bFnOL22qq2qKPtwY+BaZFtmr8U1ZjINpR+k6zGys8OH1XbqOZAn6H0J6FqrHRG1m8j6EulRTbT0FZU44xu/Ayn+mRX3l15TsN66RXs3MfSoWhcauCIU+ufaspK3l0xHVbyrcztgvVG21C6B2PQrlCZE8VEvlvM1E/NiQ4uHbvjM2b+D2FVrpuoHlC6pxLjYfGooa00I/K9glfVtLpuNQ7pzGOxTPvyN8uHovHCh+L0uqyJfshvh9A2lF6hMrcSU4HlYRqjGrcTrDvaBupoMw3tiEpsJabCrjyI1S+ro/W9j1NzohiWB3XmY2OvsRgW66nGzdKpmYr9U/gn4EqMAuehrTSD+Zh2G1ZT1NBeoZqrGoeoeUxf0d6R3f225VA0DmzsNk9skMqefuBjMnPd5Eaf+DVwPfYByWzD5rIcDIypzhtk7iuw3oh6RPm8HsVkPVjJw8C9d+aeYqX+aDOi3mR6ZjMNbUVlvSfge3DlqlCJ6+RjrMxdIeq9U5xab9uhaFxu+hNrrTTUytwvf1OpL8agbSjdgzGZnRHFM5/XmP9VVPpaxXg9i2H+zvyISsxoxJ0kqn3ki2DzUEObaVGPov3l35zorRM5v2w+FI0LD8efQ6fDLqwhVzTFz+I/OXSJ1lP6KbDOaDMsBmMzm2nWa6gzfGwlfgR7fResV6o9U4lBrZI7iqmyOt/TqSeLZZpR7bEoDjW0meZt9O1gJefO2p1gdX9sflV7BSu1rHJyje2HonFww6fyRuxotKfkWOHE+iv1xLloM43ZTOvAcjAwBm2mMbu6XgVWU3/QYEQHkEjP8nqiWMulYtD2qDmeyNdhV40ifD9U18M4tJmGtoF6Zx83YLVk2klur8d4xR5O9cKpvJ4jh6JBHtgd1ztQbcBq3G8hqm/X5zX0o83IYro9WY17CpXexBh/4LALiWJQi+Yr2+jk2I2qNepoMzo9Ngrx6EObaZmtNCPyvYrTPWB01+nGV1A5lb4b68ld1w2OHYreFdYsTBvuRX6L0+udzL1K9ECYL4rxYBzaShuBrsB4tF9Bt87VvsM4Ngc1tKsa2oj5s7hXgr2AtqF0BYtHDW2moc2oxCA2Z2buLk73xcn8LHdVG4H+5R++h6Iiq420Ot/zU/xIvSvRyzLyDeH/Q/4pg9lK81cFFY+20iqw/B1Y/zDNUz1oeD+Ltf71+VDD+Mj2RD5P9Wd5FVhb1VOKqG/RZhqzTWM5PbO+dyfq4VkquSoxX+p8D0UNdjffar7V+VVurVPFv5wVzIca2krLsI9E9rEwKjGebnyXmfp2PgBRTOQbxI+2aT8/P+F/vFHNU0S+VbCeaGd0+83D4lFDW+HjsjmZ/xRRHSNfl9VcbD7TqszOnZ33SXwPRYRuY3TjPStzh/sg7Lg+jcqLuBJzA9sH7gftU7D6M43B4lBD26N8pkc9yjTE56lQjfOoOikdqcYxqnNZHGpor6B6+kn43lq5VliZ350bxUe+38SvPRSdbIDug9KJ/W1UX6g+To3NZlpke2y+v1bZkSPDeqzaaxiP92jsNf8sVJ8LnKNQ61fmVujOV3VU+iC+2Z7KerKiMRvzYYyiGvfb6fZYN/5Ln197KMpQzdfVq6zOP8WT9sVezkxDmI4as9k1g5rLNCPyeapxY0Mts/mRHw8tdmWaIvKNgv8VZLVC/0rfZX2LWmajVhk/Beyrp3BqTyqv0r/8m++hKEA1UVc3socz8r2CbD+Z/xbspYwvZ7QZlZgumFPtzxP5TtOtKfZ0NB9jTUNYnAd9aD+ZV9bWwD2gzWDPWDS+TdYzTyPbK/OjhrbSRqB/+Zv/Ag9g3WYLU8tLAAAAAElFTkSuQmCC"
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
        if frame_log.winfo_ismapped(): frame_log.pack_forget()
        else: frame_log.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(0, 15))

    view_menu.add_command(label="Toggle Terminal", command=toggle_terminal)
    menubar.add_cascade(label="View", menu=view_menu)
    root.config(menu=menubar)

    frame_log = ttk.LabelFrame(root, text="Terminal", padding=10)
    frame_log.pack(side=tk.BOTTOM, fill=tk.X, padx=15, pady=(0, 15))
    log_scroll = ttk.Scrollbar(frame_log)
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text_log = tk.Text(frame_log, height=8, bg="#1E1E1E", fg="#CCCCCC", font=("Consolas", 10), yscrollcommand=log_scroll.set, relief="flat")
    text_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    log_scroll.config(command=text_log.yview)
    sys.stdout = RedirectText(text_log)
    sys.stderr = RedirectText(text_log)

    # --- Draggable PanedWindow System ---
    paned_window = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned_window.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

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

    # --- Debounced Resize Bindings (FIXES LAG) ---
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

    # --- Group 1: Gabor Library ---
    frame_gabor = ttk.LabelFrame(frame_left, text="Gabor Library", padding=15)
    frame_gabor.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 10), padx=10)
    frame_gabor.columnconfigure(1, weight=1)
    
    gabor_entries = {}
    for i, (label, default) in enumerate(gabor_param.items()):
        tk.Label(frame_gabor, text=label, bg=frame_color, fg=text_color, font=main_font).grid(row=i, column=0, sticky="w", pady=3)
        
        entry_wrap = ttk.Frame(frame_gabor, style="TFrame")
        entry_wrap.grid(row=i, column=1, pady=3, padx=(10,0), sticky="ew")
        entry_wrap.columnconfigure(0, weight=1)
        
        entry = tk.Entry(entry_wrap, relief="solid", bd=1)
        entry.insert(0, default)
        entry.grid(row=0, column=0, sticky="ew")
        ToolTip(entry)
        gabor_entries[label] = entry
        
        btn_browse = ttk.Button(entry_wrap, text="📁", style="Browse.TButton", width=3, command=lambda e=entry: popup_browser(e))
        btn_browse.grid(row=0, column=1, padx=(5,0))
        
    btn_submit_gabor = ttk.Button(frame_gabor, text="Create Gabor Library", style="Primary.TButton", command=run_in_thread(create_gabor))
    btn_submit_gabor.grid(row=len(gabor_param), column=0, columnspan=2, pady=(15,0), sticky="ew")

    # --- Group 2: Model Parameters ---
    frame_params = ttk.LabelFrame(frame_left, text="Model Parameters", padding=15)
    frame_params.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_params.columnconfigure(1, weight=1)
    
    param_entries = {}
    for i, (label, default) in enumerate(param_defaults.items()):
        tk.Label(frame_params, text=label, bg=frame_color, fg=text_color, font=main_font).grid(row=i, column=0, sticky="w", pady=3)
        
        entry_wrap = ttk.Frame(frame_params, style="TFrame")
        entry_wrap.grid(row=i, column=1, pady=3, padx=(10,0), sticky="ew")
        entry_wrap.columnconfigure(0, weight=1)
        
        entry = tk.Entry(entry_wrap, relief="solid", bd=1)
        entry.insert(0, default)
        entry.grid(row=0, column=0, sticky="ew")
        ToolTip(entry)
        param_entries[label] = entry
        
        btn_browse = ttk.Button(entry_wrap, text="📁", style="Browse.TButton", width=3, command=lambda e=entry: popup_browser(e))
        btn_browse.grid(row=0, column=1, padx=(5,0))

    # --- Group 3: Single RF Analysis ---
    frame_single_rf = ttk.LabelFrame(frame_left, text="Single RF Analysis", padding=15)
    frame_single_rf.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
    frame_single_rf.columnconfigure(1, weight=1)
    
    tk.Label(frame_single_rf, text='Neuron ID', bg=frame_color, fg=text_color, font=bold_font).grid(row=0, column=0, sticky="w", pady=3)
    
    rf_wrap = ttk.Frame(frame_single_rf, style="TFrame")
    rf_wrap.grid(row=0, column=1, pady=3, padx=(10,0), sticky="ew")
    rf_wrap.columnconfigure(0, weight=1)
    
    entry_neuron = tk.Entry(rf_wrap, relief="solid", bd=1)
    entry_neuron.insert(0, '1173')
    entry_neuron.grid(row=0, column=0, sticky="ew")
    param_entries['Neuron ID'] = entry_neuron

    btn_runRF = ttk.Button(frame_single_rf, text="Run Single RF", style="Primary.TButton")
    btn_runRF.grid(row=1, column=0, columnspan=2, pady=(15,0), sticky="ew")

    # --- Group 4: Export Options ---
    frame_export = ttk.LabelFrame(frame_left, text="Export Options", padding=15)
    frame_export.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    btn_save_ret = ttk.Button(frame_export, text="Export Retinotopy Matrix", style="Primary.TButton", command=click_save)
    btn_save_ret.pack(fill=tk.X, pady=3)

    btn_export = ttk.Button(frame_export, text="Export Plots as SVG", style="Primary.TButton", command=export_plots)
    btn_export.pack(fill=tk.X, pady=3)

    # --- Global Controls ---
    frame_controls = ttk.Frame(frame_left, style="TFrame")
    frame_controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(10, 5))

    btn_submit_wavelet = ttk.Button(frame_controls, text="Run Wavelet Transform", style="Primary.TButton", command=run_in_thread(run_wavelet))
    btn_submit_wavelet.pack(fill=tk.X, pady=3)

    btn_submit_plot = ttk.Button(frame_controls, text="Run Primary Analysis", style="Success.TButton", command=run_in_thread(plot_data))
    btn_submit_plot.pack(fill=tk.X, pady=3)

    btn_quit = ttk.Button(frame_controls, text="Quit Application", style="Danger.TButton", command=quit_app)
    btn_quit.pack(fill=tk.X, pady=(15, 5))

    all_buttons = [btn_submit_gabor, btn_submit_wavelet, btn_submit_plot, btn_runRF, btn_save_ret, btn_export, btn_quit]

    root.mainloop()