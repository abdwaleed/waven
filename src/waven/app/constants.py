"""Static presentation vocabulary owned by the interactive application.

This module deliberately contains no Tkinter objects or numerical imports.  It
lets the GUI composition root focus on wiring actions and keeps user-facing
labels, hints, and export notes in one discoverable place.
"""
from __future__ import annotations

from typing import Dict

from ..config import WORKFLOW_2P


GABOR_FIELD_LABELS: Dict[str, str] = {
    "N_thetas": "Orientation Count",
    "Sigmas": "Filter Sizes (analysis px; use recommender for degrees)",
    "Frequencies": "Spatial Frequencies (cyc/analysis px; use recommender for cpd)",
    "Phases": "Phases (degrees)",
    "Save Path": "Gabor Cache Folder",
    "Coarse Library Path": "Coarse Gabor Folder",
    "Fine Library Path": "Fine Gabor Folder",
}

ANALYSIS_FIELD_LABELS: Dict[str, str] = {
    "Project Root": "Project Root Folder",
    "Dir": "Raw Data Folder",
    "Path Directory": "Coarse Wavelet Folder",
    "Experiment Info": "Experiment ID (mouse, date, #)",
    "Number of Planes": "Imaging Planes",
    "Block End": "Session Block Start Frame",
    "Resolution": "Microscope Resolution (µm/px)",
    "Sampling Rate (samples / sec)": "Recording Sampling Rate (Hz)",
    "Sigmas": "RF Filter Sizes (analysis px)",
    "Sigmas Full Model": "Full-Model Filter Sizes (analysis px)",
    "Frequencies": "Stimulus Frequencies (cyc/analysis px)",
    "Visual Coverage": "Visual Field Coverage (°)",
    "Analysis Coverage": "Analysis Field Coverage (°)",
    "Number of Frames": "Frames per Trial",
    "Movie Path": "Stimulus Movie Folder",
    "Library Path": "Gabor Library Folder",
    "Coarse Library Path": "Coarse Gabor Folder",
    "Fine Library Path": "Fine Gabor Folder",
    "Spks Path": "Neural Cache Folder",
    "Full Model Wavelet Path": "Full-Model Wavelet Folder",
    "Full Model Save Path": "Full-Model Results Directory",
    "Plot Cache Path": "Plot Cache Folder",
    "Recovery Cache Directory": "Recovery Checkpoint Directory",
    "Train Trial Indices": "Train Trials",
    "Test Trial Indices": "Test Trials",
    "Use Last Minute Holdout": "Last-Minute Holdout",
    "Model Fit Minutes": "Model Fit Duration (minutes)",
    "Neuron ID": "Neuron Index",
}

BROWSE_KIND_BY_FIELD: Dict[str, str] = {
    "Project Root": "dir",
    "Save Path": "dir",
    "Movie Path": "dir",
    "Library Path": "dir",
    "Coarse Library Path": "dir",
    "Fine Library Path": "dir",
    "Spks Path": "dir",
    "Path Directory": "dir",
    "Dir": "dir",
    "Full Model Wavelet Path": "dir",
    "Full Model Save Path": "dir",
    "Plot Cache Path": "dir",
    "Recovery Cache Directory": "dir",
}

INPUT_HINTS: Dict[str, str] = {
    "Experiment Info": "Tuple: ('subject', 'YYYY-MM-DD', experiment_number)",
    "Number of Planes": "Integer, e.g. 1",
    "Sampling Rate (samples / sec)": "Number in Hz, e.g. 30000",
    "Train Trial Indices": "'auto' or zero-based list, e.g. [0, 2]",
    "Test Trial Indices": "'auto' or zero-based list, e.g. [1]",
    "Use Last Minute Holdout": "Boolean: True or False",
    "Model Fit Minutes": "Positive whole number of stimulus minutes used for fitting, e.g. 5",
    "Block End": "Integer acquisition block index, e.g. 0",
    "Resolution": "Micrometers per pixel, e.g. 1.3671",
    "Visual Coverage": "[left, right, top, bottom] in degrees",
    "Analysis Coverage": "[left, right, top, bottom] in degrees",
    "N_thetas": "Integer orientation bins, e.g. 18",
    "Sigmas": "Numeric list in square-degree analysis pixels, e.g. [2, 4, 8]",
    "Sigmas Full Model": "Numeric list in square-degree analysis pixels, e.g. [2, 4, 8, 12]",
    "Frequencies": "Numeric list in cycles/analysis pixel; displayed/exported values are converted to cpd",
    "Phases": "Numeric list in degrees, e.g. [0, 90]",
    "Neuron ID": "Zero-based neuron index, e.g. 1173",
}

BROWSE_ICONS: Dict[str, str] = {"file": "📄", "savefile": "📄", "dir": "📁"}

ORIENTATION_EXPORT_COMPARISON_NOTE = """WAVEN ORIENTATION-TUNING EXPORT NOTE

These curves are derived from the Coarse RF Gabor-wavelet analysis, not from
traditional discrete orientation-stimulus trials. The firing-rate curve is a
wavelet-energy-weighted firing-rate response at the neuron's preferred RF
feature. The correlation curve is the Pearson correlation between the
frame-aligned neural response and each orientation's Gabor-wavelet feature.

`trial_values` contains one frame-aligned trial estimate per orientation bin.
It should therefore not be interpreted as a conventional repeated-static-
orientation stimulus trial unless the stimulus design independently supports
that interpretation. Preferred orientation is the discrete orientation bin
with the largest mean value; no interpolation or curve fitting is applied.
"""


def workflow_display_name(workflow: str) -> str:
    """Return the stable user-facing label for a workflow identifier."""
    return "Two-Photon" if workflow == WORKFLOW_2P else "Electrophysiology"
