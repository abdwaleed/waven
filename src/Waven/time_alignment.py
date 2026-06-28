"""Time-alignment entry points for two-photon and electrophysiology workflows.

Two-photon alignment delegates to the existing Cortex Lab / suite2p pipeline.
Electrophysiology alignment is provided as a template for user implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

WORKFLOW_2P = "2p"
WORKFLOW_EPHYS = "ephys"


@dataclass
class AlignedNeuralData:
    """Aligned spike responses and neuron positions shared by both workflows."""

    spikes: np.ndarray
    neuron_pos: np.ndarray
    aligned_spikes: Optional[np.ndarray] = None


def load_two_photon_spikes(
    experiment_info: Tuple[str, str, int],
    data_dirs: Sequence[str],
    suite2p_dir: str,
    block_end: int,
    n_planes: int,
    nb_frames: int,
    resolution: float,
    *,
    spks_path: Optional[Path] = None,
    threshold: float = 1.25,
    method: str = "frame2ttl",
    correct_positions: bool = True,
) -> AlignedNeuralData:
    """Load and time-align two-photon (suite2p) spike data.

    When ``spks_path`` is set, reads pre-aligned ``spikes.npy`` and sibling
    ``pos.npy`` and skips suite2p alignment.
    """
    from . import LoadPinkNoise as lpn

    if spks_path is None:
        spikes, aligned_spikes, neuron_pos = lpn.loadSPKMesoscope(
            experiment_info,
            list(data_dirs),
            suite2p_dir,
            block_end,
            n_planes,
            nb_frames,
            threshold=threshold,
            last=True,
            method=method,
        )
        if correct_positions:
            neuron_pos = lpn.correctNeuronPos(neuron_pos, resolution)
    else:
        spikes = np.load(spks_path, mmap_mode="r")
        pos_path = spks_path.parent / "pos.npy"
        neuron_pos = np.load(pos_path)
        aligned_spikes = None

    return AlignedNeuralData(
        spikes=spikes,
        neuron_pos=neuron_pos,
        aligned_spikes=aligned_spikes,
    )


def align_ephys_data(
    data_dir: Path,
    nb_frames: int,
    sampling_rate: float,
    **kwargs: Any,
) -> AlignedNeuralData:
    """Time-align electrophysiology recordings to the stimulus timeline.

    Implement this function for your ephys data layout.  ``data_dir`` should
    contain a ``.pkl`` metadata file and a subfolder of ``.din`` trace files.

    The returned object MUST provide:

    - ``spikes``: ``np.ndarray`` with shape ``(n_trials, n_timepoints, n_neurons)``
      containing trial-aligned spike counts or firing rates, binned to the same
      frame grid used by the stimulus wavelets (``n_timepoints`` is typically
      ``nb_frames`` per trial).

    - ``neuron_pos``: ``np.ndarray`` with shape ``(n_neurons, 2)`` giving each
      unit's spatial coordinates (same convention as the 2p pipeline: column 0
      is X, column 1 is Y, in micrometers or another consistent spatial unit).

    - ``aligned_spikes`` (optional): ``np.ndarray`` with the same shape as
      ``spikes``, or ``None`` if not used.

    Downstream RF analysis validates that ``neuron_pos.shape[0] == spikes.shape[2]``.
    """
    raise NotImplementedError(
        "Implement align_ephys_data() in time_alignment.py for your ephys data. "
        "See the docstring for the required output structure."
    )


def load_aligned_spikes(
    workflow: str,
    *,
    experiment_info: Tuple[str, str, int],
    data_dir: Path,
    data_dir_strings: Sequence[str],
    suite2p_dir: Path,
    block_end: int,
    n_planes: Optional[int],
    nb_frames: int,
    resolution: Optional[float],
    sampling_rate: Optional[float],
    spks_path: Optional[Path] = None,
    threshold: float = 1.25,
    method: str = "frame2ttl",
    correct_positions: bool = True,
) -> AlignedNeuralData:
    """Dispatch spike loading to the workflow-specific alignment routine."""
    if spks_path is not None:
        spikes = np.load(spks_path, mmap_mode="r")
        pos_path = spks_path.parent / "pos.npy"
        neuron_pos = np.load(pos_path)
        return AlignedNeuralData(spikes=spikes, neuron_pos=neuron_pos)

    if workflow == WORKFLOW_2P:
        if resolution is None or n_planes is None:
            raise ValueError("Two-photon workflow requires Resolution and Number of Planes")
        return load_two_photon_spikes(
            experiment_info,
            data_dir_strings,
            str(suite2p_dir),
            block_end,
            n_planes,
            nb_frames,
            resolution,
            threshold=threshold,
            method=method,
            correct_positions=correct_positions,
        )

    if workflow == WORKFLOW_EPHYS:
        if sampling_rate is None:
            raise ValueError("Ephys workflow requires Sampling Rate (samples / sec)")
        return align_ephys_data(
            data_dir,
            nb_frames,
            sampling_rate,
            experiment_info=experiment_info,
            threshold=threshold,
            method=method,
        )

    raise ValueError(f"Unknown workflow: {workflow!r}")


# Re-export core 2p alignment helpers for visibility in the codebase.
from .LoadPinkNoise import align_datas, correctNeuronPos, loadSPKMesoscope  # noqa: E402

__all__ = [
    "AlignedNeuralData",
    "WORKFLOW_2P",
    "WORKFLOW_EPHYS",
    "align_datas",
    "align_ephys_data",
    "correctNeuronPos",
    "load_aligned_spikes",
    "load_two_photon_spikes",
    "loadSPKMesoscope",
]
