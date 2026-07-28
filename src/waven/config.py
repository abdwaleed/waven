"""Configuration helpers for waven analysis pipelines.

The GUI historically stores values as strings so they can be edited in
text fields.  This module keeps that interface available while providing
typed, validated objects for scripts and tests.
"""
from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .stimulus.metadata import validate_coverage_pair


DEFAULT_GABOR_PARAMS: Dict[str, str] = {
    "N_thetas": "8",
    "Sigmas": "[2, 3, 4, 5, 6, 8]",
    "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
    # GUI values are degrees; conversion happens at the convolution boundary.
    "Phases": "[0, 90]",
    "NX": "135",
    "NY": "54",
    "Save Path": "your_experiment/cache/gabor/full",
}

WORKFLOW_2P = "2p"
WORKFLOW_EPHYS = "ephys"

DEFAULT_COMMON_PARAMS: Dict[str, str] = {
    "Project Root": "your_experiment",
    "Dir": "data",
    "Path Directory": "your_experiment/cache/wavelets/coarse",
    "Experiment Info": "('SS002', '2024-07-23', 3)",
    "Block End": "0",
    "screen_x": "4096",
    "screen_y": "1536",
    "NX": "135",
    "NY": "54",
    "Sigmas": "[2, 3, 4, 5, 6, 8]",
    "Sigmas Full Model": "[2, 4, 6, 8, 10]",
    "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
    "Visual Coverage": "[-135, 45, 34, -34]",
    "Analysis Coverage": "[-135, 0, 34, -34]",
    "Hz": "30",
    "Number of Frames": "18000",
    "Number of Trials to Keep": "3",
    "Excluded Trial Numbers": "[]",
    "Movie Path": (
        "your_experiment/input/stimulus_movie"
    ),
    "Library Path": "your_experiment/cache/gabor/full",
    "Spks Path": "your_experiment/input/neural_cache",
    "Full Model Wavelet Path": "your_experiment/cache/wavelets/full",
    "Full Model Save Path": "your_experiment/output/models",
    "Plot Cache Path": "",
    "Recovery Cache Directory": "",
    # Alternating automatic splits remain valid for every session with at
    # least two trials; fixed [0, 2] did not work for two-trial recordings.
    "Train Trial Indices": "auto",
    "Test Trial Indices": "auto",
    "Use Last Minute Holdout": "False",
    "Model Fit Minutes": "5",
}

DEFAULT_TWO_PHOTON_PARAMS: Dict[str, str] = {
    "Resolution": "1.3671",
    "Number of Planes": "1",
}

DEFAULT_EPHYS_PARAMS: Dict[str, str] = {
    "Sampling Rate (samples / sec)": "30000",
    "Photodiode Port": "3",
}

# Legacy flat dict kept for backward-compatible merges.
DEFAULT_ANALYSIS_PARAMS: Dict[str, str] = {
    **DEFAULT_COMMON_PARAMS,
    **DEFAULT_TWO_PHOTON_PARAMS,
}

COMMON_PARAM_KEYS = tuple(DEFAULT_COMMON_PARAMS.keys())
TWO_PHOTON_PARAM_KEYS = tuple(DEFAULT_TWO_PHOTON_PARAMS.keys())
EPHYS_PARAM_KEYS = tuple(DEFAULT_EPHYS_PARAMS.keys())

NONE_STRINGS = {"", "none", "null", "nil"}


def parse_literal(value: Any, field_name: str = "value") -> Any:
    """Parse a Python literal from a string without executing code."""
    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped.lower() in NONE_STRINGS:
        return None

    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid literal for {field_name}: {value!r}") from exc


def parse_path(value: Any, field_name: str) -> Path:
    """Return a normalized path, expanding ``~`` and environment variables."""
    if value is None:
        raise ValueError(f"{field_name} is required")
    path = os.path.expandvars(os.path.expanduser(str(value)))
    return Path(path)


def parse_optional_path(value: Any) -> Optional[Path]:
    """Return ``None`` for empty/None-like values, otherwise a normalized path."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in NONE_STRINGS:
        return None
    return parse_path(value, "path")


def sibling_path_with_suffix(path: Any, suffix: str) -> Path:
    """Return ``path`` with ``suffix`` inserted before the extension."""
    base = parse_path(path, "path")
    if base.suffix:
        return base.with_name(f"{base.stem}{suffix}{base.suffix}")
    return base.with_name(f"{base.name}{suffix}.npy")


def _as_tuple(value: Any, field_name: str) -> Tuple[Any, ...]:
    """Function for as tuple.

    Args:
        value: Input value for this operation.
        field_name: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    parsed = parse_literal(value, field_name)
    if parsed is None:
        return tuple()
    if isinstance(parsed, np.ndarray):
        parsed = parsed.tolist()
    if isinstance(parsed, (list, tuple)):
        return tuple(parsed)
    return (parsed,)


def _as_float_tuple(value: Any, field_name: str) -> Tuple[float, ...]:
    """Function for as float tuple.

    Args:
        value: Input value for this operation.
        field_name: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return tuple(float(item) for item in _as_tuple(value, field_name))


def _as_int(value: Any, field_name: str) -> int:
    """Function for as int.

    Args:
        value: Input value for this operation.
        field_name: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    parsed = parse_literal(value, field_name)
    if parsed is None:
        raise ValueError(f"{field_name} is required")
    return int(parsed)


def _as_positive_int_tuple(value: Any, field_name: str) -> Tuple[int, ...]:
    """Parse a one-indexed trial-number list and reject invalid entries."""
    values = []
    for item in _as_tuple(value, field_name):
        if isinstance(item, bool):
            raise ValueError(f"{field_name} must contain positive whole numbers")
        number = int(item)
        if number != item or number < 1:
            raise ValueError(f"{field_name} must contain positive whole numbers")
        if number not in values:
            values.append(number)
    return tuple(values)


def _as_float(value: Any, field_name: str) -> float:
    """Function for as float.

    Args:
        value: Input value for this operation.
        field_name: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    parsed = parse_literal(value, field_name)
    if parsed is None:
        raise ValueError(f"{field_name} is required")
    return float(parsed)


def _as_bool(value: Any, field_name: str) -> bool:
    """Function for as bool.

    Args:
        value: Input value for this operation.
        field_name: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    parsed = parse_literal(value, field_name)
    if isinstance(parsed, bool):
        return parsed
    if isinstance(parsed, str):
        normalized = parsed.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if parsed in (0, 1):
        return bool(parsed)
    raise ValueError(f"{field_name} must be a boolean value")


def _as_path_tuple(value: Any, field_name: str) -> Tuple[Path, ...]:
    """Function for as path tuple.

    Args:
        value: Input value for this operation.
        field_name: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    if value is None:
        return tuple()

    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NONE_STRINGS:
            return tuple()
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return (parse_path(stripped, field_name),)
    else:
        parsed = value

    if parsed is None:
        return tuple()
    if isinstance(parsed, (list, tuple)):
        return tuple(parse_path(item, field_name) for item in parsed)
    return (parse_path(parsed, field_name),)


def _get(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Function for get.

    Args:
        mapping: Input value for this operation.
        key: Input value for this operation.
        default: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return mapping[key] if key in mapping else default


def _path_to_gui(path: Optional[Path]) -> str:
    """Function for path to gui.

    Args:
        path: Input value for this operation.

    Returns:
        Result produced by the operation.
    """
    return "None" if path is None else str(path)


def coarse_grid_dimensions(nx: int, ny: int) -> Tuple[int, int]:
    """Return coarse RF grid size as 20% of full NX/NY (rounded)."""
    return round(0.2 * nx), round(0.2 * ny)


def coarse_to_full_scale(nx: int, ny: int) -> Tuple[float, float]:
    """Return multipliers from coarse RF grid indices to full-resolution pixels."""
    coarse_nx, coarse_ny = coarse_grid_dimensions(nx, ny)
    scale_x = nx / coarse_nx if coarse_nx else 1.0
    scale_y = ny / coarse_ny if coarse_ny else 1.0
    return scale_x, scale_y


def resolve_sigma_indices(
    library_sigmas: Sequence[float],
    requested_sigmas: Sequence[float],
) -> Tuple[int, ...]:
    """Map requested sigma values to indices in the Gabor library."""
    lib = tuple(float(s) for s in library_sigmas)
    indices = []
    for sigma in requested_sigmas:
        value = float(sigma)
        try:
            indices.append(lib.index(value))
        except ValueError as exc:
            raise ValueError(
                f"Sigma {value} is not present in the Gabor library sigmas {lib}"
            ) from exc
    return tuple(indices)


def _parse_data_dir(mapping: Mapping[str, Any], field_name: str = "Dir") -> Path:
    """Parse the primary data directory, accepting legacy ``Dirs`` key."""
    value = _get(mapping, "Dir") or _get(mapping, "Dirs")
    if value is None:
        raise ValueError(f"{field_name} is required")
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in NONE_STRINGS:
            raise ValueError(f"{field_name} is required")
        try:
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, (list, tuple)) and parsed:
                return parse_path(parsed[0], field_name)
        except (SyntaxError, ValueError):
            pass
        return parse_path(stripped, field_name)
    if isinstance(value, (list, tuple)) and value:
        return parse_path(value[0], field_name)
    return parse_path(value, field_name)


@dataclass(frozen=True)
class GaborConfig:
    """Typed configuration for the Gabor filter library."""

    n_thetas: int
    sigmas: Tuple[float, ...]
    frequencies: Tuple[float, ...]
    phases: Tuple[float, ...]
    nx: int
    ny: int
    save_path: Path
    coarse_save_path: Path
    fine_save_path: Path

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "GaborConfig":
        """Function for from mapping.

        Args:
            mapping: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        save_path = parse_path(_get(mapping, "Save Path"), "Save Path")

        def library_path(value, kind):
            """Resolve a configured library folder or legacy file to one artifact."""
            if value is not None:
                path = parse_path(value, f"{kind.title()} Library Path")
                return path if path.suffix else path / f"gabor_library_{kind}.npy"
            if save_path.suffix:
                return sibling_path_with_suffix(save_path, f"_{kind}")
            folder = save_path
            if kind == "coarse" and save_path.name.lower() in {"full", "fine"}:
                folder = save_path.parent / "coarse"
            elif kind == "fine" and save_path.name.lower() == "coarse":
                folder = save_path.parent / "full"
            return folder / f"gabor_library_{kind}.npy"

        coarse_path = (
            library_path(_get(mapping, "Coarse Library Path"), "coarse")
        )
        fine_path = (
            library_path(_get(mapping, "Fine Library Path"), "fine")
        )
        return cls(
            n_thetas=_as_int(_get(mapping, "N_thetas"), "N_thetas"),
            sigmas=_as_float_tuple(_get(mapping, "Sigmas"), "Sigmas"),
            frequencies=_as_float_tuple(
                _get(mapping, "Frequencies"),
                "Frequencies",
            ),
            phases=_as_float_tuple(_get(mapping, "Phases"), "Phases"),
            nx=_as_int(_get(mapping, "NX"), "NX"),
            ny=_as_int(_get(mapping, "NY"), "NY"),
            save_path=save_path,
            coarse_save_path=coarse_path,
            fine_save_path=fine_path,
        )

    @property
    def x_positions(self) -> np.ndarray:
        """Function for x positions.

        Returns:
            Result produced by the operation.
        """
        return np.arange(self.nx)

    @property
    def y_positions(self) -> np.ndarray:
        """Function for y positions.

        Returns:
            Result produced by the operation.
        """
        return np.arange(self.ny)

    @property
    def theta_radians(self) -> np.ndarray:
        """Function for theta radians.

        Returns:
            Result produced by the operation.
        """
        return np.array(
            [(index * np.pi) / self.n_thetas for index in range(self.n_thetas)]
        )

    @property
    def sigmas_array(self) -> np.ndarray:
        """Function for sigmas array.

        Returns:
            Result produced by the operation.
        """
        return np.array(self.sigmas)

    @property
    def phases_array(self) -> np.ndarray:
        """Function for phases array.

        Returns:
            Result produced by the operation.
        """
        return np.array(self.phases)

    @property
    def frequencies_array(self) -> np.ndarray:
        """Function for frequencies array.

        Returns:
            Result produced by the operation.
        """
        return np.array(self.frequencies)

    @property
    def has_independent_frequencies(self) -> bool:
        """Function for has independent frequencies.

        Returns:
            Result produced by the operation.
        """
        return bool(self.frequencies) and any(freq != 0 for freq in self.frequencies)

    def to_gui_mapping(self) -> Dict[str, str]:
        """Function for to gui mapping.

        Returns:
            Result produced by the operation.
        """
        return {
            "N_thetas": str(self.n_thetas),
            "Sigmas": repr(list(self.sigmas)),
            "Frequencies": repr(list(self.frequencies)),
            "Phases": repr(list(self.phases)),
            "NX": str(self.nx),
            "NY": str(self.ny),
            "Save Path": str(self.save_path),
            "Coarse Library Path": str(self.coarse_save_path),
            "Fine Library Path": str(self.fine_save_path),
        }


@dataclass(frozen=True)
class AnalysisConfig:
    """Typed configuration for stimulus, neural, and RF analysis."""

    workflow: str
    path_directory: Path
    data_dir: Path
    experiment_info: Tuple[str, str, int]
    block_end: int
    screen_x: int
    screen_y: int
    nx: int
    ny: int
    sigmas: Tuple[float, ...]
    sigmas_full_model: Tuple[float, ...]
    frequencies: Tuple[float, ...]
    visual_coverage: Tuple[float, float, float, float]
    analysis_coverage: Tuple[float, float, float, float]
    hz: int
    nb_frames: int
    n_trials_to_keep: int
    excluded_trial_numbers: Tuple[int, ...]
    train_trial_indices: str
    test_trial_indices: str
    use_last_minute_holdout: bool
    model_fit_minutes: int
    movie_path: Path
    library_path: Path
    resolution: Optional[float] = None
    n_planes: Optional[int] = None
    sampling_rate: Optional[float] = None
    photodiode_port: Optional[int] = None
    spks_path: Optional[Path] = None
    full_model_wavelet_path: Optional[Path] = None
    full_model_save_path: Optional[Path] = None
    plot_cache_path: Optional[Path] = None
    recovery_cache_dir: Optional[Path] = None

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        workflow: str = WORKFLOW_2P,
    ) -> "AnalysisConfig":
        """Function for from mapping.

        Args:
            mapping: Input value for this operation.
            workflow: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        experiment_info = _as_tuple(
            _get(mapping, "Experiment Info"),
            "Experiment Info",
        )
        if len(experiment_info) != 3:
            raise ValueError("Experiment Info must be (mouse, date, number)")

        visual_coverage = _as_float_tuple(
            _get(mapping, "Visual Coverage"),
            "Visual Coverage",
        )
        analysis_coverage = _as_float_tuple(
            _get(mapping, "Analysis Coverage"),
            "Analysis Coverage",
        )
        if len(visual_coverage) != 4 or len(analysis_coverage) != 4:
            raise ValueError("Coverage values must contain four numbers")
        validate_coverage_pair(visual_coverage, analysis_coverage)

        model_fit_minutes = _as_int(
            _get(mapping, "Model Fit Minutes", DEFAULT_COMMON_PARAMS["Model Fit Minutes"]),
            "Model Fit Minutes",
        )
        if model_fit_minutes <= 0:
            raise ValueError("Model Fit Minutes must be a positive whole number")

        resolution: Optional[float] = None
        n_planes: Optional[int] = None
        sampling_rate: Optional[float] = None
        photodiode_port: Optional[int] = None

        if workflow == WORKFLOW_2P:
            resolution = _as_float(_get(mapping, "Resolution"), "Resolution")
            n_planes = _as_int(
                _get(mapping, "Number of Planes"),
                "Number of Planes",
            )
        elif workflow == WORKFLOW_EPHYS:
            sampling_rate = _as_float(
                _get(mapping, "Sampling Rate (samples / sec)"),
                "Sampling Rate (samples / sec)",
            )
            photodiode_port = _as_int(
                _get(mapping, "Photodiode Port"),
                "Photodiode Port",
            )
            if photodiode_port < 0:
                raise ValueError("Photodiode Port must be a non-negative digital-input port number")
        else:
            raise ValueError(f"Unknown workflow: {workflow!r}")

        return cls(
            workflow=workflow,
            path_directory=parse_path(
                _get(mapping, "Path Directory"),
                "Path Directory",
            ),
            data_dir=_parse_data_dir(mapping),
            experiment_info=(
                str(experiment_info[0]),
                str(experiment_info[1]),
                int(experiment_info[2]),
            ),
            block_end=_as_int(_get(mapping, "Block End"), "Block End"),
            screen_x=_as_int(_get(mapping, "screen_x"), "screen_x"),
            screen_y=_as_int(_get(mapping, "screen_y"), "screen_y"),
            nx=_as_int(_get(mapping, "NX"), "NX"),
            ny=_as_int(_get(mapping, "NY"), "NY"),
            resolution=resolution,
            n_planes=n_planes,
            sampling_rate=sampling_rate,
            photodiode_port=photodiode_port,
            sigmas=_as_float_tuple(_get(mapping, "Sigmas"), "Sigmas"),
            sigmas_full_model=_as_float_tuple(
                _get(mapping, "Sigmas Full Model"),
                "Sigmas Full Model",
            ),
            frequencies=_as_float_tuple(
                _get(mapping, "Frequencies"),
                "Frequencies",
            ),
            visual_coverage=visual_coverage,  # type: ignore[arg-type]
            analysis_coverage=analysis_coverage,  # type: ignore[arg-type]
            hz=_as_int(_get(mapping, "Hz"), "Hz"),
            nb_frames=_as_int(_get(mapping, "Number of Frames"), "Number of Frames"),
            n_trials_to_keep=_as_int(
                _get(mapping, "Number of Trials to Keep"),
                "Number of Trials to Keep",
            ),
            excluded_trial_numbers=_as_positive_int_tuple(
                _get(mapping, "Excluded Trial Numbers", "[]"),
                "Excluded Trial Numbers",
            ),
            train_trial_indices=str(_get(mapping, "Train Trial Indices") or "auto"),
            test_trial_indices=str(_get(mapping, "Test Trial Indices") or "auto"),
            use_last_minute_holdout=_as_bool(
                _get(mapping, "Use Last Minute Holdout"),
                "Use Last Minute Holdout",
            ),
            model_fit_minutes=model_fit_minutes,
            movie_path=parse_path(_get(mapping, "Movie Path"), "Movie Path"),
            library_path=parse_path(_get(mapping, "Library Path"), "Library Path"),
            spks_path=parse_optional_path(_get(mapping, "Spks Path")),
            full_model_wavelet_path=parse_optional_path(
                _get(mapping, "Full Model Wavelet Path"),
            ),
            full_model_save_path=parse_optional_path(
                _get(mapping, "Full Model Save Path"),
            ),
            plot_cache_path=parse_optional_path(_get(mapping, "Plot Cache Path")),
            recovery_cache_dir=parse_optional_path(
                _get(mapping, "Recovery Cache Directory"),
            ),
        )

    @property
    def data_dirs(self) -> Tuple[Path, ...]:
        """Function for data dirs.

        Returns:
            Result produced by the operation.
        """
        return (self.data_dir,)

    @property
    def coarse_nx(self) -> int:
        """Function for coarse nx.

        Returns:
            Result produced by the operation.
        """
        return coarse_grid_dimensions(self.nx, self.ny)[0]

    @property
    def coarse_ny(self) -> int:
        """Function for coarse ny.

        Returns:
            Result produced by the operation.
        """
        return coarse_grid_dimensions(self.nx, self.ny)[1]

    @property
    def data_dir_strings(self) -> Sequence[str]:
        """Function for data dir strings.

        Returns:
            Result produced by the operation.
        """
        return [str(path) for path in self.data_dirs]

    @property
    def experiment_dir(self) -> Path:
        """Function for experiment dir.

        Returns:
            Result produced by the operation.
        """
        if not self.data_dirs:
            raise ValueError("At least one data directory is required")
        subject, date, experiment_number = self.experiment_info
        return self.data_dirs[0] / subject / date / str(experiment_number)

    @property
    def suite2p_dir(self) -> Path:
        """Function for suite2p dir.

        Returns:
            Result produced by the operation.
        """
        return self.experiment_dir / "suite2p"

    @property
    def screen_ratio(self) -> float:
        """Function for screen ratio.

        Returns:
            Result produced by the operation.
        """
        return abs(self.visual_coverage[0] - self.visual_coverage[1]) / self.nx

    @property
    def sigmas_array(self) -> np.ndarray:
        """Function for sigmas array.

        Returns:
            Result produced by the operation.
        """
        return np.array(self.sigmas)

    @property
    def sigmas_deg(self) -> np.ndarray:
        """Function for sigmas deg.

        Returns:
            Result produced by the operation.
        """
        x_max, x_min, _, _ = self.analysis_coverage
        deg_per_pix = abs(x_max - x_min) / self.nx
        return np.trunc(2 * deg_per_pix * self.sigmas_array * 100) / 100

    @property
    def frames_per_minute(self) -> int:
        """Number of movie frames in one minute of stimulus."""
        return self.hz * 60

    def coverage_ratios(self) -> Tuple[float, float]:
        """Function for coverage ratios.

        Returns:
            Result produced by the operation.
        """
        if self.visual_coverage == self.analysis_coverage:
            return 1.0, 1.0

        visual = np.array(self.visual_coverage)
        analysis = np.array(self.analysis_coverage)
        ratio_x = 1 - (
            ((visual[0] - visual[1]) - (analysis[0] - analysis[1]))
            / (visual[0] - visual[1])
        )
        ratio_y = 1 - (
            ((visual[2] - visual[3]) - (analysis[2] - analysis[3]))
            / (visual[2] - visual[3])
        )
        return float(ratio_x), float(ratio_y)

    def to_gui_mapping(self) -> Dict[str, str]:
        """Function for to gui mapping.

        Returns:
            Result produced by the operation.
        """
        mapping = {
            "Dir": str(self.data_dir),
            "Path Directory": str(self.path_directory),
            "Experiment Info": repr(self.experiment_info),
            "Block End": str(self.block_end),
            "screen_x": str(self.screen_x),
            "screen_y": str(self.screen_y),
            "NX": str(self.nx),
            "NY": str(self.ny),
            "Sigmas": repr(list(self.sigmas)),
            "Sigmas Full Model": repr(list(self.sigmas_full_model)),
            "Frequencies": repr(list(self.frequencies)),
            "Visual Coverage": repr(list(self.visual_coverage)),
            "Analysis Coverage": repr(list(self.analysis_coverage)),
            "Hz": str(self.hz),
            "Number of Frames": str(self.nb_frames),
            "Number of Trials to Keep": str(self.n_trials_to_keep),
            "Excluded Trial Numbers": repr(list(self.excluded_trial_numbers)),
            "Train Trial Indices": self.train_trial_indices,
            "Test Trial Indices": self.test_trial_indices,
            "Use Last Minute Holdout": str(self.use_last_minute_holdout),
            "Model Fit Minutes": str(self.model_fit_minutes),
            "Movie Path": str(self.movie_path),
            "Library Path": str(self.library_path),
            "Spks Path": _path_to_gui(self.spks_path),
            "Full Model Wavelet Path": _path_to_gui(self.full_model_wavelet_path),
            "Full Model Save Path": _path_to_gui(self.full_model_save_path),
            "Plot Cache Path": _path_to_gui(self.plot_cache_path),
            "Recovery Cache Directory": _path_to_gui(self.recovery_cache_dir),
        }
        if self.workflow == WORKFLOW_2P:
            mapping["Resolution"] = str(self.resolution)
            mapping["Number of Planes"] = str(self.n_planes)
        elif self.workflow == WORKFLOW_EPHYS:
            mapping["Sampling Rate (samples / sec)"] = str(self.sampling_rate)
            mapping["Photodiode Port"] = str(self.photodiode_port)
        return mapping

    @classmethod
    def gui_param_keys(cls, workflow: str) -> Tuple[str, ...]:
        """Return ordered GUI parameter keys for the selected workflow."""
        if workflow == WORKFLOW_2P:
            return COMMON_PARAM_KEYS + TWO_PHOTON_PARAM_KEYS
        if workflow == WORKFLOW_EPHYS:
            return COMMON_PARAM_KEYS + EPHYS_PARAM_KEYS
        raise ValueError(f"Unknown workflow: {workflow!r}")


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level configuration for a waven analysis run."""

    gabor: GaborConfig
    analysis: AnalysisConfig
    workflow: str = WORKFLOW_2P

    @classmethod
    def from_mappings(
        cls,
        gabor_params: Optional[Mapping[str, Any]] = None,
        analysis_params: Optional[Mapping[str, Any]] = None,
        workflow: str = WORKFLOW_2P,
    ) -> "PipelineConfig":
        """Function for from mappings.

        Args:
            gabor_params: Input value for this operation.
            analysis_params: Input value for this operation.
            workflow: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        merged_gabor = dict(DEFAULT_GABOR_PARAMS)
        merged_common = dict(DEFAULT_COMMON_PARAMS)
        merged_workflow = dict(
            DEFAULT_TWO_PHOTON_PARAMS
            if workflow == WORKFLOW_2P
            else DEFAULT_EPHYS_PARAMS
        )
        if gabor_params:
            merged_gabor.update(gabor_params)
        if analysis_params:
            merged_common.update(analysis_params)
            merged_workflow.update(analysis_params)
        merged_analysis = {**merged_common, **merged_workflow}
        return cls(
            gabor=GaborConfig.from_mapping(merged_gabor),
            analysis=AnalysisConfig.from_mapping(merged_analysis, workflow=workflow),
            workflow=workflow,
        )

    @classmethod
    def from_json(cls, path: Path, workflow: str = WORKFLOW_2P) -> "PipelineConfig":
        """Function for from json.

        Args:
            path: Input value for this operation.
            workflow: Input value for this operation.

        Returns:
            Result produced by the operation.
        """
        import os
        path = Path(path)

        with path.open("r", encoding="utf-8") as handle:
            raw_text = handle.read()

        actual_root = os.getcwd().replace("\\", "/")
        resolved_text = raw_text.replace("{PROJECT_ROOT}", actual_root)
        payload = json.loads(resolved_text)
        workflow = payload.get("workflow", workflow)
        if workflow not in (WORKFLOW_2P, WORKFLOW_EPHYS):
            raise ValueError(
                f"Unknown workflow {workflow!r}; expected {WORKFLOW_2P!r} or {WORKFLOW_EPHYS!r}"
            )

        gabor_params = payload.get("gabor") or payload.get("gabor_param", {})
        common_params = payload.get("common", {})
        two_photon_params = (
            payload.get("two_photon")
            or payload.get("2p")
            or payload.get("two_p")
            or {}
        )
        ephys_params = payload.get("ephys") or {}

        # Legacy flat ``param_defaults`` / ``analysis`` sections.
        legacy_analysis = payload.get("analysis") or payload.get("param_defaults", {})
        if legacy_analysis:
            common_params = {**legacy_analysis, **common_params}
            two_photon_params = {**legacy_analysis, **two_photon_params}
            ephys_params = {**legacy_analysis, **ephys_params}

        workflow_params = (
            two_photon_params if workflow == WORKFLOW_2P else ephys_params
        )
        analysis_params = {**common_params, **workflow_params}

        merged_gabor = dict(DEFAULT_GABOR_PARAMS)
        merged_gabor.update(gabor_params)

        merged_common = dict(DEFAULT_COMMON_PARAMS)
        merged_common.update(common_params)
        for canonical_key in ("NX", "NY", "Sigmas", "Frequencies"):
            if canonical_key not in common_params and canonical_key in gabor_params:
                merged_common[canonical_key] = gabor_params[canonical_key]
        if "Library Path" not in common_params:
            library_folder = gabor_params.get("Fine Library Path") or gabor_params.get("Save Path")
            if library_folder not in NONE_STRINGS and library_folder is not None:
                merged_common["Library Path"] = library_folder

        merged_workflow = dict(
            DEFAULT_TWO_PHOTON_PARAMS
            if workflow == WORKFLOW_2P
            else DEFAULT_EPHYS_PARAMS
        )
        merged_workflow.update(workflow_params)

        merged_analysis = {**merged_common, **merged_workflow}
        return cls(
            gabor=GaborConfig.from_mapping(merged_gabor),
            analysis=AnalysisConfig.from_mapping(merged_analysis, workflow=workflow),
            workflow=workflow,
        )


def default_pipeline_config(workflow: str = WORKFLOW_2P) -> PipelineConfig:
    """Return the default example configuration."""
    return PipelineConfig.from_mappings(workflow=workflow)
