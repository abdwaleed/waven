"""Regression coverage for cache/model handoffs that previously failed late."""

from __future__ import annotations

import numpy as np
import pytest

from waven.analysis import model_runs
from waven.analysis.receptive_fields import PearsonCorrelationPinkNoise


def test_run_model_converts_scalar_correlation_metrics(monkeypatch):
    """The public Run Model payload must not contain a nested correlation matrix."""

    def fake_single_neuron(*_args, **_kwargs):
        return (
            np.array([0.0, 1.0]),
            np.array([1.0]),
            np.array([2.0]),
            [0.1, 0.2, 0.3, 0.4, np.nan],
            object(),
        )

    monkeypatch.setattr(model_runs, "_process_single_neuron", fake_single_neuron)
    phase = np.zeros((2, 1, 1, 1, 1), dtype=np.float32)
    spikes = np.zeros((2, 2, 1), dtype=np.float32)
    best = np.zeros((4, 1), dtype=int)

    _prediction, _nonlinear, _rho_phi, metrics, _interpolators = model_runs.run_Model(
        best,
        best,
        spikes,
        phase,
        phase,
        dt1=2,
        n_min=1,
        train_idx=[0],
        test_idx=[1],
        frames_per_minute=1,
    )

    assert metrics.shape == (1, 3)
    np.testing.assert_allclose(metrics[0], [0.1, 0.2, 0.3])


def test_rf_preferred_feature_honours_absolute_flag():
    """Signed RF selection must not silently choose the strongest negative feature."""

    generator = np.random.default_rng(7)
    response = generator.normal(size=(128, 1)).astype(np.float32)
    negative = -response[:, 0]
    positive = 0.35 * response[:, 0] + generator.normal(scale=1.0, size=128)
    stimulus = np.stack((negative, positive), axis=1).reshape(128, 1, 1, 2, 1)

    signed = PearsonCorrelationPinkNoise(
        stimulus,
        response,
        np.zeros((1, 2)),
        nx=1,
        ny=1,
        ns=1,
        nf=1,
        visual_coverage=[1, -1, 1, -1],
        screen_ratio=1,
        sigmas=np.array([1.0]),
        frequencies=np.array([1.0]),
        n_orientations=2,
        absolute=False,
    )
    absolute = PearsonCorrelationPinkNoise(
        stimulus,
        response,
        np.zeros((1, 2)),
        nx=1,
        ny=1,
        ns=1,
        nf=1,
        visual_coverage=[1, -1, 1, -1],
        screen_ratio=1,
        sigmas=np.array([1.0]),
        frequencies=np.array([1.0]),
        n_orientations=2,
        absolute=True,
    )

    assert int(signed[1][2][0]) == 1
    assert int(absolute[1][2][0]) == 0


def test_direct_coarse_bundle_writes_power_and_phase_pair(tmp_path):
    """One convolution pass must leave Run Model-ready quadrature caches."""

    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    pytest.importorskip("torch")
    from waven.storage.array_store import load_array
    from waven.wavelets.decomposition import waveletPowerDecompositionConv

    movie = np.random.default_rng(5).normal(size=(3, 7, 7)).astype(np.float32)
    waveletPowerDecompositionConv(
        movie,
        sigmas=[1.0],
        folder_path=str(tmp_path),
        n_orientations=1,
        phase_offsets=[0.0, np.pi / 2],
        output_stem="power",
        phase_output_stems=("real", "imag"),
        frame_chunk_size=2,
        filter_group_size=1,
    )

    power = np.asarray(load_array(tmp_path / "power.zarr", mmap_mode="r"))
    real = np.asarray(load_array(tmp_path / "real.zarr", mmap_mode="r"))
    imaginary = np.asarray(load_array(tmp_path / "imag.zarr", mmap_mode="r"))
    assert power.shape == real.shape == imaginary.shape == (3, 7, 7, 1, 1)
    np.testing.assert_allclose(power, real**2 + imaginary**2, rtol=1e-5, atol=1e-6)


def test_direct_coarse_bundle_selects_coupled_phase_from_independent_power(tmp_path):
    """Run Model phase caches remain compact while Coarse RF sweeps frequency."""

    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    pytest.importorskip("torch")
    from waven.storage.array_store import load_array
    from waven.wavelets.decomposition import waveletPowerDecompositionConv

    movie = np.random.default_rng(6).normal(size=(3, 7, 7)).astype(np.float32)
    frequencies = [0.5, 0.25, 0.125]
    coupled = [0.5, 0.25]
    waveletPowerDecompositionConv(
        movie,
        sigmas=[1.0, 2.0],
        folder_path=str(tmp_path),
        n_orientations=1,
        phase_offsets=[0.0, np.pi / 2],
        output_stem="power_independent",
        phase_output_stems=("real_coupled", "imag_coupled"),
        frequencies=frequencies,
        phase_coupled_frequencies=coupled,
        frame_chunk_size=2,
        filter_group_size=1,
    )

    power = np.asarray(load_array(tmp_path / "power_independent.zarr", mmap_mode="r"))
    real = np.asarray(load_array(tmp_path / "real_coupled.zarr", mmap_mode="r"))
    imaginary = np.asarray(load_array(tmp_path / "imag_coupled.zarr", mmap_mode="r"))
    assert power.shape == (3, 7, 7, 1, 2, 3)
    assert real.shape == imaginary.shape == (3, 7, 7, 1, 2)
    for sigma_index, frequency_index in enumerate((0, 1)):
        np.testing.assert_allclose(
            power[..., sigma_index, frequency_index],
            real[..., sigma_index] ** 2 + imaginary[..., sigma_index] ** 2,
            rtol=1e-5,
            atol=1e-6,
        )
