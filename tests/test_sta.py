import numpy as np

from waven.analysis.sta import compute_sta, fit_phase_gabor, phase_gabor_image
from waven.storage.neural_cache import (
    load_spike_counts_cache,
    save_spike_counts_cache,
)


def test_sta_uses_raw_frame_counts_with_requested_lag():
    movie = np.array(
        [
            [[0.0, 1.0], [2.0, 3.0]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=np.float32,
    )
    counts = np.zeros((1, 5, 1), dtype=np.int32)
    counts[0, 1:, 0] = np.array([1, 0, 2, 1], dtype=np.int32)

    result = compute_sta(
        movie,
        counts,
        fps=10.0,
        max_lag_ms=100.0,
        n_shuffles=2,
        scale_stimulus=False,
        random_seed=1,
    )

    centered_movie = movie - movie.mean()
    expected = (centered_movie[:4] * counts[0, 1:, 0, None, None]).sum(axis=0) / 4.0
    np.testing.assert_allclose(result.images[1, 0], expected, rtol=1e-6, atol=1e-6)
    assert result.total_spikes[1, 0] == 4
    assert result.images.shape == (2, 1, 2, 2)


def test_phase_gabor_fit_keeps_signed_phase_structure():
    parameters = np.array(
        [7.0, 8.0, 4.0, 3.0, 0.45, 0.18, 1.2, -0.7, 0.05],
        dtype=np.float32,
    )
    image = phase_gabor_image(parameters, 17, 17)
    fitted, rmse = fit_phase_gabor(image, max_nfev=800)
    reconstruction = phase_gabor_image(fitted, 17, 17)

    assert np.all(np.isfinite(fitted))
    assert np.isfinite(rmse)
    assert np.corrcoef(image.ravel(), reconstruction.ravel())[0, 1] > 0.95


def test_spike_count_cache_is_kept_separate_from_firing_rates(tmp_path):
    counts = np.arange(24, dtype=np.int32).reshape(2, 4, 3)
    saved = save_spike_counts_cache(counts, tmp_path, "npy")
    loaded, path = load_spike_counts_cache(tmp_path, mmap_mode=None)

    assert saved == path
    np.testing.assert_array_equal(loaded, counts)
