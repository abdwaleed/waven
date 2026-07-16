import numpy as np

from waven.runtime.performance import two_photon_io_worker_count
from waven.suite2p.utils.utils import interp_event_responses


def test_vectorized_event_interpolation_matches_linear_traces():
    """All event windows retain the historical event/cell/window ordering."""
    timestamps = np.arange(8, dtype=float)
    traces = np.vstack((timestamps, 2.0 * timestamps + 1.0))
    events = np.array((1.0, 4.0))
    window = np.array((0.0, 0.5, 1.0))

    actual = interp_event_responses(timestamps, traces, events, window=window)

    expected = np.empty((events.size, traces.shape[0], window.size))
    for event_index, event in enumerate(events):
        for cell_index, trace in enumerate(traces):
            expected[event_index, cell_index] = np.interp(
                event + window, timestamps, trace, left=np.nan, right=np.nan
            )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_vectorized_event_interpolation_mean_keeps_a_window_axis():
    timestamps = np.arange(8, dtype=float)
    traces = np.vstack((timestamps, timestamps ** 2))
    actual = interp_event_responses(
        timestamps, traces, np.array((2.0, 3.0)), window=np.array((0.0, 1.0)), mean_over_window=True
    )
    assert actual.shape == (2, 2, 1)


def test_suite2p_plane_workers_are_bounded():
    assert two_photon_io_worker_count(1) == 1
    assert 1 <= two_photon_io_worker_count(100) <= 4
