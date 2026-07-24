"""Regression coverage for one-indexed neural-cache trial exclusions."""

import numpy as np
import pytest

from waven.config import PipelineConfig
from waven.time_alignment import exclude_identified_trials


def test_excluded_trial_numbers_round_trip_through_configuration():
    config = PipelineConfig.from_mappings(
        analysis_params={"Excluded Trial Numbers": "[2, 5, 2]"},
    ).analysis

    assert config.excluded_trial_numbers == (2, 5)
    assert config.to_gui_mapping()["Excluded Trial Numbers"] == "[2, 5]"


@pytest.mark.parametrize("value", ["[0]", "[-1]", "[1.5]", "[True]"])
def test_excluded_trial_numbers_must_be_positive_whole_numbers(value):
    with pytest.raises(ValueError, match="Excluded Trial Numbers"):
        PipelineConfig.from_mappings(analysis_params={"Excluded Trial Numbers": value})


def test_exclude_identified_trials_keeps_matches_and_reports_out_of_range(capsys):
    spikes = np.arange(4 * 2 * 1).reshape(4, 2, 1)

    filtered = exclude_identified_trials(spikes, [2, 6, 4])

    np.testing.assert_array_equal(filtered, spikes[[0, 2]])
    message = capsys.readouterr().out
    assert "maximum identified trial is 4" in message
    assert "[6]" in message
    assert "[2, 4]" in message
