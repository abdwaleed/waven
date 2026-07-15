# PSTH-weighted spike-triggered averaging

Coarse RF Analysis computes a standard spike-triggered average (STA) whenever
you inspect an individual neuron. It uses the neuron's trial-averaged,
frame-aligned firing-rate trace (the PSTH) and the prepared coarse stimulus
movie. There is no separate STA action, raw-count cache, shuffle test, or
Gabor-fitting stage.

## Inputs

| Input | Shape | Purpose |
| --- | --- | --- |
| aligned `spikes` | `(trials, frames, neurons)` | Coarse RF averages the trial dimension to obtain the selected neuron's PSTH |
| coarse stimulus cache | `(frames, analysis_y, analysis_x)` | Frame-for-frame stimulus movie used for STA maps |
| movie FPS | scalar | Converts the fixed 300 ms analysis window into complete frame lags |

The movie and PSTH must have the same frame alignment. Coarse RF limits the
shared analysis length to the shorter available input and never uses a lag that
would exceed 300 ms. For example, a 30 Hz movie produces lags 0 through 9;
the next lag would be 333.3 ms and is excluded.

## Calculation

For a lag `L`, the map is the standard rate-weighted stimulus average:

```text
STA(L) = sum_t PSTH[t] * stimulus[t - L] / sum_t PSTH[t]
```

Only response frames `t >= L` contribute. The implementation flattens each
stimulus frame and uses a vectorized matrix product for every lag, rather than
looping over time points in Python. It computes the pixel-intensity variance of
each 2-D map and marks the lag with the highest variance.

## Display and export

The **Individual neuron** tab shows the existing Coarse RF plots followed by a
PSTH-weighted STA grid. Every map from 0 to 300 ms is displayed, with the
highest-variance lag marked in the title. The figure payload contains:

- `sta_maps`;
- `sta_lag_frames` and `sta_lag_ms`;
- `sta_variances`;
- `sta_peak_lag_frame`, `sta_peak_lag_ms`, and `sta_peak_variance`.

Select **PSTH-weighted STA lag maps** in the individual-neuron export
checkboxes to export the current figure or the separate STA map graphs for
every analyzed neuron.

::: waven.analysis.psth_sta
