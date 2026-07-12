# Orientation selectivity

The GUI calculates OSI and gOSI from the neuron's **frame-aligned firing
rate**, never from RF correlation values. For ephys, the `spikes` cache is made
by counting spikes in every photodiode-defined movie-frame bin and dividing by
the actual bin duration in seconds. Thus its values are firing rates in Hz.

Correlation still has one important role: coarse RF analysis uses it to choose
each neuron's preferred position, size, and spatial-frequency feature. Once
those feature indices are known, the selectivity calculation takes the
non-negative wavelet energy at that feature as an orientation-specific frame
weight and computes:

```text
rate_tuning[orientation] =
  sum(frame_weight * firing_rate) / sum(frame_weight)
```

This asks: *at frames containing each orientation at the neuron's preferred RF
feature, how strongly did the neuron fire?* The correlation tensor is not used
as `R(theta)` in either index.

## OSI

The orientation selectivity index compares the firing rate at the preferred
orientation with that at the orientation nearest 90 degrees away:

```text
OSI = (R_pref - R_orth) / (R_pref + R_orth)
```

An OSI near zero means similar firing at preferred and orthogonal orientations;
an OSI near one means a strongly preferred orientation.

## gOSI

Global OSI uses all firing-rate orientation bins:

```text
gOSI = abs(sum(R(theta) * exp(2j * theta))) / sum(R(theta))
```

The doubled angle treats 0 and 180 degrees as the same orientation axis. A
larger gOSI means the firing-rate response is more concentrated around one
orientation axis.

## Population displays

The **All neurons** view renders separate OSI and gOSI panels with a KDE, mean,
median, and legend. Its descriptive table lists total units, mean, median,
mode, standard deviation, variance, range, quartiles, skewness, kurtosis, and
high/medium/low selectivity categories. Counts use finite firing-rate-derived
indices in the plotted population.

## Individual-neuron curves and uncertainty

The individual-neuron **Orientation Tuning Curve** and **Size Tuning Curve**
remain correlation curves for inspecting the RF fit. They alone show 95% CI
bars, computed from trial-wise correlations as
`1.96 * sample_std / sqrt(number_of_valid_trials)`. Those correlation curves
and their CIs do not determine OSI/gOSI: the title and exported OSI/gOSI values
come from the separate firing-rate tuning calculation above.
