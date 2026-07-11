# Orientation selectivity

`waven` measures OSI and gOSI from the **correlation orientation tuning curve**
at each neuron's best receptive-field location, size, and frequency. This is an
important distinction: the GUI no longer derives selectivity from a separate
firing-rate tuning curve.

The coarse RF analysis first searches a correlation tensor for each neuron’s
preferred position, filter size, frequency, and orientation. It then holds the
preferred non-orientation features fixed and reads the correlation values over
all orientation bins:

```text
correlation_tuning[orientation] =
  RF[neuron, best_x, best_y, orientation, best_sigma, best_frequency]
```

In plain language, this asks: *when the stimulus is represented by this
neuron's best spatial feature, how strongly does the neural response correlate
with each orientation?* Using the same correlation space for RF discovery,
tuning, OSI, and gOSI makes the result internally consistent.

## OSI

The orientation selectivity index compares the preferred orientation with the
orientation nearest 90 degrees away:

```text
OSI = (R_pref - R_orth) / (R_pref + R_orth)
```

An OSI near zero indicates similar correlation at preferred and orthogonal
orientations. An OSI near one indicates a much stronger preferred-orientation
correlation. Because classic OSI depends on two bins, it is easy to interpret
but can be sensitive to noise in either bin.

## gOSI

Global OSI uses the full curve rather than only two bins:

```text
gOSI = abs(sum(R(theta) * exp(2j * theta))) / sum(R(theta))
```

The doubled angle makes 0 and 180 degrees represent the same orientation axis.
gOSI near zero means correlations are broad or balanced across orientations;
gOSI near one means they concentrate around one axis. It is often a steadier
population summary because every orientation bin contributes.

## GUI distributions and table of contents

The **All neurons** view renders separate OSI and gOSI panels. Each includes a
blue histogram, red Gaussian KDE, green mean line, purple dashed median line,
and a legend. The table printed beneath each plot contains:

```text
OSI/gOSI DISTRIBUTION STATISTICS
Sample Size: Total units
Central Tendency: Mean, Median, Mode
Spread: Std Dev, Variance, Range
Distribution: Min, Q1 (25%), Q2 (50%), Q3 (75%), Max
Shape: Skewness, Kurtosis
Selectivity Categories: High (>0.5), Medium (0.3–0.5), Low (<0.3)
```

Counts and percentages are calculated from finite values in the plotted
population. The quality-mask overlay is still available where a mask exists.
For ephys grouping, unique acquisition IDs are not treated as population
groups—doing so creates a misleading `n=1` panel for every neuron. The GUI
instead falls back to a shared spatial/unit grouping when IDs are unique.

These summaries are descriptive, not a significance test. Always inspect the
correlation tuning curve and the number of units supporting a group before
interpreting small between-group differences.
