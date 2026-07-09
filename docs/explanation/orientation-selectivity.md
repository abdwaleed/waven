# Orientation selectivity

`waven` computes two orientation selectivity summaries from each neuron's
orientation tuning curve.

The input tuning curve is one value per orientation bin:

```text
orientation_tuning.shape == (n_orientations,)
```

For RF-derived tuning, the curve is taken from the RF tensor at the neuron's
preferred position, size, and frequency:

```text
RF[neuron, preferred_x, preferred_y, :, preferred_sigma, preferred_frequency]
```

Because these values are correlations, they can be negative. Before computing
ratios, `waven` replaces non-finite values with zero and shifts a curve upward
when its minimum is below zero. This preserves relative orientation differences
while avoiding negative denominators in OSI/gOSI.

## OSI

The orientation selectivity index compares the preferred response with the
orthogonal response:

```text
OSI = (R_pref - R_orth) / (R_pref + R_orth)
```

The preferred orientation is the orientation with the largest tuning value. The
orthogonal orientation is 90 degrees away, modulo 180 degrees.

Interpretation:

| Value | Meaning |
| --- | --- |
| near `0` | Preferred and orthogonal responses are similar. |
| near `1` | Preferred response is much larger than the orthogonal response. |
| `NaN` | The denominator was zero after cleaning, usually because the curve had no usable signal. |

Classic OSI is easy to explain but depends on two bins: the maximum and the
nearest orthogonal bin. It can be sensitive to noisy orientation samples.

## gOSI

The global OSI uses the orientation vector sum:

```text
gOSI = abs(sum(R(theta) * exp(2j * theta))) / sum(R(theta))
```

The factor of 2 makes orientation circular over 180 degrees rather than 360
degrees.

Interpretation:

| Value | Meaning |
| --- | --- |
| near `0` | Responses are broadly distributed or balanced across orientations. |
| near `1` | Responses concentrate strongly around one orientation axis. |
| `NaN` | The cleaned tuning curve sums to zero. |

gOSI uses all orientation bins, so it is often more stable for population
histograms. It can still be biased by noisy curves or weak neurons, so compare
it with repeatability and the plotted tuning curve.

## In the GUI

The selected-neuron tuning plot shows OSI and gOSI in the orientation panel
title. The **All neurons** tab shows:

- OSI/gOSI histograms over neurons with per-shank overlays when grouping
  metadata can be inferred;
- OSI/gOSI histograms by unit, or by spatial bins when explicit unit metadata is
  unavailable.

Histograms are descriptive summaries. The neuron/shank view asks whether
selectivity differs across anatomical recording groups. The unit view asks
whether a larger grouping, which can contain many shanks or spatial bins, has a
different selectivity distribution. A group with very few neurons should be read
as an annotation, not as a stable population estimate.
