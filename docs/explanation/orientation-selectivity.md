# Orientation selectivity

`waven` computes two orientation selectivity summaries from each neuron's
orientation tuning curve.

## OSI

The orientation selectivity index compares the preferred response with the
orthogonal response:

```text
OSI = (R_pref - R_orth) / (R_pref + R_orth)
```

The preferred orientation is the orientation with the largest tuning value. The
orthogonal orientation is 90 degrees away, modulo 180 degrees.

## gOSI

The global OSI uses the orientation vector sum:

```text
gOSI = abs(sum(R(theta) * exp(2j * theta))) / sum(R(theta))
```

The factor of 2 makes orientation circular over 180 degrees rather than 360
degrees.

## In the GUI

The selected-neuron tuning plot shows OSI and gOSI in the orientation panel
title. The **All neurons** tab shows:

- OSI/gOSI histograms over neurons with per-shank overlays when grouping
  metadata can be inferred;
- OSI/gOSI histograms by unit, or by spatial bins when explicit unit metadata is
  unavailable.
