# Pipeline intuition

The core idea is to express a visual stimulus in a feature space that resembles
localized oriented filters. Each Gabor filter asks: "Was there structure at this
location, orientation, size, phase, and frequency at this time?"

The workflow then compares those feature time courses to neural activity. A
single neuron is not asked to explain the raw movie pixel by pixel. Instead, it
is compared with many candidate visual features: a dark/light edge here, at this
angle, with this spatial scale, changing over the same stimulus frames as the
recorded response.

1. **Gabor libraries** define candidate visual features.
2. **Wavelet decomposition** projects the movie onto those features over time.
3. **Coarse RF analysis** finds the feature that best correlates with each
   neuron.
4. **Tuning curves** slice the RF tensor around the preferred feature.
5. **OSI/gOSI** summarize orientation tuning strength.
6. **Simple and full models** fit nonlinear relationships between selected
   wavelet variables and neural responses.

The coarse path is designed to be fast enough for screening. The full model is
more expensive but searches at higher spatial and feature resolution.

## Why Gabor filters?

Gabor filters are localized sinusoids under a Gaussian envelope. They are useful
for visual neuroscience because they provide a compact vocabulary for oriented
contrast: location, orientation, size, phase, and spatial frequency. That
vocabulary is not a claim that neurons are literally Gabors; it is a practical
coordinate system for asking whether neural activity follows oriented visual
structure in the stimulus.

`waven` stores two phases for each filter, real and imaginary. The phases are
90 degrees apart, so together they reduce sensitivity to whether a feature is
bright-on-dark or dark-on-bright at exactly one pixel alignment. The coarse
cache also stores a combined coefficient derived from the two phases, which is
the array used by the coarse RF search.

## Coarse versus full paths

The coarse path reduces the metadata-derived analysis grid before correlation.
The user chooses one downsampling percentage; movie width and height determine
the shared grid, and the coarse grid is derived from it. There are no GUI
entries for `NX`, `NY`, movie FPS, or a fixed movie duration. This stage is
meant to answer:

- where is the receptive field approximately?
- which orientation and size look plausible?
- which neurons are reliable enough to inspect?

The full path keeps the configured grid and allows the model to use independent
spatial frequencies. This is much larger because the output grows with every
axis:

```text
n_frames * full_x * full_y * n_orientations * n_sigmas * n_frequencies
```

That value is per phase. Real and imaginary full-model outputs are separate
files, so the storage requirement is effectively doubled when both are kept as
plain `.npy`.

## What correlations mean

The coarse RF tensor stores Pearson correlations between each wavelet feature
time course and each neuron's trial-averaged response. Positive values mean the
neuron tends to be active when that feature is strong. Negative values mean the
neuron tends to be less active when that feature is strong. The preferred
feature is selected by maximum absolute correlation so strongly negative
relationships are still visible instead of being discarded.

The individual RF orientation curve is a correlation slice through the RF
tensor at the preferred position, size, and frequency. It is useful for
understanding why that feature was selected, but it is **not** used for OSI or
gOSI. Those metrics first use the RF search to choose the preferred feature
location, then form an orientation-by-orientation weighted mean from the
aligned firing-rate array. This keeps the selectivity measurement in neural
response units rather than correlation units.

For each orientation \(o\), Waven computes a non-negative weighted mean
firing rate \(R_o\):

```text
R_o = sum_t(power[t, preferred_x, preferred_y, o, preferred_size] * rate[t])
      / sum_t(power[t, preferred_x, preferred_y, o, preferred_size])
```

OSI compares the preferred orientation with its orthogonal orientation; gOSI
is the magnitude of the doubled-angle vector sum of the same rates. Both range
from zero (no orientation preference) toward one (strong preference). They are
descriptive statistics, not a replacement for repeatability, signal quality,
or model performance.
