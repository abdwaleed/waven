# Scientific intuition

Waven asks a structured question: *which localized visual pattern, at which place and time, best explains a neuron's aligned response?* It does not infer that answer from raw movie pixels alone. Instead it describes every movie frame with a bank of convolutional Gabor features and aligns those features to the same frame clock as the neural data.

## From visual angle to an analysis grid

The movie has source pixels, but receptive fields are interpreted in visual space. **Visual Coverage** says which visual degrees the movie spans; **Analysis Coverage** selects the region tested. Waven resamples that crop onto an analysis grid and reports degrees per pixel and a Nyquist spatial-frequency limit.

Choose the grid from the signal you need to preserve:

- To distinguish fine gratings, use enough pixels that the reported Nyquist limit is comfortably above the highest cycles/degree of interest.
- To study broad, low-frequency structure, a coarser grid is appropriate and substantially reduces disk and convolution cost.
- The percentage slider is a compatibility representation of that choice; the degrees/pixel and retained-cpd modes make the decision physically interpretable.

## Why Gabor features

A Gabor filter is a local sinusoidal pattern inside a Gaussian envelope. Its parameters map to common visual hypotheses:

| Parameter | Question it asks |
| --- | --- |
| Position `(x, y)` | Where in the analysed field is the feature relevant? |
| Orientation | Which edge/grating direction is preferred? |
| Sigma | Over what spatial extent should the feature be pooled? |
| Frequency | How fine is the grating pattern? |
| Phase | Where does the light/dark cycle fall inside the envelope? |

The convolution kernel cache stores those filters compactly. Applying them to the binary stimulus movie produces features over time without repeatedly building the filters.

## Coarse RF first

Coarse RF uses the phase-insensitive power product. It correlates each neuron's aligned response with candidate local feature time courses and identifies a preferred location/orientation/size (and frequency, IF you choose to make an independent frequency list in the `Gabor Parameters` step). Coarse RF is a screening step: it narrows a large visual feature space to a plausible seed for inspection and optional nonlinear modelling.

Correlation maps and RF tuning curves are therefore *diagnostics of covariation*, not firing-rate units. A strong correlation says that a feature's variation over the movie tracks the response; it does not alone establish causality.

## Orientation selectivity is a different quantity

Waven also reports orientation tuning, OSI, and gOSI from aligned neural activity at the selected feature. These measure response selectivity from the neural response/firing-rate cache, not from correlation values. Keeping the two quantities separate avoids conflating a feature-search statistic with an activity-based selectivity statistic.

## Why the two optional models exist

**Run Model** uses a compact coarse real/imaginary phase pair. It starts from the Coarse RF seed and adds amplitude, phase, and drift tuning graphs when the extra detail is useful.

**Run Full Model** uses the larger full-resolution real/imaginary phase pair, including its full sigma and frequency axes. It supports a more detailed local refinement, so it has the largest optional cache and computation cost.

Neither model is required to inspect a neuron. The default Inspect Single Neuron view remains useful for quality control and scientific exploration without either phase bank.

## Time alignment matters as much as filtering

Every feature time course must refer to the same frames as `spikes`. Fresh 2-photon/ephys processing creates an aligned neural cache; existing caches are validated against the prepared movie. This is why changing the movie, crop, or grid invalidates downstream reuse: a visually plausible result with a mismatched time axis is not a scientifically valid result.
