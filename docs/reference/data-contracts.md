# Data contracts and scientific interpretation

This is the canonical input/output guide for the current pipeline. It explains
both what each stage accepts and why that representation is used scientifically.

## Coordinate and array conventions

| Convention | Meaning |
| --- | --- |
| image/movie space | `(frames, y, x)`; rows (`y`) increase downward, columns (`x`) increase rightward |
| feature/RF space | `(frames, x, y, orientation, sigma[, frequency])`; this is the order used by Gabor and RF tensors |
| neural cache | `(trials, frames, neurons)`; every trial is aligned to the same stimulus-frame clock |
| positions | `(neurons, 2)` for x/y or `(neurons, 3)` when a depth-like third coordinate exists |
| frame time | one movie frame; FPS comes from the movie, never a manual GUI field |
| ephys response unit | firing rate in Hz for each frame bin |
| two-photon response unit | the aligned activity supplied by that workflow |

The different `(y, x)` and `(x, y)` orders are intentional. Images follow
image-array convention; feature tensors follow the Gabor-grid convention.

## Inputs

| Input | Accepted type | Required contents | Scientific purpose |
| --- | --- | --- | --- |
| stimulus movie | folder containing exactly one supported video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, `.m4v`) | decodable frames, positive FPS, width, height, and frame count | establishes the visual time base and stimulus geometry |
| downsampling percentage | GUI float, **1–100** | percentage of movie width and height | chooses analysis resolution; 100 preserves pixel dimensions, lower values trade spatial detail for compute/storage |
| visual coverage | four numeric values `(x_max, x_min, y_max, y_min)` in degrees | non-zero field of view | describes the whole displayed stimulus field |
| analysis coverage | same four-value degree tuple | lies wholly inside visual coverage | selects the visual subfield that is scientifically analysed |
| raw neural data | workflow-specific folder | Suite2p/two-photon or ephys files and timing metadata | supplies responses and positions before common frame alignment |
| existing neural cache | folder containing `spikes` and `pos` as NPY or Zarr | shape-compatible response and position pair | resumes after acquisition/alignment without rereading raw data |
| Gabor axes | numeric lists/integers: orientations, sigmas, phases, frequencies | positive/valid feature values; phases entered in degrees | defines localized visual features tested against neural responses |
| runtime controls | booleans/integers/environment-backed GUI values | optional | changes scheduling only; never changes the scientific definition of a result |

## Stage-by-stage outputs

| Stage | Output type and shape | Description | Intuition |
| --- | --- | --- | --- |
| movie metadata | mapping: `width`, `height`, `frames`, `fps`, `duration` | values read from the selected video | makes all later arrays refer to the same physical movie and clock |
| downsampled movie | `bool`, `(frames, analysis_y, analysis_x)`, NPY or Zarr | cropped binary stimulus, then anti-aliased/resized and thresholded | a compact, common visual representation for every later feature calculation |
| aligned neural cache | numeric `spikes`, `(trials, frames, neurons)`; numeric `pos`, `(neurons, 2|3)` | each response trial placed on the movie-frame clock | allows comparison of each neuron to the exact stimulus frame seen at each time |
| Gabor library/kernel cache | float filters or compact convolution kernels | localized position/orientation/size/phase/frequency filters | asks whether a neuron covaries with a specific local visual pattern |
| Coarse RF power | `float32` Zarr, `(frames, x, y, orientation, sigma)` | phase-insensitive Gabor power | appropriate first-pass RF search when phase is not the quantity of interest |
| coarse model phases | real and imaginary `float32` Zarr arrays, `(frames, x, y, orientation, sigma)` | phase-resolved coarse features | retains phase for the nonlinear coarse model |
| full model phases | real and imaginary `float32` Zarr arrays, `(frames, x, y, orientation, full_sigma, frequency)` | denser, frequency-resolved feature cache | supports local nonlinear refinement around preferred coarse features |
| RF tensor | numeric, `(neurons, x, y, orientation, sigma, frequency)` | Pearson correlation between each neural response and each feature time course | a screening map of feature-response covariation, not a firing-rate map |
| selectivity | firing-rate tuning `(neurons, orientations)`, OSI/gOSI `(neurons,)` | non-negative response summary at each neuron's preferred RF feature | separates true response preference from signed RF correlation |
| PSTH-weighted STA | `float32` maps `(lags, y, x)` or `(neurons, lags, y, x)` for a bounded batch | rate-weighted mean of preceding binary stimulus frames | reveals the average stimulus pattern preceding stronger neural responses |
| model result | dataclass/arrays of fitted parameters, predictions, metrics, interpolators | simple or full nonlinear model fit | tests whether selected visual features predict response dynamics beyond the RF screen |

## Coarse RF reasoning

For each feature time series \(s_f(t)\) and trial-averaged neural response
\(r_n(t)\), Coarse RF estimates Pearson correlation. The preferred index is
the feature with the strongest selected correlation according to the analysis
policy. This makes the search comparable across feature amplitudes, but a
correlation can be positive or negative and is not a response rate.

The individual-neuron RF map displays that selected correlation slice. Its
azimuth and elevation profiles use a signed SVD projection of the map, which
summarizes a spatial pattern more robustly than selecting one noisy pixel.

## Orientation selectivity reasoning

After RF search identifies a preferred location/size/frequency, Waven computes
orientation tuning from the aligned neural responses at that preferred feature.
OSI and gOSI therefore summarize non-negative response modulation across
orientation. They must not be interpreted as a transformation of the RF
correlation tensor.

## STA reasoning

For a lag \(\ell\), a trial-averaged PSTH \(r(t)\), and binary stimulus frame
\(S(t)\), the standard map is:

\[
\operatorname{STA}_{\ell} =
\frac{\sum_{t=\ell}^{T-1} r(t) S(t-\ell)}
     {\sum_{t=\ell}^{T-1} r(t)}.
\]

Only non-negative, finite PSTH values are accepted. Lags are constrained to
the requested maximum and to 300 ms based on the actual movie FPS. Waven
reports pixel variance for each 2-D lag map and marks the largest-variance map
as a descriptive peak; it is not a statistical significance test.

## Cache validity and reproducibility

Reusable artifacts are checked against expected shape and, where supported, a
`.waven.json` sidecar with parameter fingerprints. A cache is regenerated when
movie-derived dimensions, crop coverage, output format, or relevant Gabor
parameters no longer match. This prevents a scientifically invalid mixture of
responses, stimulus geometry, and feature axes.

## Export contract

Every graph export contains a rendered PNG and SVG plus a JSON manifest. The
array selection adds NPY, Zarr, or both; a pickle retains richer payload data
when possible. Section B exports exactly one graph per folder, so an RF map,
azimuth profile, orientation curve, and each STA lag map remain independently
usable rather than becoming inseparable panels of a dashboard.
