# Interpret the analysis graphs

This guide explains every graph produced by **Run Coarse RF Analysis** and
**Inspect Single Neuron**. It is a reading guide, not a substitute for a
stimulus calibration or statistical validation. First identify the y-axis
quantity: most RF graphs show a **Pearson correlation** between a Gabor feature
and neural response, whereas OSI/gOSI and the red orientation curve use the
neural **firing-rate/activity** response itself.

## What a preferred Gabor feature means

For each neuron, Coarse RF searches a Gabor bank over position, orientation,
size, and, when enabled, spatial frequency. Its *preferred feature* is the
discrete bank entry with the largest correlation to that neuron's
training-trial-average response. It is not a fitted continuous parameter, a
significance test, or necessarily the largest firing rate.

| Factor | What it changes |
| --- | --- |
| aligned neural cache and training trials | The response time series for RF correlation and trial summaries. Ephys input is frame-bin firing rate (Hz); other workflows use aligned activity in arbitrary units. |
| stimulus movie, downsampling, and **Analysis Coverage** | The pixels, visual-degree coordinates, and frames supplied to the Gabor features. A changed crop or movie can change every RF map. |
| `N_thetas` | The number of discrete orientation bins from 0 degrees inclusive to 180 degrees exclusive. More bins improve sampling; they do not create extra information. |
| Gabor sigmas and frequencies | Candidate envelope sizes and spatial frequencies. Sigma is converted from analysis pixels to visual degrees; frequency is cycles per visual degree. |
| phase offsets and prepared wavelet/cache product | The features used for correlation and, for model plots, the real/imaginary phase signals. |
| response quality | Only changes dot transparency in population retinotopy maps. A dot is opaque when repeatability is at least 0.2 and skewness is at most 20; otherwise it is faint. It does not recolor or remove the neuron from the main distributions. |

## Orientation convention used by Waven

Waven uses *orientation*, not directional motion: an angle is axial, so 0 and
180 degrees describe the same orientation. Raw Gabor labels in the GUI and
exports are `theta_waven = k * 180 / N_thetas`, for `k = 0 ... N_thetas - 1`.

The angle specifies the direction in which the sinusoidal Gabor carrier varies
(the normal to the visible bars), in displayed-image coordinates:

| Raw Waven angle | Carrier/normal points | Visible bars run approximately |
| --- | --- | --- |
| 0 degrees | right | vertical |
| 45 degrees | down and right | down and left |
| 90 degrees | down | horizontal |

The GUI's raw 0-degree label is therefore **not** the conventional “arrow
down, horizontal bars” 0-degree convention in the supplied example. If a
traditional tuning curve defines 0 degrees as down and positive 45 degrees as
down-left, relabel the *angle axis* with:

`theta_traditional = (theta_waven - 90) mod 180`

or, in the other direction:

`theta_waven = (theta_traditional + 90) mod 180`.

This is a 90-degree relabeling, not a reversal of the tuning values. For an
even number of equally spaced bins, it is a circular shift by half the bins.
Confirm the mapping once with a known calibration grating before comparing
datasets. Positive versus negative **drift** below is a temporal phase
direction, not another orientation convention.

## Individual Neuron graphs

Choose a point/neuron and click **Inspect Single Neuron**. Its standard output
is a black spike/activity trace, the multi-panel **Selected Neuron Tuning**
dashboard, and PSTH-weighted STA maps. If the phase cache exists, the button
also adds Run Model curves; the **Run Full Model** checkbox adds their
full-resolution refinement.

### Trial-averaged Spike Train

This black line is the selected neuron's response at every stimulus-aligned
frame, averaged over trials.

| Axis/color | Meaning |
| --- | --- |
| x: frame | Prepared/aligned movie frame index, not elapsed seconds. Divide by movie FPS for seconds. |
| y | Ephys input is frame-bin firing rate in Hz. Other input is aligned activity in arbitrary units. The compact plot's generic `Activity (a.u.)` label does not distinguish the workflows. |
| black line | One selected neuron; black has no value encoding. |

Use this graph to find transients, trial-average modulation, or alignment
problems. It is not a trial raster and does not identify a preferred feature by
itself.

### Selected Neuron Tuning dashboard

| Panel | What is plotted | How to read it | Important limits |
| --- | --- | --- | --- |
| **Receptive Field** | Two-dimensional Pearson-correlation slice at this neuron's preferred orientation, size, and frequency. Axes are azimuth/elevation in the configured analysis coverage. | `coolwarm` is centered at zero for this map: warm/red is positive feature-response correlation, cool/blue is negative, and pale/white is near zero. | Limits are symmetric but independently scaled to this neuron's largest absolute correlation. Compare color within a map, not between neurons. Correlation is association, not a causal excitatory/inhibitory map. |
| **Elevation** and **Azimuth** | Signed leading spatial profiles of that preferred RF map; a fallback uses a spatial mean. | Blue line is RF correlation `r` as position changes. | These are summaries of the displayed RF slice, not separately fitted tuning experiments. The SVD sign is chosen for readability, so its sign is not absolute response polarity. |
| **Orientation correlation** | Correlation while orientation changes and the selected position, size, and frequency stay fixed. | Blue points/line are Pearson `r`; light-blue bars are 95% confidence intervals across trials when available. The 180-degree point repeats 0 degrees to close the axial curve. | This is a feature-correlation curve. Its title shows firing-rate OSI/gOSI for reference, but those indices are not calculated from the blue curve. Use the raw Waven angles above. |
| **Orientation tuning from firing rate** | Trial-weighted neural response at each orientation with the same preferred non-orientation Gabor feature fixed. | Red points/line are mean firing rate (Hz for ephys) or activity (a.u.); pale-red bars are SEM across trials. | This is the curve used for headline OSI/gOSI. It uses the same raw Waven labels and repeated 180-degree endpoint. |
| **Size (deg)** | Correlation as Gabor sigma changes at preferred position, orientation, and frequency. | Purple points/line are Pearson `r`; pale-purple bars are 95% confidence intervals when available. | “Size” is Gaussian-envelope standard deviation (`sigma`) in visual degrees, not stimulus diameter or measured biological RF diameter. It samples the configured discrete sigma list. |
| **Spatial Frequency** | Correlation as carrier frequency changes at preferred position, orientation, and size. | Black points/line are Pearson `r`; black has no additional category meaning. x is cycles/degree. | It appears only when Coarse RF has an independent frequency axis with more than one value. More cycles/degree means finer light-dark spacing, not faster motion. |
| **Tuning statistics** (two text panels) | One panel describes the correlation orientation curve and the other firing-rate orientation curve, plus cached unit metadata. | No color encoding. Each lists preferred discrete bin/degree, max, min, mean “baseline,” modulation index, available trials, OSI/gOSI, position, and available unit/shank metadata. | Preferred means highest sampled bin—there is no peak fitting or interpolation. The firing-rate panel is the primary OSI/gOSI result. |

#### OSI, gOSI, and error bars

Both selectivity measures range roughly from 0 (little orientation preference)
to 1 (strong preference) when response values are non-negative and
well-behaved.

- **OSI** compares the largest sampled firing-rate response with the bin nearest
  its orthogonal orientation: `(R_pref - R_orth) / (R_pref + R_orth)`.
- **gOSI** is the magnitude of the response-weighted circular vector using
  doubled angles, divided by summed response. Doubling makes angles 180 degrees
  apart equivalent.
- **SEM** describes uncertainty of the trial mean. A **95% CI** is wider and
  is used on the correlation and size panels. Neither is a multiple-comparison
  correction or an RF significance test.

Correlation can be negative while firing rate/activity should be read on its
native scale. Treat correlation-derived OSI/gOSI in the first statistics panel
as a diagnostic; population summaries and headline selectivity use the
firing-rate/activity curve.

### PSTH-weighted Spike-Triggered Averages (0–300 ms)

Each panel is a prepared stimulus image preceding the selected neuron's
trial-averaged response by the labeled lag. For every lag, Waven computes:

`STA(lag) = sum(PSTH[t] * stimulus[t - lag]) / sum(PSTH[t])`.

The map asks which stimulus values tended to precede high activity; it is
separate from the Gabor-correlation RF.

| Mark/color | Meaning |
| --- | --- |
| panel title `STA lag ... ms` | 0 ms pairs response and same stimulus frame; larger positive lags look farther back. Lags are limited to 300 ms. |
| `peak variance` note | Lag whose STA image has greatest across-pixel variance. It is a contrast/structure heuristic, not a p-value, latency estimate, or largest firing rate. |
| `coolwarm` colors | All lag maps for one neuron share one local minimum-to-maximum scale: cool/blue is lower rate-weighted stimulus value and warm/red is higher. There is currently no drawn colorbar; exported `sta_maps` provide the numerical scale. |

The prepared movie is binary and the weighting is non-negative. STA colors are
therefore not signed correlations and should not be called excitation (red)
versus inhibition (blue). Compare colors across lags for one neuron, not across
neurons without exported numbers.

### Run Model and Run Full Model: amplitude, phase, and drift

These are direct 20-bin marginal curves of an occupancy-normalized modeled
response surface, not extra raw neural tuning curves. The vertical label
`Firing rate / response (a.u.)` means model response/gain; compare bins within
a curve, not its height as a universally calibrated spike amplitude.

| Graph/color | x-axis quantity | Meaning and cautions |
| --- | --- | --- |
| **Amplitude tuning** (blue) | `rho = sqrt(real^2 + imaginary^2)` | Quadrature Gabor-response magnitude (Gabor energy) at the selected feature. It is not physical image luminance/contrast or neural spike-voltage amplitude. Its scale depends on movie preprocessing and the filter. |
| **Phase tuning** (purple) | `phi = atan2(imaginary, real)`, radians from 0 to `2*pi` | Location/alignment within a spatial Gabor carrier cycle. A phase shift can mean a light/dark-pattern shift; it is not an orientation change. Phase is unwrapped internally before drift but displayed over one 0-to-`2*pi` cycle. |
| **Drift tuning** (orange) | Temporal phase derivative in cycles/second (Hz) | `diff(phi) * FPS / (2*pi)`. Gray vertical line marks zero; positive/negative values are opposite temporal phase-progression directions. Hz means phase cycles per second, not firing rate and not automatically physical stimulus speed. Speed also requires spatial frequency and stimulus calibration. |

Color here is only graph identity: blue amplitude, purple phase, orange drift.
It does not represent a feature category or confidence value. **Run Model** uses
the compact coarse real/imaginary phase pair and is automatic when available.
**Run Full Model** uses the larger full-resolution phase bank with full sigma
and frequency axes, so it is an optional refinement. Definitions and colors are
the same for both; the sampled feature data differ.

## All Neurons graphs

The **All Neurons** area reports layout, preferred-feature maps, and
orientation-selectivity distributions. It does not show model curves for every
neuron unless those are explicitly selected in the every-neuron export.

| Graph | What each mark represents | Colors and factors that control them |
| --- | --- | --- |
| **Neuron Positions** | One cell/unit at acquisition or imaging position, in micrometres. The display is 3-D when a third coordinate exists; forced 2-D drops one coordinate for display. Two-photon y positions are converted to the GUI display convention. | All dots are semitransparent black. Black encodes no feature, quality, firing rate, or group; overlaps look darker. |
| **Population Retinotopy — Azimuth** | Same positions, colored by visual azimuth of each neuron's highest-correlation Gabor feature. | `jet`. Read this panel's own colorbar: its automatic limits depend on finite preferred azimuths in this analysis. Faint dots failed the repeatability/skewness mask, but their hue still represents azimuth. |
| **Population Retinotopy — Elevation** | Positions colored by preferred visual elevation. | Reversed `jet_r`. Never infer higher elevation from color without its colorbar. Limits are independently autoscaled. |
| **Population Retinotopy — Orientation** | Positions colored by raw Waven preferred Gabor orientation bin. | Cyclic `hsv`: first and last hues meet because 0 and 180 degrees are equivalent. Hue is around-a-circle, not a low-to-high brightness scale. Use the colorbar and the 90-degree conversion above for traditional curves. |
| **Population Retinotopy — Size** | Positions colored by preferred Gabor sigma in degrees. | `coolwarm`. This is selected Gabor envelope size, not biological RF diameter. Its colorbar is autoscaled from finite preferred sizes. |
| **OSI and gOSI distribution** | Histograms of one firing-rate/activity-derived OSI or gOSI per finite neuron. | Filled blue bars: all finite neurons. Red: KDE density scaled to counts. Green solid: mean. Purple dashed: median. Black outline: quality-mask neurons. Colors identify summaries/subsets, not selectivity magnitude. |
| **Ephys shank outlines** | Quality-mask neurons in an inferred shank group, when available. | Thin categorical outlines are shank IDs; colors can be reused with many groups. They do not encode OSI, depth, waveform, or a continuous value. Group source is reported in the figure caption. |
| **OSI and gOSI by Unit** (ephys only) | Separate histograms for up to ten valid multi-neuron groups. | Each row has one categorical color for its OSI and gOSI histograms. Quality-mask finite values are shown unless none pass. Repeated acquisition IDs form groups; otherwise grouping is derived from position. Row color is a label, not a score. |

Every retinotopy hue comes from the **maximum-correlation feature** for that
neuron. It can change with movie, coverage, downsampling, Gabor
orientations/sigmas/frequencies, phase bank, usable frames/training trials, and
neural alignment. It is not set by OSI/gOSI, spike count, shank, or quality
transparency. The four panels use different colormaps and independently
autoscaled colorbars: compare numeric values on colorbars, not color appearance
between panels, sessions, or separate analyses.

## A safe interpretation workflow

1. Confirm movie, coverage, Gabor settings, and aligned neural cache match the
   session being interpreted.
2. Read the units and decide whether a graph is correlation, firing
   rate/activity, model response, or rate-weighted stimulus average.
3. For traditional orientation curves, relabel Waven angles by minus 90 degrees
   modulo 180; leave the curve values unchanged.
4. Read every population-map hue against that panel's colorbar. Check faint
   dots against the quality mask rather than treating them as absent.
5. Use trial error bars, quality overlays, exports, and suitable experimental
   statistics before claiming selectivity, latency, or RF structure.

See [Run RF Analysis](run-rf-analysis.md) for prerequisites and
[Export results](export-results.md) for arrays and manifests that preserve the
exact values behind a graph.
