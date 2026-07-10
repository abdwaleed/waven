# Methods transparency

The GUI workflow is staged:

1. Select two-photon or electrophysiology data.
2. Build or reuse Gabor filter libraries.
3. Decompose the stimulus into Gabor wavelet arrays.
4. Align neural data or load existing `spikes` and `pos` caches in `.npy` or `.zarr` format.
5. Run coarse receptive-field analysis.
6. Inspect population maps and individual-neuron tuning.
7. Run simple or full nonlinear models.
8. Export figures and reusable data.

## Coarse RF search

The coarse RF step reshapes the wavelet cache into a `(time, features)` matrix
and compares it with trial-averaged neural responses. For each neuron, the best
feature is selected by maximum absolute Pearson correlation.

The selected feature indices are converted into interpretable values:

- azimuth;
- elevation;
- orientation;
- size;
- spatial frequency.

The main arrays are:

| Array | Shape | Description |
| --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | Responses after two-photon or electrophysiology alignment. |
| trial average | `(n_frames, n_neurons)` | Mean response used for Pearson correlation. |
| coarse wavelet cache | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | Combined stimulus feature time courses. |
| flattened stimulus | `(n_frames, n_features)` | Temporary 2D view used for correlation. |
| RF tensor | `(n_neurons, coarse_nx, coarse_ny, n_orientations, n_sigmas, n_frequencies)` | Correlation value for each neuron-feature pair. |

Pearson correlation removes the mean and normalizes by variance, so it is a
shape-of-time-course comparison rather than a raw-amplitude comparison. This is
helpful when neurons have different response scales, but it also means a neuron
with weak activity can still obtain an apparently strong coefficient. That is
why repeatability, skewness, and visual inspection remain part of the workflow.

## Quality mask

The GUI computes repeatability and skewness. Neurons with repeatability below
the current threshold or very high skewness are shown with lower alpha rather
than silently removed.

The quality mask is intentionally visual. It supports exploratory screening
without changing the underlying arrays exported for later analysis. When a plot
shows a faint neuron, the data point is still present; the styling is telling
you to interpret it with less confidence.

## Full-model wavelets

The full-model path keeps the configured `NX` by `NY` grid and writes dense
phase-specific tensors:

```text
(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)
```

These arrays are the largest objects in most projects. They are useful because
the nonlinear model can look beyond the single best coarse feature, but they are
not necessary for an initial RF screen.

## Orientation selectivity

After RF analysis, the orientation tuning curve for a neuron is the vector:

```text
RF[neuron, preferred_x, preferred_y, :, preferred_sigma, preferred_frequency]
```

`waven` reports both OSI and gOSI. OSI compares the preferred orientation to the
nearest orthogonal orientation. gOSI uses all orientation bins as a vector sum,
so it is less dependent on a single orthogonal bin. Because RF tuning curves are
correlations and can be negative, the implementation shifts each curve to a
nonnegative baseline before computing ratios.

## Caching and recovery

Long-running buttons reuse completed outputs when shapes match the current
configuration. Recovery checkpoints are removed after successful or cancelled
runs and kept after failures for debugging.

Shape validation is the main safety rule. If `NX`, `NY`, `N_thetas`, `Sigmas`,
or `Frequencies` change, a previous cache can have the right filename but the
wrong scientific meaning. The GUI tries to reject wrong-shaped artifacts and
resume from the last completed compatible step.
