# Methods transparency

The GUI workflow is staged:

1. Select two-photon or electrophysiology data.
2. Build or reuse Gabor filter libraries.
3. Decompose the stimulus into Gabor wavelet arrays.
4. Align neural data or load existing `spikes.npy` and `pos.npy`.
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

## Quality mask

The GUI computes repeatability and skewness. Neurons with repeatability below
the current threshold or very high skewness are shown with lower alpha rather
than silently removed.

## Caching and recovery

Long-running buttons reuse completed outputs when shapes match the current
configuration. Recovery checkpoints are removed after successful or cancelled
runs and kept after failures for debugging.
