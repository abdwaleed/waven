# Spike-triggered averaging (STA)

STA estimates a unit's visual receptive field by asking a direct question:
which stimulus patterns tend to occur immediately before that unit emits a
spike? It is an ephys-only, convolution-backend action in **Run Coarse RF
Analysis**. It does not alter Coarse RF correlation, OSI/gOSI, model runs, or
the normal firing-rate `spikes` cache.

## When to use it

Use STA when the stimulus is a repeated, time-varying movie and raw ephys spike
events are available. It is particularly useful for simple-cell-like fields:
the STA can retain alternating positive and negative lobes, rather than
collapsing them into phase-invariant energy.

STA is intentionally unavailable for two-photon data. Those responses are
calcium-derived and are not raw event counts in the ephys frame-bin sense used
by this method. It is also intentionally unavailable in the legacy backend.

## Inputs and caches

The normal ephys alignment still writes the firing-rate cache used by the rest
of waven:

| Artifact | Shape | Units / purpose |
| --- | --- | --- |
| `spikes.npy` or `.zarr` | `(trials, frames, neurons)` | Hz; existing RF, OSI/gOSI, and model inputs |
| `spike_counts.npy` or `.zarr` | `(trials, frames, neurons)` | integer count per photodiode-derived frame bin; STA input |
| prepared coarse stimulus cache | `(frames, analysis_y, analysis_x)` | disk-backed binary/downsampled movie; STA stimulus |

For a frame with three events, `spike_counts[..., frame, unit]` is exactly
`3`. The corresponding `spikes` value remains
`3 / frame_duration_seconds`. Keeping both values prevents the STA
calculation from silently changing the scientific meaning of existing analyses.

STA uses the already prepared coarse stimulus cache. NPY caches are opened as
read-only memmaps; Zarr caches remain lazy. The movie is scanned in chunks to
get a temporal mean for every stimulus pixel, then every working chunk has
that pixel-wise baseline removed. This prevents fixed screen structure—such as
a top-left display-sync/photodiode patch—from appearing as a red or blue
square in every STA lag. By default a safe global bound scales the centred
working movie to approximately `[-1, 1]` without constructing a second full
movie in RAM.

## Core computation

Let `M[t, p]` be the flattened stimulus frame after subtracting pixel `p`'s
temporal mean,
and let `C[trial, t, unit]` be a raw spike count. For a lag `L`, waven first
sums counts across repeated trials at each matching frame:

```python
frame_counts = spike_counts.sum(axis=0)  # (frames, neurons)
stimulus_slice = movie[0 : frames - L]   # (valid_frames, pixels)
counts_slice = frame_counts[L : frames]  # (valid_frames, neurons)
raw_sta = stimulus_slice.T @ counts_slice
sta = raw_sta / counts_slice.sum(axis=0)
```

The stored result is image ordered as:

```text
sta_images: (lags, neurons, analysis_y, analysis_x)
```

The implementation performs the matrix product in bounded frame chunks. It
uses CUDA when available and the existing **Use all available GPUs** setting
is enabled: lag jobs are distributed across available GPUs. A failed CUDA
chunk falls back to CPU instead of failing the complete cache action.

Trials are not concatenated across their boundaries. Summing same-index frame
counts is equivalent for a repeated movie while preventing a lag from treating
the end of one presentation as the beginning of the next.

## Lag, shuffle, and display controls

The **Neural & RF Analysis** panel exposes these STA settings when the ephys
workflow and convolution backend are selected. Neural alignment still creates
the raw-count cache; STA is deliberately a separate action so users can tune
the lag and shuffle settings without recreating their neural cache.

| Control | Default | Meaning |
| --- | --- | --- |
| Max lag (ms) | 150 | largest backward-looking stimulus offset; only whole stimulus frames at or below this time are used |
| Shuffle count | 100 | number of circular null draws per lag |
| Threshold (SD) | 3 | required real-STA peak in units of shuffled pixel SD; the GUI allows 3–5 |
| Random seed | 0 | reproducible circular-shift sequence; change to sample another valid null set |
| STA cache format | NPY | persistent STA arrays: NPY is directly memory-mappable; compressed Zarr can reduce disk use and supports chunked reads |

For every null draw, the count vector is circularly shifted by a random 3–5
second offset (or a non-zero available frame offset for a very short movie).
The operation is equivalent to `np.roll(counts, shift)` but is indexed in
chunks so it does not create a full copied count tensor for each shuffle.

For the normal streamed path, several circular shifts are concatenated into a
bounded response batch and evaluated by one matrix multiplication. This avoids
rereading the same stimulus chunk for every shuffle. The experimental
**FFT STA actual maps** runtime option uses `torch.fft` only for an in-memory,
CUDA, many-lag actual-STA calculation; it automatically retains the exact
batched matrix-product implementation for the shuffle null and falls back to
streaming when its bounded FFT conditions are not met.

For each neuron and lag, the standard deviation is calculated across every
pixel from every shuffled STA image. A real STA passes only when:

```text
max(abs(real_sta_pixels)) >= threshold_sd * shuffled_pixel_sd
```

Images without spikes, with zero null variance, or below that threshold are
marked **unresponsive / no RF found**. They remain in `sta_images` for
inspection but never reach Gabor fitting.

## Phase-sensitive Gabor fitting

The STA fitter is independent of waven's existing power/squared-sum Gabor
path. It fits the signed image with a Gaussian envelope and both carriers:

```text
prediction = baseline + envelope * (a_cos * cos(carrier) + a_sin * sin(carrier))
```

This form explicitly estimates spatial phase. It can represent the bright and
dark stripe pattern of a simple-cell STA; a phase-invariant power fit cannot.
Only images that pass the circular shuffle test are fitted. The cached
`sta_phase_gabor_params.npy` last axis is:

```text
[x0_px, y0_px, sigma_x_px, sigma_y_px, theta_rad,
 frequency_cycles_per_px, cosine_amplitude, sine_amplitude, baseline,
 amplitude, phase_rad]
```

`sta_phase_gabor_rmse.npy` contains one fit RMSE per `(lag, neuron)`. A
non-converged significant image has NaN parameters and is reported as such in
the GUI; an unresponsive image has no fit by design.

## Output and display

The action writes a `sta/` folder inside the selected neural cache folder. The
**STA cache format** radio buttons select either `.npy` or compressed `.zarr`
for every numeric artifact below; the selection is saved with the GUI state and
is also used by **Export All STA Data and Lag Graphs**. Choose NPY when simple
portable files or OS-level memory mapping are most useful. Choose Zarr when
storage pressure matters or chunked compressed access is preferable.

| Artifact | Shape |
| --- | --- |
| `sta_images.{npy,zarr}` | `(lags, neurons, analysis_y, analysis_x)` |
| `sta_lag_frames.{npy,zarr}`, `sta_lag_ms.{npy,zarr}` | `(lags,)` |
| `sta_noise_std.{npy,zarr}`, `sta_peak_values.{npy,zarr}`, `sta_significant.{npy,zarr}`, `sta_total_spikes.{npy,zarr}` | `(lags, neurons)` |
| `sta_phase_gabor_params.{npy,zarr}` | `(lags, neurons, 11)` |
| `sta_phase_gabor_rmse.{npy,zarr}` | `(lags, neurons)` |
| `sta_metadata.json` | movie centring/scaling, shuffle, threshold, and parameter-schema metadata |

The right-side **STA Receptive Fields** tab contains a nested tab for every
computed lag. Each shows the selected neuron's signed STA beside its
phase-sensitive Gabor fit, or a clear explanation that the shuffle test
rejected the fit. STA has its own **STA Neuron Index** field and **Display STA
Neuron** button; it never inherits the normal coarse-RF Neuron ID. On
completion, the GUI displays a prompt rather than an arbitrary neuron, unless
the STA field already contains a valid index.

The Export panel includes **Export All STA Data and Lag Graphs**. It writes the
full numeric STA tensors and metadata once, then writes the signed STA/Gabor
figure bundle for every neuron and lag. **Export Every Individual Graph Type
for Every Neuron** additionally writes the normal coarse-RF individual plots
and, when available, STA lag plots into separate folders. For coarse RF it
also writes a standalone image and JSON data description for each visible
panel, including the firing-rate orientation-tuning curve.

## Programmatic API

```python
from waven.analysis.sta import compute_sta, fit_phase_gabor, phase_gabor_image

result = compute_sta(
    disk_backed_movie,      # (frames, y, x)
    raw_spike_counts,       # (trials, frames, neurons)
    fps=30.0,
    max_lag_ms=150,
    n_shuffles=100,
    significance_sd=3,
)
```

`compute_sta` accepts a NumPy array, NumPy memmap, or Zarr-style array-like
movie. It validates exact movie/count frame agreement before calculation.

::: waven.analysis.sta
