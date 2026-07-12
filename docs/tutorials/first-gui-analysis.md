# First GUI analysis

This tutorial follows the GUI in its required order. The essential idea is
simple: the stimulus movie defines the timing and spatial grid for the whole
session, so it is prepared before neural alignment or Gabor work.

## 1. Configure folders

Set `Project Root`, `Movie Path`, neural-data paths, and output folders in
`pipeline_config.json` or the GUI. `Movie Path` is a folder containing one
stimulus movie. The project layout is documented in
[Project Layout](../reference/project-layout.md).

Do not enter stimulus width, height, frame count, or FPS. They are read from
the movie. Do not enter `NX` or `NY`; one downsampling percentage determines
the shared analysis grid.

## 2. Stimulus & Metadata

Open **1 Stimulus & Metadata**, select the movie folder, choose the percentage,
and click **Prepare Stimulus Cache**. This reads and validates:

- original width and height;
- frame count;
- frames per second;
- duration (`frames / FPS`).

It then writes the binary downsampled movie with shape `(time, y, x)`. The GUI
uses this metadata-derived frame count for neural-cache alignment and the
duration to validate ephys photodiode trial boundaries. This is why **Session
Setup** is intentionally unavailable for neural cache creation until this step
succeeds.

## 3. Session Setup

Open **2 Session Setup** and choose the neural source.

- **Fresh / raw data** builds an aligned `spikes`/`pos` cache. For ephys,
  `spikes` is frame-bin firing rate in Hz: spike count divided by the actual
  photodiode-defined bin duration.
- **Continue / existing cache** validates an existing compatible cache pair.

The resulting neural array has shape `(trials, frames, neurons)`.

## 4. Gabor and wavelet products

In **3 Gabor**, select Legacy or Convolution.

- Legacy creates coarse and fine flattened Gabor libraries.
- Convolution creates compact coarse and fine kernel caches instead.

In **4 Wavelet Products**, prepare only the products you need:

| Action | Used by | Product |
| --- | --- | --- |
| Prepare Coarse RF Power Cache | Run Coarse RF Analysis | `coarse_rf_power.zarr` |
| Prepare Run Model Phase Caches | Run Model | `coarse_model_real.zarr`, `coarse_model_imag.zarr` |
| Prepare Run Full Model Phase Caches | Run Full Model | `dwt_videodata2_r.zarr`, `dwt_videodata2_i.zarr` |

All large internal products are chunked Zarr arrays. They are streamed from
disk rather than eagerly loaded into memory.

## 5. Analyze

Click **Run Coarse RF Analysis**. RF maps and displayed orientation/size curves
are correlation-based diagnostics. The OSI and gOSI distributions are different:
they use firing-rate orientation tuning from the aligned `spikes` cache, weighted
by the preferred wavelet feature. Individual orientation and size correlation
curves show 95% confidence intervals only.

After RF analysis, **Run Model (Coarse RF)** requires the named coarse-model
phase pair. **Run Full Model** requires the named full-model phase pair and
uses the coarse RF feature seeds for local refinement.

## Resuming safely

Every cache is validated by shape and a parameter fingerprint. Re-running an
action reuses a compatible completed product and regenerates a stale one. If a
run fails, read the reported artifact name and shape; do not substitute a cache
made with a different movie, percentage, backend, orientation count, sigma
list, phase list, or frequency list.
