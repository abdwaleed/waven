# Data and cache contracts

This reference defines what each stage accepts and produces. Shapes are part of the contract: a cache is safe to reuse only when its time axis, spatial grid, and feature axes match the current experiment settings.

## Primary inputs

| Input | Type | Required? | Contract | Scientific role |
| --- | --- | --- | --- | --- |
| Stimulus movie | one supported movie in a folder | Yes | Metadata supplies `frames`, `width`, `height`, and FPS. | Defines the visual sequence and clock. |
| Visual Coverage | 4 floats `[left, right, top, bottom]` in degrees | Yes | Full movie field. | Associates source pixels with visual angles. |
| Analysis Coverage | 4 floats in the same order | Yes | Must lie inside Visual Coverage. | Selects visual field analysed by every downstream feature. |
| Aligned `spikes` | numeric array `(trials, frames, neurons)` | Yes before Coarse RF | Frame axis must match the prepared movie. Values must be non-negative for STA. | Trial-resolved neural response; ephys is frame-bin firing rate in Hz. |
| Aligned `pos` | numeric position array matching neurons | Yes before Coarse RF | One position per neural channel/unit. | Population layout and retinotopy display. |
| Gabor axes | orientation count, sigma list, frequency list, phase list | Yes before matching cache product | Finite, valid values on the derived grid. | Hypothesis space of local visual features. |

## Durable products

| Product | Type / shape | Produced by | Required by | Description |
| --- | --- | --- | --- | --- |
| Stimulus cache | binary NPY/Zarr `(frames, y, x)` | Prepare Stimulus Cache | wavelets, STA | Cropped/resampled common movie. |
| Coarse RF power | float32 Zarr `(frames, x, y, orientations, sigmas)` | Prepare Coarse RF Cache | Coarse RF analysis | Phase-insensitive local feature power for screening RF correlations. |
| Coarse model real/imaginary pair | two float32 Zarr arrays `(frames, x, y, orientations, sigmas)` | Prepare Run Model Phase Caches | optional Run Model | Phase-aware coarse features. |
| Full model real/imaginary pair | two float32 Zarr arrays `(frames, x, y, orientations, sigmas, frequencies)` | Prepare Run Full Model Phase Caches | optional Full Model | Larger phase-aware feature bank for refinement. |
| RF correlation tensor | numeric `(neurons, x, y, orientations, sigmas, frequencies)` where applicable | Run Coarse RF Analysis | inspection and population plots | Correlation diagnostic used to identify preferred features. |
| PSTH-weighted STA | float32 maps `(lags, y, x)` for one inspected neuron | Inspect Single Neuron | selected-neuron view/export | Direct response-weighted preceding-frame average. |

Waven uses image coordinate order `(y, x)` for movies and displayed images, but feature/RF order `(x, y)` after the time axis. Do not transpose one merely to match the other without checking the consumer's contract.

## What the analysis means

**Coarse RF** compares the aligned neural response against the Coarse RF power features. Its maps and tuning curves are correlation diagnostics: they say how strongly a response varies with a particular local feature over the stimulus.

**Orientation selectivity** is separate. OSI/gOSI and firing-rate orientation tuning are computed from aligned neural activity at the preferred feature; they are not correlation amplitudes. This avoids interpreting a correlation scale as a firing-rate selectivity scale.

**Run Model** and **Run Full Model** use phase-aware real/imaginary feature pairs and the Coarse RF result as a seed. They add amplitude, phase, and drift graphs for a selected neuron. Run Model uses the compact coarse phase bank; Full Model searches/refines using the larger full sigma/frequency bank.

## Cache validity

Changing any of the following requires a new matching downstream cache:

- movie source or its metadata;
- Visual Coverage or Analysis Coverage;
- sampling/grid setting;
- Gabor orientation, sigma, frequency, or phase settings;
- the requested product (Coarse RF power, Run Model pair, or Full Model pair);
- neural cache whose frame dimension no longer matches the movie.

Waven records provenance and validates compatibility before reuse. It may read old compatible paths as a convenience, but the active GUI workflow produces convolutional cache products only.

## Export contract

Exports are derived from already calculated figures/results. PNG and SVG are visual representations. PKL is an optional Python-specific record of graph data and analysis values. Exporting does not change the source cache format or create NPY/Zarr graph exports. See [Export Results](../../how-to/export-results.md).
