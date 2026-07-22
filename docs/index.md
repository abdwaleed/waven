# waven documentation

`waven` analyzes neural responses to visual stimuli with Gabor-wavelet
features. The GUI is organized around one principle: stimulus metadata comes
first and provides the dimensions, frame count, timing, and duration used by
every downstream stage.

## Recommended GUI path

1. **Stimulus & Metadata** — select the movie, choose a downsampling
   percentage, and prepare the stimulus cache.
2. **Session Setup** — create or validate the aligned neural `spikes`/`pos`
   cache. Ephys `spikes` values are frame-bin firing rates in Hz.
3. **Gabor** — create legacy libraries or convolution kernel caches.
4. **Wavelet Products** — make only the named Zarr product required by Coarse
   RF, Run Model, or Run Full Model.
5. **Analysis** — run Coarse RF, then use **Inspect Single Neuron**. Run
   Model curves are added automatically when their coarse phase pair is ready;
   Run Full Model remains an opt-in inspection refinement.

## Core data model

| Object | Shape | Meaning |
| --- | --- | --- |
| aligned neural responses | `(trials, frames, neurons)` | Ephys frame-bin firing rate in Hz; other workflows store aligned non-negative activity. |
| downsampled movie | `(frames, y, x)` | Binary stimulus image sequence. |
| Coarse RF power | `(frames, x, y, orientations, sigmas)` | Zarr product used only for receptive-field correlation. |
| Run Model real/imaginary phases | `(frames, x, y, orientations, sigmas)` each | Separate Zarr pair for the coarse model. |
| Run Full Model real/imaginary phases | `(frames, x, y, orientations, full_sigmas, frequencies)` each | Separate Zarr pair for local refinement. |
| RF tensor | `(neurons, x, y, orientations, sigmas, frequencies)` | Pearson-correlation RF diagnostic tensor. |

Movie arrays use image order `(y, x)` while Gabor/RF arrays use feature order
`(x, y)`. Width and height always originate from movie metadata; no manual
`NX`, `NY`, stimulus FPS, or duration is used by the GUI.

## Reading selectivity

RF maps and individual tuning displays use correlation to make the preferred
feature understandable. OSI/gOSI use the separate firing-rate tuning extracted
from the aligned neural cache—not correlation amplitudes. See
[Orientation Selectivity](explanation/orientation-selectivity.md).

## Documentation map

- [First GUI Analysis](tutorials/first-gui-analysis.md)
- [Prepare Wavelet Products](how-to/run-wavelet-decomposition.md)
- [Interpret Analysis Graphs](how-to/interpret-analysis-graphs.md)
- [GUI Reference](reference/gui.md)
- [Data Contracts and Scientific Interpretation](reference/data-contracts.md)
- [Source Architecture and Ownership](reference/source-architecture.md)
- [Project Layout](reference/project-layout.md)
- [Pipeline Intuition](explanation/pipeline-intuition.md)
