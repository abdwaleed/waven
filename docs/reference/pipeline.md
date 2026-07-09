# Pipeline

`waven.pipeline` is the recommended scripting surface. It connects typed config
objects to the lower-level wavelet, alignment, RF, and model modules without
requiring a user to call historical helper functions directly.

## Main flow

| Function | Input | Output | Size intuition |
| --- | --- | --- | --- |
| `create_coarse_gabor_library()` | `GaborConfig` | `.npy` library path | scales roughly with `(coarse_nx * coarse_ny)^2` |
| `create_fine_gabor_library()` | `GaborConfig` plus optional extra sigmas | `.npy` library path | scales roughly with `(NX * NY)^2 * n_frequencies` |
| `prepare_stimulus_wavelets()` | `AnalysisConfig`, coarse library | coarse wavelet folder | writes boolean movie plus real/imag coarse phases |
| `prepare_full_model_wavelets()` | `AnalysisConfig`, `GaborConfig`, fine library | full wavelet folder | writes one dense `float32` tensor per phase |
| `load_spikes_and_positions()` | `AnalysisConfig` | `SpikeData` | returns `(n_trials, n_frames, n_neurons)` spikes |
| `run_rf_analysis()` | config plus `SpikeData` | `RFAnalysisResult` | RF tensor has one feature map per neuron |

## Dataclass outputs

| Dataclass | Fields | Meaning |
| --- | --- | --- |
| `SpikeData` | `spikes`, `aligned_spikes`, `neuron_pos` | neural responses and anatomical coordinates |
| `WaveletData` | `wavelets_r`, `wavelets_i`, `wavelets_complex` | coarse wavelet phases and combined cache |
| `RFAnalysisResult` | `repeatability`, `rf_results`, `wavelets` | quality metrics plus RF tensor payload |
| `SimpleModelResult` | parameters, predictions, metrics, interpolators | fast nonlinear model outputs |

The pipeline layer truncates RF analysis to the shortest available time axis
among config frame count, wavelet frames, and spike frames. Small mismatches are
therefore tolerated, but large mismatches usually indicate an alignment or cache
selection problem.

::: waven.pipeline
