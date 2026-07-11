# Analysis

Analysis modules transform wavelet features and neural responses into RF maps,
orientation selectivity summaries, trial-quality metrics, and nonlinear model
fits.

## RF and tuning shapes

| Object | Shape | Interpretation |
| --- | --- | --- |
| response matrix | `(n_frames, n_neurons)` | trial-averaged neural activity |
| flattened stimulus | `(n_frames, n_features)` | wavelet tensor reshaped for correlation |
| RF tensor | `(n_neurons, coarse_nx, coarse_ny, n_orientations, n_sigmas, n_frequencies)` | Pearson correlation per neuron-feature pair |
| RF orientation tuning | `(n_orientations,)` | correlation slice at preferred position, sigma, and frequency |
| OSI/gOSI arrays | `(n_neurons,)` | selectivity summaries for population plots |

RF values are correlations, so they can be negative. In the GUI, the correlation
orientation slice at each neuron's preferred feature is the input to OSI/gOSI.

## Which module to read

| Module | Role |
| --- | --- |
| `orientation_selectivity` | OSI/gOSI from orientation tuning curves or RF results |
| `receptive_fields` | Pearson RF search, retinotopy, prediction helpers |
| `nonlinear_models` | lower-level model functions |
| `model_runs` | orchestration for simple and full nonlinear models |
| `trial_stats` | repeatability, skewness, and response-quality helpers |

::: waven.analysis.orientation_selectivity

::: waven.analysis.receptive_fields

::: waven.analysis.nonlinear_models

::: waven.analysis.model_runs

::: waven.analysis.trial_stats
