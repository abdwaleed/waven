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
| RF orientation tuning | `(n_orientations,)` | correlation slice at preferred position, size, and frequency; used for RF inspection |
| firing-rate orientation tuning | `(n_neurons, n_orientations)` | weighted mean firing rate at each neuron's preferred RF location/scale |
| OSI/gOSI arrays | `(n_neurons,)` | selectivity summaries computed from firing-rate tuning, never RF correlations |

RF values are correlations, so they can be negative. They answer *which visual
feature time course covaries with this neuron?* OSI/gOSI answer a different
question: *how differently does the neuron's firing rate respond across
orientations?* The GUI therefore derives the preferred location/size/frequency
from the RF tensor, then uses the aligned `spikes` cache to compute a
non-negative firing-rate tuning curve. Ephys values are frame-bin Hz; two-photon
values are the aligned activity used by that workflow.

The Individual Neuron tab deliberately keeps these measurements separate. Its
RF map, orientation, size, and frequency panels use the selected correlation
tensor features. Azimuth and elevation use the established signed SVD
projection of the selected RF map, which summarizes the spatial pattern without
allowing one noisy feature cell to define a profile. Its orientation title
reports OSI/gOSI from the separate firing-rate calculation. This avoids falsely
presenting a correlation coefficient as a firing rate.

## Which module to read

| Module | Role |
| --- | --- |
| `orientation_selectivity` | OSI/gOSI from orientation tuning curves or RF results |
| `tuning` | Validated direct extraction of one neuron's RF-map and tuning axes |
| `receptive_fields` | Pearson RF search, retinotopy, prediction helpers |
| `nonlinear_models` | lower-level model functions |
| `model_runs` | orchestration for simple and full nonlinear models |
| `trial_stats` | repeatability, skewness, and response-quality helpers |

::: waven.analysis.orientation_selectivity

::: waven.analysis.tuning

::: waven.analysis.receptive_fields

::: waven.analysis.nonlinear_models

::: waven.analysis.model_runs

::: waven.analysis.trial_stats
