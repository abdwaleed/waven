# Run RF analysis

In the GUI, click **Run Coarse RF Analysis** after wavelet decomposition.

From Python:

```python
import waven

config = waven.PipelineConfig.from_json("pipeline_config.json")
spike_data = waven.load_spikes_and_positions(config.analysis)
rf_result = waven.run_rf_analysis(
    config.analysis,
    config.gabor,
    spike_data,
    plotting=True,
)
```

The RF stage computes:

- repeatability across trials;
- skewness of neural activity;
- Pearson correlation between wavelet features and neural responses;
- preferred azimuth, elevation, orientation, size, and frequency;
- OSI and gOSI distributions in the GUI.

## Inputs

RF analysis needs aligned neural responses and the coarse wavelet cache:

| Input | Shape | Meaning |
| --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | Trial responses aligned to stimulus frames. Values may be deconvolved activity, thresholded events, or aligned spike-like responses depending on workflow. |
| `neuron_pos` | `(n_neurons, 2)` or `(n_neurons, 3)` | Anatomical positions used for scatter plots, quality overlays, shank/unit grouping, and neighborhood smoothing. |
| `coarse_rf_power` | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` | Disk-backed wavelet power used as the RF stimulus feature tensor. |

`run_rf_analysis` truncates all time-dependent inputs to the shortest available
frame count. That protects against small movie/spike/cache length mismatches,
but large mismatches should be treated as an alignment problem.

## RF tensor

The stimulus cache is reshaped to `(n_frames, n_features)`, where:

```text
n_features = coarse_nx * coarse_ny * n_orientations * n_sigmas * n_frequencies
```

The RF output is then organized back into feature axes:

```text
(n_neurons, coarse_nx, coarse_ny, n_orientations, n_sigmas, n_frequencies)
```

For each neuron, `waven` stores both the full RF tensor and the preferred feature
indices. Preferred azimuth and elevation are derived from the `x` and `y` index
plus `Analysis Coverage`; preferred size is derived from sigma in pixels and
converted to approximate visual degrees for display.

## Quality interpretation

Repeatability asks whether the neuron responds consistently across repeated
trials. Skewness flags neurons whose activity is dominated by rare large events.
The GUI does not silently delete low-quality neurons; it lowers their visual
alpha so population structure remains visible while the quality mask is still
obvious.

OSI and gOSI do **not** use RF correlations. The RF tensor selects each
neuron's preferred position, size, and frequency; Waven then uses the aligned
neural response cache to calculate a firing-rate orientation curve weighted by
the power at that preferred feature. For ephys this uses frame-bin Hz directly.
A high selectivity value on a noisy, low-repeatability neuron is a hypothesis to
inspect, not a conclusion by itself.

In two-photon mode, the **All neurons** tab shows the all-cell OSI/gOSI
distributions only. In ephys mode, the GUI also overlays shank distributions
when shank-like metadata are present and adds by-unit histograms, because unit
grouping is an electrophysiology concept in this workflow.
