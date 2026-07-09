# Time alignment

Time alignment converts acquisition-specific timing into stimulus-frame-aligned
neural arrays. The rest of the pipeline assumes this step has produced a stable
`spikes` tensor.

## Expected output

| Output | Shape | Meaning |
| --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | trial-wise neural responses sampled on stimulus frames |
| `aligned_spikes` | workflow-dependent | intermediate or richer aligned response payload |
| `neuron_pos` | `(n_neurons, 2)` or `(n_neurons, 3)` | anatomical coordinates used by plots and grouping |

Frame alignment is high stakes: a one-frame offset changes correlations and RF
locations. If RF maps look noisy across all neurons, inspect alignment before
tuning model parameters.

::: waven.time_alignment
