# Time alignment

Time alignment converts acquisition-specific timing into stimulus-frame-aligned
neural arrays. The rest of the pipeline assumes this step has produced a stable
`spikes` tensor.

## Expected output

| Output | Shape | Meaning |
| --- | --- | --- |
| `spikes` | `(n_trials, n_frames, n_neurons)` | trial-wise neural responses sampled on stimulus frames |
| `spike_counts` (ephys STA option) | `(n_trials, n_frames, n_neurons)` | raw integer events per stimulus frame; never divided by frame duration |
| `aligned_spikes` | workflow-dependent | intermediate or richer aligned response payload |
| `neuron_pos` | `(n_neurons, 2)` or `(n_neurons, 3)` | anatomical coordinates used by plots and grouping |

Frame alignment is high stakes: a one-frame offset changes correlations and RF
locations. If RF maps look noisy across all neurons, inspect alignment before
tuning model parameters.

For ephys cache creation, waven preserves both representations: `spikes` is
the firing-rate (Hz) array consumed by the established RF/OSI/model workflow,
while `spike_counts` is a sibling cache for optional spike-triggered averaging.
See [Spike-Triggered Averaging](spike-triggered-averaging.md) for why counts
must not be replaced by rates in that calculation.

::: waven.time_alignment
