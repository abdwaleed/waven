# Plotting

Plotting helpers turn analysis arrays into Matplotlib figures and export
payloads. They should not be the source of scientific truth; they are views over
arrays produced by the pipeline and analysis modules.

## Plot payloads

| Plot family | Typical input | What to verify |
| --- | --- | --- |
| population maps | RF preferred values plus `neuron_pos` | coordinate axes and quality mask |
| individual neuron | spike trace, RF map, tuning curves | selected neuron index and frame range |
| OSI/gOSI histograms | `(n_neurons,)` metric arrays plus grouping labels | group sizes and NaN handling |
| exports | figure plus numeric arrays/metadata | saved image matches saved data bundle |

::: waven.plotting
