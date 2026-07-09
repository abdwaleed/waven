# Neural data

Neural-data modules read two-photon and electrophysiology sources before or
during alignment. They are lower-level than `waven.pipeline.load_spikes_and_positions()`;
use the pipeline helper unless you are debugging a loader.

## Shape expectations

| Data | Shape | Notes |
| --- | --- | --- |
| Suite2p traces/events | commonly `(n_neurons, n_samples)` before alignment | transposed/reshaped during frame alignment |
| ephys spike/event streams | event times plus unit/channel metadata | binned to stimulus frames later |
| final spikes | `(n_trials, n_frames, n_neurons)` | common shape consumed by RF/model code |
| positions | `(n_neurons, 2)` or `(n_neurons, 3)` | can encode x/y or x/y/depth-like coordinates |

The reference below is useful for exact loader signatures. For the conceptual
alignment contract, read [Time Alignment](time-alignment.md).

::: waven.data.neural

::: waven.suite_ephys.DIO

::: waven.suite_ephys.readTrodesExtractedDataFile3
