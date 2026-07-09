# Scripted pipeline

This tutorial runs the main analysis stages without opening the GUI.

```python
from pathlib import Path

import waven

config = waven.PipelineConfig.from_json(Path("pipeline_config.json"))

coarse_library = waven.create_coarse_gabor_library(config.gabor)
fine_library = waven.create_fine_gabor_library(
    config.gabor,
    extra_sigmas=config.analysis.sigmas_full_model,
)

waven.prepare_stimulus_wavelets(config.analysis, library_path=coarse_library)
waven.prepare_full_model_wavelets(
    config.analysis,
    config.gabor,
    library_path=fine_library,
)

spike_data = waven.load_spikes_and_positions(config.analysis)
rf_result = waven.run_rf_analysis(config.analysis, config.gabor, spike_data)
```

At this point the main in-memory objects have these shapes:

| Object | Expected shape |
| --- | --- |
| `spike_data.spikes` | `(n_trials, n_frames, n_neurons)` |
| `spike_data.neuron_pos` | `(n_neurons, 2)` or `(n_neurons, 3)` |
| `rf_result.wavelets.wavelets_complex` | `(n_frames, coarse_nx, coarse_ny, n_orientations, n_sigmas)` |
| `rf_result.repeatability` | usually one value per neuron |
| `rf_result.rf_results[0]` | `(n_neurons, coarse_nx, coarse_ny, n_orientations, n_sigmas, n_frequencies)` |

Use those shapes as a quick debugging checklist. If the neuron axis or frame
axis is different from the table, fix alignment before interpreting RFs.

For selected neurons, run the nonlinear models:

```python
wavelets = waven.load_coarse_wavelets(config.analysis, config.gabor)
simple_model = waven.run_simple_model(
    config.analysis,
    config.gabor,
    spike_data,
    wavelets,
    rf_result,
)

full_model = waven.run_full_model(config.analysis, spike_data, rf_result)
```

Use the GUI when you want interactive inspection and export bundles. Use the
scripted path for reproducible batch work.

For scripts that run on shared machines, prefer writing large outputs to local
scratch storage and copying only the final exports or model summaries back to
shared storage. The dense full-model wavelets are deterministic cache files, not
hand-curated data products.
