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
