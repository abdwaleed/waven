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
