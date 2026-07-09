# Run wavelet decomposition

In the GUI, click **Run Wavelet Decomposition**. The button prepares:

- coarse downsampled stimulus movie;
- coarse real phase wavelets;
- coarse imaginary phase wavelets;
- durable coarse RF cache;
- full-resolution downsampled stimulus movie;
- full-model real and imaginary wavelets as `.npy` or `.zarr`.

The same work can be run from Python:

```python
import waven

config = waven.PipelineConfig.from_json("pipeline_config.json")

waven.prepare_stimulus_wavelets(
    config.analysis,
    library_path=config.gabor.coarse_save_path,
)
waven.prepare_full_model_wavelets(
    config.analysis,
    config.gabor,
    library_path=config.gabor.fine_save_path,
)
```

If a long run is interrupted, run the command or GUI button again. Existing
outputs are shape-validated and reused.
