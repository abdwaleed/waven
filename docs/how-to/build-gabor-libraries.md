# Build Gabor libraries

Build the coarse and fine libraries before decomposing the stimulus.

```python
import waven

config = waven.PipelineConfig.from_json("pipeline_config.json")

coarse = waven.create_coarse_gabor_library(config.gabor)
fine = waven.create_fine_gabor_library(
    config.gabor,
    extra_sigmas=config.analysis.sigmas_full_model,
)
```

The coarse library is used for fast receptive-field screening. The fine library
is used for full-resolution model fitting.

In the GUI, use:

1. **Build Coarse Library**
2. **Build Fine Library**

Existing `.npy` or `.zarr` libraries are reused only when their shape matches the
current configuration.
