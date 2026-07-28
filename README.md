# waven

`waven` relates time-aligned neural responses to localized, convolutional
Gabor-wavelet features from a visual stimulus movie. Its desktop GUI builds a
durable cache at each stage, so an experiment can be resumed, inspected, and
exported without repeating every calculation.

## Launch

From the repository root:

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
python -m pip install -e .
python ui.py
```

`environment.yml` is the CUDA-enabled environment. For a CPU-only machine,
create a Python 3.10 conda environment and install `requirements-cpu.txt`
instead. Launch `ui.py` from the repository root so its default paths resolve
correctly.

## What to expect

The supported GUI workflow is convolution-only:

1. Prepare the cropped/downsampled stimulus cache.
2. Create or validate aligned neural `spikes` and `pos` caches.
3. Prepare convolution kernels and the cache needed by the next analysis.
4. Run Coarse RF analysis.
5. Inspect individual neurons; Run Model and Run Full Model tuning curves are
   optional additions.
6. Export selected all-neuron or individual-neuron graphs as PNG, SVG, and/or
   PKL.

Use **Run Guided Coarse RF Pipeline** in System Configuration to run steps
1–4 after the inputs are configured. **Save pipeline_config.json** is an
optional backup of the current GUI settings; **Load pipeline_config.json**
restores that backup.

For a new experiment, start with the ordered
[First GUI Analysis guide](docs/tutorials/first-gui-analysis.md). It covers
inputs, folder layout, scientific sampling, Gabor settings, hardware, cache
storage, and disk planning without duplicating information across pages.

## Resources at a glance

- **RAM:** 32 GB or more is recommended. Waven keeps durable arrays disk-backed;
  the optional RAM acceleration cache is bounded and only keeps safe reusable
  inputs resident.
- **Disk:** use local SSD/NVMe storage and reserve at least twice the GUI's
  estimated largest cache product, plus the raw movie and existing cache
  products. Run Full Model phase caches are usually the largest artifacts.
- **GPU:** an NVIDIA CUDA GPU is optional. Comparable GPUs can be combined for
  convolution; a substantially slower card can reduce throughput, so the GUI
  excludes unsuitable mixed sets.

## Documentation

The documentation site has one ordered GUI-first path:

- [First GUI Analysis](docs/tutorials/first-gui-analysis.md)
- [GUI Onboarding](docs/tutorials/gui-onboarding.md)
- [Reusable caches and disk planning](docs/reference/storage.md)
- [Restore a saved GUI setup](docs/how-to/prepare-configuration.md)
- [Inputs, outputs, and scientific contracts](docs/reference/data-contracts.md)
- [Source architecture](docs/reference/source-architecture.md)

To preview the documentation locally:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```
