# FAQ

## Do I need a CUDA GPU?

No. Waven falls back to CPU. A compatible CUDA GPU accelerates convolution and some analysis work. Enable multi-GPU only for comparable GPUs; a slower card can reduce batch throughput.

## How much RAM and free disk do I need?

32 GB RAM or more is recommended for large experiments. Reserve at least twice the GUI estimate of the largest cache product, plus the raw movie, neural cache, and any products you will retain. See [Cache Storage and Disk Planning](reference/storage.md).

## NPY or Zarr?

Choose NPY for a simple memory-mappable file. Choose Zarr for large chunked, compressed, resumable caches. This choice applies to caches, not exported graphs. Convolutional phase and power products use Zarr.

## Why is a button disabled or unavailable?

An upstream input/cache is missing or incompatible. Work from the top of the GUI workflow: valid movie/sampling, matching neural cache, convolution kernels, the cache needed by the desired analysis, then Coarse RF analysis.

## Why did Inspect Single Neuron take a long time?

Basic inspection renders Coarse RF and STA data. The optional Run Model and Full Model checkboxes add phase-aware model fitting and amplitude/phase/drift graphs. Leave them off for routine inspection; a recently inspected neuron's STA is cached for faster revisits.

## Can I cancel without losing everything?

Yes. Cancel requests safe cooperative stopping at chunk, neuron, or iteration boundaries. Valid completed tiles/caches remain reusable. Force-closing the process is more likely to leave work incomplete.

## What does Save/Load pipeline_config.json do?

It records/restores GUI inputs and optional runtime choices for later actions. It does not alter a background task that has already captured its settings.
