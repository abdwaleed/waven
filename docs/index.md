# waven documentation

Welcome to the current, convolution-only Waven GUI workflow. This site is
organized by the order in which a new user makes decisions:

1. [First GUI Analysis](tutorials/first-gui-analysis.md) — the single ordered
   onboarding path, from hardware and inputs through Coarse RF and inspection.
2. [Install and Launch](how-to/install.md) — create the conda environment and
   start `ui.py`.
3. [Experiment Folder](reference/project-layout.md) — place raw inputs, caches,
   and results predictably.
4. [pipeline_config.json and Scientific Settings](how-to/prepare-configuration.md)
   — choose sampling, Gabor axes, and saved GUI settings.
5. [Cache Storage and Disk Planning](reference/storage.md) — choose NPY or
   Zarr and estimate resources before a long run.
6. [GUI Workflow and System Configuration](reference/gui.md) — optional speed
   controls, notifications, cancellation, and exports.

The primary scientific principle is simple: a stimulus movie and its visual
coverage define a spatial sampling grid; convolutional Gabor filters describe
localized orientation, size, frequency, and phase features on that grid; and
time-aligned neural activity is correlated or modelled against those features.
See [Scientific Intuition](explanation/pipeline-intuition.md) for the rationale
and [Inputs, Outputs, and Cache Contracts](reference/data-contracts.md) for
exact array shapes.
