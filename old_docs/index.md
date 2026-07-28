# WavEn documentation

Welcome to the current kernel-driven version of WavEn! These docs serve important information related to setup and system requirements. Please read the following sections in order:

1. [First GUI Analysis](tutorials/first-gui-analysis.md): onboarding from A to Z.
2. [Install and Launch](how-to/install.md): creating our conda environment and running `ui.py`.
3. [Experiment Folder](reference/project-layout.md): the "standard" folder layout in which to find inputs and outputs or "reference" them from external folder(s).
4. [pipeline_config.json and Scientific Settings](how-to/prepare-configuration.md): inputting our raw data paths, modifying Gabor and other params, and handling future caches and results paths.
5. [GUI Workflow and System Configuration](reference/gui.md): speed
   controls, notifications, cancellation, and exports.
6. [Cache Storage and Disk Planning](reference/storage.md): choosing between NPY and Zarr and how Zarr promises better disk utilization.

The science is simple: a stimulus movie and its visual
coverage define a spatial sampling grid; convolutional Gabor filters describe
localized orientation, size, frequency, and phase features on that grid; and
time-aligned neural activity is correlated or modelled against those features.
See [Scientific Intuition](explanation/pipeline-intuition.md) for the rationale
and [Inputs, Outputs, and Cache Contracts](reference/data-contracts.md) for
exact array shapes.
