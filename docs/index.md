# WavEn documentation

Welcome to the current kernel-driven version of WavEn! These docs serve important information related to setup and system requirements. Please read the following sections in order:

1. [Install and Launch](how-to/install.md): setting up our conda environment.
2. [First GUI Analysis](tutorials/first-gui-analysis.md): onbaording from A to Z.
3. [GUI Onboarding](tutorials/gui-onboarding.md): starting your own experiment.
4. [Reusable Caches](reference/storage.md): saving resuable intermediate files.
5. [Restore GUI Settings (`pipeline_config.json`)](how-to/prepare-configuration.md): populating inputs off the bat.
6. [Export Results](how-to/export-results.md): showcasing your findings.

The science is simple: a stimulus movie and its visual coverage define a
spatial sampling grid; convolutional Gabor filters describe localized
orientation, size, frequency, and phase features on that grid; and
time-aligned neural activity is correlated or modelled (for amplitude, phase, and drift) against those features.
See [Scientific Intuition](explanation/pipeline-intuition.md) for the rationale
and [Data and Cache Contracts](explanation/maintenance/data-and-cache-contracts.md) for
technical array contracts.
