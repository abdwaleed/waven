# FAQ

## Where should I save movies and wavelet files?

Use local storage. Reading and writing from a server or network drive can make
wavelet decomposition and model fitting much slower.

## Can Gabor libraries be reused?

Yes. Reuse them when `NX`, `NY`, orientations, sigmas, phases, and frequencies
match the experiment configuration. The GUI now checks output shapes before
reusing artifacts.

## Why does preprocessing take so long?

Wavelet decomposition projects every stimulus frame onto many filters. The first
run is expensive, but durable `.npy` or `.zarr` outputs are reused by later runs.

## What should I export for a paper figure?

Use the GUI export buttons. They save the rendered figure and a data bundle with
numeric arrays and metadata.

## What is the difference between OSI and gOSI?

OSI compares the preferred and orthogonal orientation responses. gOSI summarizes
the whole orientation tuning curve with a vector sum.
