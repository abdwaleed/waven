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

## Why does a movie shape look like `(frames, NY, NX)` but wavelets use `(frames, NX, NY, ...)`?

Downsampled movies are image arrays, so their spatial axes are row then column:
`(y, x)`. Gabor libraries and RF tensors use feature coordinates: `(x, y)`.
Both are expected. The safest check is to compare the file with the documented
stage that produced it.

## Why are `.npy` wavelets so large?

Wavelet arrays are dense `float32` tensors. Every extra frame, grid point,
orientation, sigma, and frequency multiplies the total size. One full-model
phase is:

```text
n_frames * NX * NY * n_orientations * n_sigmas * n_frequencies * 4 bytes
```

Real and imaginary phases are separate files.

## When should I use Zarr?

Use Zarr for full-model wavelets when the logical array is too large to handle
comfortably as one `.npy` file, when you want chunked access, or when compression
is useful. Zarr does not change the scientific shape or dtype; it changes how
the same array is stored on disk.

## Can a high OSI be misleading?

Yes. OSI and gOSI summarize the orientation tuning curve, but they do not prove
that the neuron is reliable. Inspect repeatability, skewness, the RF map, and
the raw tuning curve before treating a selectivity value as biologically strong.
