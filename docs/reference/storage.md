# Storage

Storage helpers abstract the difference between plain `.npy` arrays and Zarr
stores. They do not change scientific shapes; they only change how arrays are
opened, memory-mapped, chunked, or converted.

## When to use each format

| Format | Strength | Tradeoff |
| --- | --- | --- |
| `.npy` | simple, exact NumPy serialization; good for moderate arrays | one large file; awkward for very large full-model phases |
| memory-mapped `.npy` | reads slices without loading the whole file | still one large uncompressed file |
| Zarr | chunked and compressed; friendlier for huge full-model wavelets | directory store; requires `zarr` and `numcodecs` |

Full-model wavelets are the usual Zarr candidate:

```text
(n_frames, NX, NY, n_orientations, n_sigmas, n_frequencies)
```

The logical dtype remains `float32`.

::: waven.storage.array_store

::: waven.storage.neural_cache

::: waven.storage.wavelet_zarr

::: waven.project_layout
