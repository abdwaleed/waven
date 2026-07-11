# Project layout

`waven` now treats one **Project Root** as the anchor for all input folders,
generated caches, and analysis outputs. GUI path fields are folders, not files.
When a folder contains the expected artifact, the file name can be arbitrary
unless the artifact has a semantic pair such as `spikes` and `pos`.

```text
your_experiment/
  input/
    raw_data/
    stimulus_movie/
    neural_cache/
  cache/
    gabor/
      coarse/
      full/
      kernels/
    wavelets/
      coarse/
      full/
  output/
    plots/
    recovery_cache/
    models/
```

## Input folders

| Folder | Put here | Discovery rule |
| --- | --- | --- |
| `input/raw_data/` | Raw two-photon or ephys acquisition files. | Alignment code reads this folder, or a reference stored in it. |
| `input/stimulus_movie/` | One stimulus movie, such as `.mp4`, `.avi`, `.mov`, `.mkv`, `.wmv`, or `.m4v`. | The GUI searches the folder and requires exactly one movie candidate. |
| `input/neural_cache/` | Existing or generated aligned neural arrays. | Exact `spikes.npy`/`pos.npy` or `.zarr` names are preferred. If absent, one array with `spike`/`spks` in the stem and one with `pos`/`position` in the stem can be used. |

If a large input lives elsewhere, use the GUI browse button for that folder. The
GUI writes a lightweight `.waven_reference.json` into the strict folder and
keeps the visible field pointed at the strict folder. This preserves a stable
project tree without copying large source data.

## Cache folders

| Folder | Produced artifacts | Notes |
| --- | --- | --- |
| `cache/gabor/coarse/` | Coarse RF Gabor library. | Generated names are deterministic, but loading accepts a single `.npy` or `.zarr` in the folder. |
| `cache/gabor/full/` | Full-model/fine Gabor library. | Kept separate from coarse so arbitrary file names remain unambiguous. |
| `cache/gabor/kernels/` | Compact convolution kernels. | Used by the convolution backend instead of giant flattened libraries. |
| `cache/wavelets/coarse/` | Coarse downsample cache, phase scratch files, and `dwt_downsampled_videodata.npy`. | RF analysis reads this folder. |
| `cache/wavelets/full/` | Full-model real and imaginary wavelet phases. | Full model reads this folder. |

Generated caches carry `.waven.json` sidecars where practical. These sidecars
store shape and parameter fingerprints so reruns can decide whether a cache is
still compatible with the GUI inputs.

## Output folders

| Folder | Produced artifacts |
| --- | --- |
| `output/plots/` | GUI plot cache and exported plot bundles. |
| `output/recovery_cache/` | Durable task checkpoints for resumable buttons. |
| `output/models/` | Full-model outputs and model-side result files. |

## Metadata sanity checks

When the GUI resolves the stimulus movie, it reads only lightweight metadata:
frame count, frame rate, width, and height. If those disagree with `Number of
Frames`, `Hz`, `NX`, or `NY`, the GUI shows a Yes/No warning. Choosing **Yes**
updates the GUI fields to match the movie metadata; choosing **No** leaves your
typed values unchanged.
