# Experiment folder

`Project Root` is the `your_experiment` directory, the default structure the
program reads from and writes. It keeps all inputs,
caches, and outputs here and stores your inputs here either by value or by reference depending on
where you pasted your raw data and stimulus movie files. This default structure lets `pipeline_config.json`
use portable system-to-system `{PROJECT_ROOT}` paths too.

```text
your_experiment/
├── input/
│   ├── raw_data/
│   ├── stimulus_movie/
│   └── neural_cache/
├── cache/
│   ├── gabor/
│   │   ├── coarse/
│   │   ├── full/
│   │   └── kernels/
│   └── wavelets/
│       ├── coarse/
│       └── full/
└── output/
    ├── plots/
    ├── models/
    └── recovery_cache/
```

| Location | Input or output? | Content | Required? |
| --- | --- | --- | --- |
| `input/raw_data/` | Input | Raw ephys/2-photon source material. | To use your new data, *obviously* yes |
| `input/stimulus_movie/` | Input | One supported stimulus movie. | To use your new data, *obviously* yes |
| `input/neural_cache/` | Input/output | Matching aligned `spikes` and `pos` arrays; ephys may also carry unit metadata. | Optional way to reuse a cache from previous `input/raw_data/` + `input/stimulus_movie/` |
| `cache/gabor/` | Output/reusable input | Compact convolution-kernel cache keyed by grid and Gabor settings. | Created before wavelet preparation to inform the Gabors used in wavelet decomposition. Fortunately, this is **NOT** a huge library as the legacy code thanks to convolution. |
| `cache/wavelets/coarse/` | Output/reusable input | Downsampled stimulus and Coarse RF power cache. | Required for Coarse RF analysis. |
| `cache/wavelets/full/` | Output/reusable input | Run Model and/or Full Model real/imaginary phase pairs. | Optional; only for model tuning. |
| `output/plots/` | Output | Optional GUI plot cache. | Optional. |
| `output/models/` | Output | Configured model-result location. | Optional unless a workflow writes model outputs there. |
| `output/recovery_cache/` | Output | Resumable work/checkpoint data for long tasks. | Optional but recommended for restarting failed intermediate tasks, perhaps due to running out of storage. |

## Path rules

- GUI path fields name **folders**, not individual files.
- The movie folder should contain one intended movie so metadata discovery is
  unambiguous.
- For a fresh session, Waven writes the aligned neural pair to `Spks Path`.
  For a continue session, point `Spks Path` at an existing matching pair.
- If a raw-data directory lives outside the project, Waven can store a
  `.waven_reference.json` pointer in the conventional location. The source data
  are not copied merely by choosing the external folder.
- Do not manually move or rename cache products while a task is running.
  Cache provenance uses the movie, coverage, grid, Gabor settings, and product
  purpose to determine safe reuse.
