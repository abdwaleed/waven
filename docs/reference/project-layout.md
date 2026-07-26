# Experiment folder

`Project Root` is the writable `your_experiment` directory. Keeping one
experiment's inputs, caches, and outputs below it makes cache provenance clear
and lets `pipeline_config.json` use portable `{PROJECT_ROOT}` paths.

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

| Location | Owner | Input or output? | Content | Required? |
| --- | --- | --- | --- | --- |
| `input/raw_data/` | User or a Waven folder reference | Input | Raw ephys/2-photon source material. | Required for fresh alignment only. |
| `input/stimulus_movie/` | User | Input | One supported stimulus movie. | Yes. |
| `input/neural_cache/` | Waven or user | Input/output | Matching aligned `spikes` and `pos` arrays; ephys may also carry unit metadata. | Required to continue from cache; created for fresh alignment. |
| `cache/gabor/` | Waven | Output/reusable input | Compact convolution-kernel cache keyed by grid and Gabor settings. | Created before wavelet preparation. |
| `cache/wavelets/coarse/` | Waven | Output/reusable input | Downsampled stimulus and Coarse RF power cache. | Required for Coarse RF analysis. |
| `cache/wavelets/full/` | Waven | Output/reusable input | Run Model and/or Full Model real/imaginary phase pairs. | Optional; only for model tuning. |
| `output/plots/` | Waven | Output | Optional GUI plot cache. | Optional. |
| `output/models/` | Waven | Output | Configured model-result location. | Optional unless a workflow writes model outputs there. |
| `output/recovery_cache/` | Waven | Output | Resumable work/checkpoint data for long tasks. | Optional but recommended. |

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

## Portable configurations

Use forward slashes and `{PROJECT_ROOT}` in `pipeline_config.json`:

```json
{
  "common": {
    "Project Root": "{PROJECT_ROOT}/your_experiment",
    "Movie Path": "{PROJECT_ROOT}/your_experiment/input/stimulus_movie",
    "Spks Path": "{PROJECT_ROOT}/your_experiment/input/neural_cache"
  }
}
```

At launch, Waven replaces `{PROJECT_ROOT}` with the repository directory that
contains `ui.py`. See [Configuration and Scientific Settings](../how-to/prepare-configuration.md)
for the full configuration schema.
