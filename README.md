# waven

`waven` relates time-aligned neural responses to localized Gabor-wavelet
features extracted from a visual stimulus movie. Its GUI leads a new user from
movie metadata through neural alignment, Gabor assets, consumer-specific
wavelet caches, receptive fields, and nonlinear models.

## Start here

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
python -m pip install -e .
python ui.py
```

The full setup guide, including CUDA and developer instructions, is in
[docs/how-to/install.md](docs/how-to/install.md). Use `python ui.py` from the
repository root so the bundled `pipeline_config.json` and project-relative
paths resolve correctly.

## GUI workflow

1. **Stimulus & Metadata** — select one movie, choose a downsampling
   percentage, and prepare the binary stimulus cache. Movie metadata is the
   authoritative source for width, height, frame count, FPS, and duration.
2. **Session Setup** — create an aligned neural cache from raw data or validate
   an existing cache. Ephys caches hold frame-bin firing rate in Hz.
3. **Gabor** — build legacy filter libraries or compact convolution kernels.
   Gabor phases are entered in degrees.
4. **Wavelet Products** — prepare the cache needed by the next analysis:
   Coarse RF power, Run Model real/imaginary phases, or Run Full Model
   real/imaginary phases.
5. **Analysis** — run Coarse RF, then Run Model and/or Run Full Model.

There are no GUI fields for `NX`, `NY`, manual movie FPS, or hardcoded
duration. The selected movie and one percentage determine the shared spatial
grid throughout the application.

## Important arrays

| Artifact | Shape | Purpose |
| --- | --- | --- |
| aligned neural cache | `(trials, frames, neurons)` | Ephys values are firing rate in Hz; two-photon values are aligned activity. |
| downsampled movie | `(frames, y, x)` | Disk-backed binary movie used to make wavelets. |
| Coarse RF power | `(frames, x, y, orientations, sigmas)` | Zarr input to Coarse RF correlation. |
| coarse model phases | `(frames, x, y, orientations, sigmas)` each | Real/imaginary inputs to Run Model. |
| full model phases | `(frames, x, y, orientations, sigmas, frequencies)` each | Real/imaginary inputs to Run Full Model. |
| RF tensor | `(neurons, x, y, orientations, sigmas, frequencies)` | Correlation map used to select and inspect features. |

The Individual Neuron RF curves are correlation diagnostics. OSI and gOSI are
calculated from the aligned neural response/firing-rate cache at each neuron's
preferred RF feature—not from correlation values.

## Documentation

- [First GUI analysis](docs/tutorials/first-gui-analysis.md)
- [Prepare configuration](docs/how-to/prepare-configuration.md)
- [Prepare wavelet products](docs/how-to/run-wavelet-decomposition.md)
- [Run RF analysis](docs/how-to/run-rf-analysis.md)
- [Pipeline intuition](docs/explanation/pipeline-intuition.md)
- [Code organization](docs/explanation/maintainability.md)

Build the local documentation site with:

```bash
python -m pip install -e ".[docs]"
python -m mkdocs serve
```
