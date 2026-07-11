# First GUI analysis

This tutorial runs the interactive application from a configuration file and
produces the first coarse receptive-field plots.

## 1. Install and activate the environment

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
python -m pip install -e ".[docs]"
```

If you do not need documentation tooling on the analysis machine, use
`pip install -e .` instead.

## 2. Prepare `pipeline_config.json`

Copy the provided `pipeline_config.json` and edit paths so they point to local
folders. The GUI uses a strict folder layout under `Project Root`; file names
inside the final folder usually do not matter when there is only one valid
candidate.

- `Project Root`: root folder containing `input/`, `cache/`, and `output/`.
- `Movie Path`: folder containing one stimulus movie.
- `Save Path`, `Coarse Library Path`, and `Fine Library Path`: Gabor cache folders.
- `Spks Path`: folder containing or receiving aligned `spikes` and `pos` arrays.
- `Full Model Wavelet Path`: folder containing or receiving full-model wavelets.
- `Full Model Save Path`: folder for model outputs.

The standard layout is documented in [Project Layout](../reference/project-layout.md).

Before launching the GUI, sanity-check the largest dimensions:

| Quantity | Where to check | Why it matters |
| --- | --- | --- |
| `Number of Frames` | movie/config | Multiplies every wavelet output. |
| `NX`, `NY` | config | Full-model arrays grow with `NX * NY`; libraries grow roughly with `(NX * NY)^2`. |
| `N_thetas` | config | Adds orientation bins to libraries, wavelets, RF tensors, and OSI/gOSI curves. |
| `Sigmas Full Model` | config | Adds size bins to full-model wavelets. |
| `Frequencies` | config | Adds frequency bins to fine libraries and full-model wavelets. |

For a first run, it is reasonable to start with the coarse path and inspect RFs
before committing disk space to a large full-model wavelet set.

## 3. Launch the GUI

```python
from pathlib import Path

import waven

config = waven.PipelineConfig.from_json(Path("pipeline_config.json"))
waven.gui.run(
    config.analysis.to_gui_mapping(),
    config.gabor.to_gui_mapping(),
    workflow=config.workflow,
)
```

## 4. Build libraries

Choose **Analysis scale** in the session configuration before pressing the large
action buttons. The same buttons change their target:

| Scale | Button sequence | Result |
| --- | --- | --- |
| `coarse` | **Build Gabor Library (Coarse RF)**, then **Run Wavelet Decomposition (Coarse RF)** | coarse library, coarse phases, durable RF cache |
| `full` | **Build Gabor Library (Full model)**, then **Run Wavelet Decomposition (Full model)** | fine library and full-model real/imaginary wavelets |

Long tasks are resumable. If the application stops after a completed stage, run
the same button again; existing outputs are shape-checked and reused.

The status text beside the button reports the current stage, such as
downsampling, real phase, imaginary phase, or durable cache creation. If a task
fails, rerunning the button should skip completed compatible files and continue
from the next missing or invalid artifact.

## 5. Run coarse RF analysis

Click **Run Coarse RF Analysis**. The **All neurons** tab will show:

- neuron layout;
- population retinotopy maps;
- OSI and gOSI distributions by neuron/shank;
- OSI and gOSI histograms by unit or spatial fallback group.

The **Individual neuron** tab shows the selected neuron's spike train, receptive
field, feature tuning curves, and OSI/gOSI in the orientation panel title.

When judging the first plots, read them in this order:

1. Use repeatability and skewness to identify neurons worth trusting.
2. Check whether population retinotopy changes smoothly across anatomy.
3. Inspect individual RFs before trusting a population histogram.
4. Compare OSI/gOSI with the raw orientation tuning curve.

This order helps separate a real orientation preference from a noisy neuron that
happened to produce a sharp correlation peak.

After RF analysis, **Run Model Plots** also follows the selected scale:
`coarse` runs the simple coarse model, while `full` runs the full model using
the full wavelets and the coarse RF feature seeds.
