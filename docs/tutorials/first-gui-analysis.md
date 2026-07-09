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
files. The most important paths are:

- `Movie Path`: stimulus movie, usually `.mp4`.
- `Save Path`: default Gabor-library output path.
- `Coarse Library Path` and `Fine Library Path`: explicit reusable library paths.
- `Spks Path`: optional aligned `spikes.npy`; use `None` to run alignment.
- `Full Model Wavelet Path`: folder containing or receiving full-model wavelets.
- `Full Model Save Path`: folder for model outputs, plot cache, and exports.

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

Run these buttons in order:

1. **Build Coarse Library**
2. **Build Fine Library**
3. **Run Wavelet Decomposition**

Long tasks are resumable. If the application stops after a completed stage, run
the same button again; existing outputs are shape-checked and reused.

## 5. Run coarse RF analysis

Click **Run Coarse RF Analysis**. The **All neurons** tab will show:

- neuron layout;
- population retinotopy maps;
- OSI and gOSI distributions by neuron/shank;
- OSI and gOSI histograms by unit or spatial fallback group.

The **Individual neuron** tab shows the selected neuron's spike train, receptive
field, feature tuning curves, and OSI/gOSI in the orientation panel title.
