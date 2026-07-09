# Prepare configuration

The easiest starting point is `pipeline_config.json`.

Check these fields first:

| Field | Purpose |
| --- | --- |
| `Movie Path` | Stimulus movie used for wavelet decomposition. |
| `Dir` | Root data directory. |
| `Experiment Info` | Tuple-like experiment identifier. |
| `NX`, `NY` | Full analysis stimulus grid. |
| `Visual Coverage` | Full stimulus visual field in degrees. |
| `Analysis Coverage` | Cropped analysis field in degrees. |
| `Sigmas` | Coarse RF sigma list. |
| `Sigmas Full Model` | Full model sigma list. |
| `Frequencies` | Spatial frequencies for the fine library/model. |
| `Spks Path` | Existing aligned spikes or `None`. |

Programmatic loading:

```python
from pathlib import Path

import waven

config = waven.PipelineConfig.from_json(Path("pipeline_config.json"))
```
