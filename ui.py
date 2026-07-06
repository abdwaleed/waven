from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_SRC = PROJECT_ROOT / "src"
if str(LOCAL_SRC) not in sys.path:
    sys.path.insert(0, str(LOCAL_SRC))

import waven

workflow = waven.gui.select_workflow()
config = waven.PipelineConfig.from_json(PROJECT_ROOT / "pipeline_config.json", workflow=workflow)

waven.gui.run(config.analysis.to_gui_mapping(),
    config.gabor.to_gui_mapping(),
    workflow=workflow,
)
