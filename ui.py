import waven
from pathlib import Path

config = waven.PipelineConfig.from_json(Path(r'C:\Users\aa508\Documents\waven_june24\pipeline_config.json'))

waven.gui.run(config.analysis.to_gui_mapping(),
    config.gabor.to_gui_mapping(),
)