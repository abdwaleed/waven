import waven
from pathlib import Path

config = waven.PipelineConfig.from_json(Path(r'C:\Users\Abdelrahman\Documents\VISUAL STUDIO CODE PROJECTS\surf\gabor-analysis\waven\pipeline_config.json'))

waven.gui.run(config.analysis.to_gui_mapping(),
    config.gabor.to_gui_mapping(),
)