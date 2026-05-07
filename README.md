<div>
    <img src="https://github.com/skriabineSop/waven/blob/main/img/image1630.png" width="250" align="right"/>
</div>

**Waven**



This project provides a Python package designed to analyze neuronal responses in the visual cortex to visual stimuli. Using a Gabor transform of the stimulus, the package enables users to extract tuning curves for key visual features such as azimuth, elevation, orientation, spatial frequency, phase, size, and drift speed.<br />

**General Documentation and tutorial**<br />
can be found here <https://waven.readthedocs.io/en/latest/><br />


**Stimulus Generation package**<br />
check out <https://github.com/mwshinn/zebra_noise><br />


**Waven Analysis package**<br />
packages required:

- python 3.8
- matplotlib
- numpy
- opencv_python
- scikit learn
- scikit_image
- scipy
- tifffile
- pandas
- torch
- tensorly
- zarr


**installation procedure:**<br />
From the repository root, create and activate the conda environment,
then install Waven in editable mode:

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
pip install -e .
```

**Example script**<br />

check out
/example/run_waven_parts

**GUI Documentation:**<br />
https://docs.google.com/presentation/d/1nEv07CzCwYUoozucwwqi6qgS_t0jBy7KwqHKKoh2f2U/edit?usp=sharing<br />


**GUI**<br />
<p align="center">
  <img src="https://github.com/skriabineSop/waven/blob/main/img/image1618.png" title="hover text">
 </p>



**Tutorial**

Setting up the parameters**

```python
	

	# List of default parameters for the Gabor Library
	gabor_param={
	    "N_thetas":"8",
	    "Sigmas": "[2, 3, 4, 5, 6, 8]",
	    "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
	    "Phases": "[0, 90]",
	    "NX": "135",
	    "NY": "54",
	    "Save Path":"/home/sophie/Documents/POSTDOC/TEMP/gabors_library.npy"
	}

	# List of default parameters
	param_defaults = {
	    "Path Directory": "/media/sophie/Expansion1/UCL/datatest/videos",
	    "Dirs": "/media/sophie/Seagate Basic/datasets",
	    "Experiment Info": "('SS002', '2024-07-23', 3)",
	    "Number of Planes": "1",
	    "Block End": "0",
	    "screen_x":"4096",
	    "screen_y":"1536",
	    "NX": "135",
	    "NY": "54",
	    "Resolution":"1.3671",
	    "Sigmas": "[2, 3, 4, 5, 6, 8]",
	    "Frequencies": "[0.015, 0.04, 0.07, 0.1]",
	    "Visual Coverage":"[-135, 45, 34, -34]",
	    "Analysis Coverage": "[-135, 0, 34, -34]",
		"Hz": "30",
	    "Number of Frames": "18000",
	    "Number of Trials to Keep": "3",
	    "Movie Path": "/home/sophie/Documents/POSTDOC/TEMP/videos/perlin_stimulus_10min.mp4",
	    "Library Path": "/home/sophie/Documents/POSTDOC/TEMP/gabors_library.npy",
	    "Spks Path": "None"
		"Full Model Wavelet Path": "/home/sophie/Documents/POSTDOC/TEMP",
		"Full Model Save Path": "/home/sophie/Projects/ZebrAnalysis/zebranalysis/tests"
	}
```

Here is a quick explanation of each parameter:

```python
	
	"""
	Parameters Gabor Library:
	    N_thetas (int): number of orientatuion equally spaced between 0 and 180 degree.
	    Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
	    Frequencies (list): spatial frequencies expressed in pixels per cycles.
	    Phases (list): 0 and pi/2.
	    NX (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
	    NY (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
	    Save Path (string): where to save the gabor library

	Parameters alignement:
	    Dirs (string): where the raw data are.
	    Experiment Info: (mouse name, data, experiment number)
	    Number of Planes (int): number of acquisition planes.
	    Block End (int): timeframe where the experiment starts.
	    Number of Frames (int): number of frames stim 30 Hz -> 1800 frame/min.
	    Number of Trials to Keep(int): Number of Trials to Keep.

	Parameters analysis:
	    screen_x: stimulus screen x size inn pixels.
	    screen_y: stimulus screen y size inn pixels.
	    NX (int): number of azimuth positions (pix) (x shape of the downsampled stimuli).
	    NY (int): number of elevation positions (pix) (y shape of the downsampled stimuli).
	    Resolution (float): microscope resolution (um per pixels)
	    Sigmas (list): standart deviation of theb gabor filters expressed in pixels (radius of the gaussian half peak wigth).
	    Visual Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
	    Analysis Coverage (list): [azimuth left, azimuth right, elevation top , elevation bottom] in visual degree.
		Hz: the frame rate of the moplayed movie
	    Movie Path: path to the stimulus (.mp4)
	    Library Path: path to Gabor library (same as save path if ran)
	    Spks Path (opt): path to the spks.npy file to skip the alignement procedure, if set ignores Parameter alignment
	"""
```

2. **To run the UI:**
```python
	import waven

	config = waven.PipelineConfig.from_json(Path('path to pipeline_config.json'))
	waven.gui.run(
    config.analysis.to_gui_mapping(),
    config.gabor.to_gui_mapping(),
)
```
documentation can be found here <https://docs.google.com/presentation/d/1nEv07CzCwYUoozucwwqi6qgS_t0jBy7KwqHKKoh2f2U/edit?usp=sharing>

3. **To create a new Gabor library**

```python

    library_path = create_gabor_library(config.gabor)
    print(f"Created Gabor library: {library_path}")
```

An already made Gabor Library well suited for mice can be found on FigShare <https://doi.org/10.5522/04/31295536>

4. ** Running the wavelet decomposition :**

```python

	config = waven.PipelineConfig.from_json(Path('path to pipeline_config.json'))
	library_path = config.analysis.library_path
	
	output_dir = prepare_stimulus_wavelets(
        config.analysis,
        library_path=library_path,
    )
	print(f"Prepared wavelets in: {output_dir}")
```
For more effiscient analysis, we advise to save the resulting library as a zarr folder (check the wavelet_zarr.py script for more details) and to set the parameter "Full Model Wavelet Path" to the path of the zarr folder, this way the wavelet decomposition will be skipped when running the full model.

5. **Loading you neural activity and neuron positions :**

```python
	
	spike_data = load_spikes_and_positions(config.analysis)
```

	
6. **Running the Quick receptive field Analysis:**

```python
	
	rf_analysis = run_rf_analysis(
		config.analysis,
		config.gabor,
		spike_data,
		plotting=True,
		neuron_id=2441,
	)
	print("RF correlation analysis complete")
```

7. **Running the Full Model :**

```python

	full_model = run_full_model(config, spike_data, rf_analysis, tt = [0, 36000])
	print(f"Full model complete: {type(full_model).__name__}")
```
	
