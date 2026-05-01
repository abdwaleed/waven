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
- scipy
- scikit learn
- skimage
- tifffile
- pandas
- torch


**installation procedure:**<br />
From the repository root, create and activate the conda environment,
then install Waven in editable mode:

```bash
conda env create --solver libmamba -f environment.yml
conda activate waven
pip install -e .
```


**GUI Documentation:**<br />
https://docs.google.com/presentation/d/1nEv07CzCwYUoozucwwqi6qgS_t0jBy7KwqHKKoh2f2U/edit?usp=sharing<br />


**Example script**<br />
check out example.py

**GUI**<br />
<p align="center">
  <img src="https://github.com/skriabineSop/waven/blob/main/img/image1618.png" title="hover text">
 </p>



**Tutorial**

Setting up the parameters**

```python
	
	import waven.WaveletGenerator as wg
	import waven.Analysis_Utils as au
	import waven.LoadPinkNoise as lpn
	import waven.zebraGUI as ui
	import numpy as np
	import gc
	import os

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
	    "Number of Frames": "18000",
	    "Number of Trials to Keep": "3",
	    "Movie Path": "/home/sophie/Documents/POSTDOC/TEMP/videos/perlin_stimulus_10min.mp4",
	    "Library Path": "/home/sophie/Documents/POSTDOC/TEMP/gabors_library.npy",
	    "Spks Path": "None"
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
	    Movie Path: path to the stimulus (.mp4)
	    Library Path: path to Gabor library (same as save path if ran)
	    Spks Path (opt): path to the spks.npy file to skip the alignement procedure, if set ignores Parameter alignment
	"""
```

2. **To run the UI:**
```python
	
	ui.run(param_defaults,gabor_param)
```
documentation can be found here <https://docs.google.com/presentation/d/1nEv07CzCwYUoozucwwqi6qgS_t0jBy7KwqHKKoh2f2U/edit?usp=sharing>

3. **To create a new Gabor library**

```python

	if f!=0:
	    freq=True
	else:
	    freq=False
	L = wg.makeFilterLibrary(xs, ys, thetas, sigmas, offsets, f, freq=freq)
	np.save(path_save, L)
	lib_path=path_save
```

An already made Gabor Library well suited for mice can be found here <>

4. **Downsampling and adjusting the range of visual coverage to your analysis:**

```python
	
	if (visual_coverage!=analysis_coverage):
	    visual_coverage=np.array(visual_coverage)
	    analysis_coverage=np.array(analysis_coverage)
	    ratio_x=1-((visual_coverage[0]-visual_coverage[1])-(analysis_coverage[0]-analysis_coverage[1]))/(visual_coverage[0]-visual_coverage[1])
	    ratio_y=1-((visual_coverage[2]-visual_coverage[3])-(analysis_coverage[2]-analysis_coverage[3]))/(visual_coverage[2]-visual_coverage[3])
	else:
	    ratio_x=1
	    ratio_y=1

	## downsamples and wavelet transforms the stimulus
	wg.downsample_video_binary(movpath,visual_coverage,  analysis_coverage, shape=(ny, nx), chunk_size=1000, ratios=(ratio_x, ratio_y))
	path=os.path.dirname(movpath)
	videodata=np.load(movpath[:-4]+'_downsampled.npy')

	wg.waveletDecomposition(videodata, 0, sigmas, path, lib_path)
	wg.waveletDecomposition(videodata, 1, sigmas, path, lib_path)
```

5. **Loading you neural activity and neuron positions :**

```python
	
	spks=np.load(spks_path)
	parent_dir = os.path.dirname(spks_path)
	neuron_pos=np.load(os.join(parent_dir, 'pos.npy'))
	## converts neuron position in microns
	neuron_pos=lpn.correctNeuronPos(neuron_pos, resolution)
```
	
6. **Running the Analysis:**

```python
	
	## the spikes data have to be time registered to the stimulus frames
	respcorr_zebra = au.repetability_trial3(spks, neuron_pos, plotting=True)
	wavelets0, wavelets1, wavelet_c = lpn.coarseWavelet(path,False, nx, ny, 27, 11, n_theta, ns)

	## runs correlation analysis
	rfs_zebra = au.PearsonCorrelationPinkNoise(wavelet_c.reshape(18000, -1), np.mean(spks[:, :18000], axis=0),
		                                   neuron_pos, 27, 11, ns, analysis_coverage, screen_ratio, sigmas_deg,
		                                   plotting=True)
	## plot neuron receptive field
	idx=2441
	au.Plot_RF(rfs_zebra[0][idx],4, title=np.max(rfs_zebra[0][idx]))

	## plots neuron tuning curves
	tuning_curve=au.PlotTuningCurve(rfs_zebra, 2441, analysis_coverage, sigmas_deg, screen_ratio)
```
	
