.. _quickstart:

Quick Start Guide
=================

This page explains how to use the above code to extract visual feature preferences from 2p neural datas

Setting up the parameters**

.. code-block:: python
	
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
	    "Save Path":"/path/to/your/ressources/folder/gabors_library.npy"
	}

	# List of default parameters
	param_defaults = {
	    "Path Directory": "/path/to/your/folder/videos",
	    "Dirs": "/path/to/your/dataset",
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
	    "Movie Path": "/path/to/your/folder/perlin_stimulus_10min.mp4",
	    "Library Path": "/path/to/your/folder/gabors_library.npy",
	    "Spks Path": "None"
	}
Here is a quick explanation of each parameter:

.. code-block:: python
	
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

further documentation can be found here <https://github.com/skriabineSop/waven/tree/main/pydocs>

2. **To run the UI:**
.. code-block:: python
	
	ui.run(param_defaults,gabor_param)

GUI documentation can be found here <https://docs.google.com/presentation/d/1nEv07CzCwYUoozucwwqi6qgS_t0jBy7KwqHKKoh2f2U/edit?usp=sharing>

3. **To create new coarse and fine Gabor libraries**

.. code-block:: python

	import waven

	config = waven.PipelineConfig.from_json("pipeline_config.json")
	coarse_library_path = waven.create_coarse_gabor_library(config.gabor)
	fine_library_path = waven.create_fine_gabor_library(
	    config.gabor,
	    extra_sigmas=config.analysis.sigmas_full_model,
	)

An already made Gabor Library well suited for mice can be found here <>

4. **Preparing coarse RF and full-model wavelets:**

.. code-block:: python
	
	coarse_dir = waven.prepare_stimulus_wavelets(
	    config.analysis,
	    library_path=coarse_library_path,
	)
	full_dir = waven.prepare_full_model_wavelets(
	    config.analysis,
	    config.gabor,
	    library_path=fine_library_path,
	)

5. **Loading you neural activity and neuron positions :**

.. code-block:: python
	
	spks=np.load(spks_path)
	parent_dir = os.path.dirname(spks_path)
	neuron_pos=np.load(os.join(parent_dir, 'pos.npy'))
	## converts neuron position in microns
	neuron_pos=lpn.correctNeuronPos(neuron_pos, resolution)
	
6. **Running the Analysis:**

.. code-block:: python
	
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
	


	



