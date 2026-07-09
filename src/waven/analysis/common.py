"""Shared imports for analysis modules.

This module centralizes the heavy scientific stack used by the legacy analysis
functions. The split modules import from here so moving functions did not change
runtime dependencies or numerical behavior.
"""
import gc
import math
import os
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor

import cv2
import cv2 as cv
import matplotlib

if os.environ.get("waven_NO_PLOTS") == "1":
    matplotlib.use("Agg", force=True)
else:
    matplotlib.use("TkAgg", force=True)

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
import skimage
import tifffile
import torch
from joblib import Parallel, delayed
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from numba import njit, prange
from numpy import exp
from scipy import interpolate, ndimage, signal
from scipy.fftpack import fft, ifft
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, differential_evolution
from scipy.sparse.linalg import svds
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_2d, pearsonr, skew
from skimage import transform
from sklearn.cluster import KMeans, spectral_clustering
from sklearn.decomposition import NMF
from sklearn.feature_extraction import image
from sklearn.metrics import explained_variance_score, r2_score

from ..config import coarse_grid_dimensions, coarse_to_full_scale
from ..runtime.performance import gpu_neuron_chunk_size, model_parallel_jobs
from ..storage.array_store import load_array

