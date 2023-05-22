import sys

sys.path.append("..")

import glob
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path

import h5py
import hdf5storage
import joypy
import matplotlib as mpl
import matplotlib.pyplot as plt
import natsort
import numpy as np
import palettable
import pandas as pd
import seaborn as sns
import utils.motionmapperpy.motionmapperpy as mmpy
import utils.trx_utils as trx_utils
from matplotlib.colors import ListedColormap
from scipy.io import loadmat, savemat
from seaborn.distributions import distplot
from tqdm import tqdm
import importlib

# importlib.reload(mmpy)

parameters = mmpy.setRunParameters()
# parameters.projectPath = "20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian"
projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
projectionFiles = natsort.natsorted(projectionFiles)
m = h5py.File(projectionFiles[0], "r")["projections"]
# %%%%%
parameters.pcaModes = m.shape[1]  # %Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

mmpy.findWatershedRegions_training(
    parameters,
    minimum_regions=100,
    startsigma=1,
    pThreshold=[0.33, 0.67],
    saveplot=True,
    endident="embedding.mat",
)
