import sys
sys.path.append("..")

import logging
from seaborn.distributions import distplot
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import palettable
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import pandas as pd
import joypy
import h5py
import numpy as np
import utils.trx_utils as trx_utils
import glob, os, pickle
from datetime import datetime
import numpy as np
from scipy.io import loadmat, savemat
import hdf5storage
import utils.motionmapperpy.motionmapperpy as mmpy
from pathlib import Path
import natsort

parameters = mmpy.setRunParameters()

projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
projectionFiles = natsort.natsorted(projectionFiles)
m = h5py.File(projectionFiles[0], 'r')['projections']
# %%%%%
parameters.pcaModes = m.shape[1]  #%Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

mmpy.findWatershedRegions(
    parameters,
    minimum_regions=20,
    startsigma=2,
    # pThreshold=[0.33, .67],
    saveplot=True,
    endident="*-pcaModes.mat",
)
