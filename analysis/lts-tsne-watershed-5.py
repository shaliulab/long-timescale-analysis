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

projectPath = "mmpy_lts_3h"
frame_rate = 100

parameters = mmpy.setRunParameters()

# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.projectPath = projectPath
parameters.method = "TSNE"

parameters.waveletDecomp = True  #% Whether to do wavelet decomposition. If False, PCA projections are used for

#% tSNE embedding.

parameters.minF = 1  #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 50  #% Maximum frequency for Morlet Wavelet Transform,
#% equal to Nyquist frequency for your measurements.

parameters.perplexity = 32
parameters.training_perplexity = 32
parameters.maxNeighbors = 5

parameters.samplingFreq = frame_rate  #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 25  #% No. of frequencies between minF and maxF.

parameters.numProcessors = -1 #% No. of processor to use when parallel
#% processing (for wavelets, if not using GPU). -1 to use all cores.

parameters.useGPU = 0  # GPU to use, set to -1 if GPU not present

parameters.training_numPoints=8000      #% Number of points in mini-tSNEs.

# %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
parameters.trainingSetSize=64000        #% Total number of representative points to find. Increase or decrease based on
                                        #% available RAM. For reference, 36k is a good number with 64GB RAM.

parameters.embedding_batchSize = 32000  #% Lower this if you get a memory error when re-embedding points on learned
                                        #% tSNE map.

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
    minimum_regions=50,
    startsigma=3,
    # pThreshold=[0.33, .67],
    saveplot=True,
    endident="*-pcaModes.mat",
)
