import sys

sys.path.append("..")

import glob

import h5py
import natsort
import utils.motionmapperpy.motionmapperpy as mmpy

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
