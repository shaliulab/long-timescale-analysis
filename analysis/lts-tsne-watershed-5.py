import sys

sys.path.append("..")

import glob

import h5py
import natsort
import utils.motionmapperpy.motionmapperpy as mmpy


parameters = mmpy.setRunParameters()
projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
projectionFiles = natsort.natsorted(projectionFiles)
m = h5py.File(projectionFiles[0], "r")["projections"]
# %%%%%
parameters.pcaModes = m.shape[1]  # %Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

mmpy.findWatershedRegions(
    parameters,
    minimum_regions=100,
    startsigma=0.5,
    pThreshold=[0.33, 0.67],
    saveplot=True,
    endident="*-pcaModes.mat",
    prev_wshed_file=None #"/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data/20230531_finalmap_adaptive.h5",
)

# %%
