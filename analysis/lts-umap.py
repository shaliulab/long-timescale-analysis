# %% [markdown]
# t-SNE
# Remove eyes
# Instead of linearly interpolating, replace all nans with coordinate of head point

# %%
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


logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("analysis_logger")
filenames = ["/Genomics/ayroleslab2/scott/git/lts-manuscript/data/cam1_20220217_0through190_cam1_20220217_0through190_1-tracked.analysis.h5"]
# video_filenames = ["data/videos/0through23_cam1.mp4","data/videos/0through23_cam2.mp4"]

frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px

# %%

with h5py.File(filenames[0], "r") as f:
    dset_names = list(f.keys())
    locations = f["tracks"][:].T
    node_names = [n.decode() for n in f["node_names"][:]]
    locations[:,:,1,:] = -locations[:,:,1,:]
    assignment_indices, locations, freq = trx_utils.hist_sort(
        locations, ctr_idx=node_names.index("thorax")
    )
    locations[:,:,1,:] = -locations[:,:,1,:]

# We want the proboscis to capture the deviation from the head -- so here we replace nans
head_prob_interp = np.where(np.isnan(locations[:,node_names.index('proboscis'),:,:]), locations[:,node_names.index('head'),:,:],locations[:,node_names.index('proboscis'),:,:])
locations[:,node_names.index('proboscis'),:,:] = head_prob_interp



print("===locations data shape===")
print(locations.shape)
print()

print("===nodes===")
for i, name in enumerate(node_names):
    print(f"{i}: {name}")

# %%
frame_count, node_count, _, instance_count = locations.shape
locations = trx_utils.fill_missing(locations)

print("frame count:", frame_count)
print("node count:", node_count)
print("instance count:", instance_count)

px_per_mm = 28.25

HEAD_INDEX = node_names.index("head")
THORAX_INDEX = node_names.index("thorax")
ABDO_INDEX = node_names.index("abdomen")

# %%
import glob, os, pickle
from datetime import datetime
import numpy as np
from scipy.io import loadmat, savemat
import hdf5storage
import utils.motionmapperpy.motionmapperpy as mmpy

projectPath = "lts_mm"
mmpy.createProjectDirectory(projectPath)

for i in range(instance_count):
    data = locations[
        : int(0 * 60 * frame_rate), :, :, i
    ]  # 0:int(24*60*60*frame_rate)
    data = trx_utils.smooth_median(data, window=5)

    data = trx_utils.normalize_to_egocentric(
        x=data, ctr_ind=THORAX_INDEX, fwd_ind=HEAD_INDEX
    )
    # data = data[np.random.randint(0,data.shape[0],size=32000),:]
    data = np.delete(data, [node_names.index("thorax"),node_names.index("head"),node_names.index("eyeR"),node_names.index("eyeL")],axis = 1)
    # vels = trx_utils.instance_node_velocities(data, 0, data.shape[0]).astype(np.float32)
    # mask = (vels > .1*px_per_mm).any(axis=1)
    # data = data[mask,:]
    data = data.reshape((data.shape[0], 2 * data.shape[1]))
    print(data.shape)
    savemat(
        projectPath + "/Projections/dataset_%i_pcaModes.mat" % (i + 1),
        {"projections": data},
    )

parameters = mmpy.setRunParameters()

# %%%%%%% PARAMETERS TO CHANGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parameters.projectPath = projectPath
parameters.method = "TSNE"

parameters.waveletDecomp = True  #% Whether to do wavelet decomposition. If False, PCA projections are used for

#% tSNE embedding.

parameters.minF = 0.5  #% Minimum frequency for Morlet Wavelet Transform

parameters.maxF = 50  #% Maximum frequency for Morlet Wavelet Transform,
#% equal to Nyquist frequency for your measurements.

# parameters.perplexity = 5
# parameters.training_perplexity = 20

parameters.samplingFreq = frame_rate  #% Sampling frequency (or FPS) of data.

parameters.numPeriods = 25  #% No. of frequencies between minF and maxF.

parameters.numProcessors = 128  #% No. of processor to use when parallel
#% processing (for wavelets, if not using GPU). -1 to use all cores.

parameters.useGPU = 0  # GPU to use, set to -1 if GPU not present

parameters.training_numPoints = int(
    64000
)  #% Number of points in mini-tSNEs.

# %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
parameters.trainingSetSize = 36000  #% Total number of representative points to find. Increase or decrease based on
#% available RAM. For reference, 36k is a good number with 64GB RAM.

parameters.embedding_batchSize = (
    8000  #% Lower this if you get a memory error when re-embedding points on learned
)
#% tSNE map.


projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")

m = loadmat(projectionFiles[0], variable_names=["projections"])["projections"]

# %%%%%
parameters.pcaModes = m.shape[1]  #%Number of PCA projections in saved files.
parameters.numProjections = parameters.pcaModes
# %%%%%
del m

print(datetime.now().strftime("%m-%d-%Y_%H-%M"))
print("tsneStarted")

if parameters.method == "TSNE":
    if parameters.waveletDecomp:
        tsnefolder = parameters.projectPath + "/TSNE/"
    else:
        tsnefolder = parameters.projectPath + "/TSNE_Projections/"
elif parameters.method == "UMAP":
    tsnefolder = parameters.projectPath + "/UMAP/"

if not os.path.exists(tsnefolder + "training_tsne_embedding.mat"):
    print("Running minitSNE")
    mmpy.subsampled_tsne_from_projections(parameters, parameters.projectPath)
    print("minitSNE done, finding embeddings now.")
    print(datetime.now().strftime("%m-%d-%Y_%H-%M"))

import h5py

with h5py.File(tsnefolder + "training_data.mat", "r") as hfile:
    trainingSetData = hfile["trainingSetData"][:].T


with h5py.File(tsnefolder + "training_embedding.mat", "r") as hfile:
    trainingEmbedding = hfile["trainingEmbedding"][:].T


if parameters.method == "TSNE":
    zValstr = "zVals" if parameters.waveletDecomp else "zValsProjs"
else:
    zValstr = "uVals"


for i in range(len(projectionFiles)):
    print("Finding Embeddings")
    print("%i/%i : %s" % (i + 1, len(projectionFiles), projectionFiles[i]))
    if os.path.exists(projectionFiles[i][:-4] + "_%s.mat" % (zValstr)):
        print("Already done. Skipping.\n")
        continue
    projections = loadmat(projectionFiles[i])["projections"]
    projections[np.isnan(projections)] = 1e-12
    zValues, outputStatistics = mmpy.findEmbeddings(
        projections, trainingSetData, trainingEmbedding, parameters
    )
    hdf5storage.write(
        data={"zValues": zValues},
        path="/",
        truncate_existing=True,
        filename=projectionFiles[i][:-4] + "_%s.mat" % (zValstr),
        store_python_metadata=False,
        matlab_compatible=True,
    )
    with open(
        projectionFiles[i][:-4] + "_%s_outputStatistics.pkl" % (zValstr), "wb"
    ) as hfile:
        pickle.dump(outputStatistics, hfile)
    print("Embeddings saved.\n")
    del zValues, projections, outputStatistics




print("All Embeddings Saved!")

mmpy.findWatershedRegions(
    parameters,
    minimum_regions=150,
    startsigma=0.3,
    pThreshold=[0.33, 0.67],
    saveplot=True,
    endident="*_pcaModes.mat",
)

# %%
