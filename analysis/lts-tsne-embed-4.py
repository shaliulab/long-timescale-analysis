

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
import argparse
parser = argparse.ArgumentParser(description='Bulk embeddings')
parser.add_argument("--number",type=int)

if __name__ == '__main__':
    args = parser.parse_args()
    # i = args.number

    frame_rate = 100
    projectPath = "mmpy_lts_3h"

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

    parameters.samplingFreq = 100  #% Sampling frequency (or FPS) of data.

    parameters.numPeriods = 25  #% No. of frequencies between minF and maxF.

    parameters.numProcessors = -1 #% No. of processor to use when parallel
    #% processing (for wavelets, if not using GPU). -1 to use all cores.

    parameters.useGPU = -1  # GPU to use, set to -1 if GPU not present

    parameters.training_numPoints=8000      #% Number of points in mini-tSNEs.

    # %%%%% NO NEED TO CHANGE THESE UNLESS RAM (NOT GPU) MEMORY ERRORS RAISED%%%%%%%%%%
    parameters.trainingSetSize=64000        #% Total number of representative points to find. Increase or decrease based on
                                            #% available RAM. For reference, 36k is a good number with 64GB RAM.

    parameters.embedding_batchSize = 32000  #% Lower this if you get a memory error when re-embedding points on learned
                                            #% tSNE map.

    projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)
    with h5py.File(projectionFiles[args.number], "r") as f:
        m = f['projections'][:].T
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

    with h5py.File(tsnefolder + "training_data.mat", "r") as hfile:
        trainingSetData = hfile["trainingSetData"][:].T
    trainingSetData[~np.isfinite(trainingSetData)] = 1e-12
    trainingSetData[trainingSetData == 0] = 1e-12


    with h5py.File(tsnefolder + "training_embedding.mat", "r") as hfile:
        trainingEmbedding = hfile["trainingEmbedding"][:].T


    if parameters.method == "TSNE":
        zValstr = "zVals" if parameters.waveletDecomp else "zValsProjs"
    else:
        zValstr = "uVals"


    print("Finding Embeddings")
    # print("%i/%i : %s" % (args.number + 1, len(projectionFiles), projectionFiles[i]))
    if os.path.exists(projectionFiles[args.number][:-4] + "_%s.mat" % (zValstr)):
        print("Already done. Skipping.\n")
        exit
    with h5py.File(projectionFiles[args.number], "r") as f:
        projections = f['projections'][:].T

    projections[~np.isfinite(projections)] = 1e-12
    projections[projections == 0] = 1e-12

    zValues, outputStatistics = mmpy.findEmbeddings(
        projections, trainingSetData, trainingEmbedding, parameters, projectionFiles[args.number]
    )
    hdf5storage.write(
        data={"zValues": zValues},
        path="/",
        truncate_existing=True,
        filename=projectionFiles[args.number][:-4] + "_%s.mat" % (zValstr),
        store_python_metadata=False,
        matlab_compatible=True,
    )
    with open(
        projectionFiles[args.number][:-4] + "_%s_outputStatistics.pkl" % (zValstr), "wb"
    ) as hfile:
        pickle.dump(outputStatistics, hfile)
    print("Embeddings saved.\n")
    del zValues, projections, outputStatistics

