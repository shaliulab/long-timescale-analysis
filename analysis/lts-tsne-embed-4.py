

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

    parameters = mmpy.setRunParameters()

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

