import sys

sys.path.append("..")

import argparse
import glob
import os
import pickle
from datetime import datetime

import h5py
import hdf5storage
import natsort
import numpy as np
import utils.motionmapperpy.motionmapperpy as mmpy

parser = argparse.ArgumentParser(description="Bulk embeddings")
parser.add_argument("--number", type=int)


def main():
    print(f"Starting {datetime.now().strftime('%m-%d-%Y_%H-%M')}")
    args = parser.parse_args()
    # i = args.number

    parameters = mmpy.setRunParameters()

    projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)
    with h5py.File(projectionFiles[args.number], "r") as f:
        m = f["projections"][:].T
    # %%%%%
    parameters.pcaModes = m.shape[1]  # %Number of PCA projections in saved files.
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

    print(f"Finding Embeddings for {projectionFiles[args.number]}")
    # print("%i/%i : %s" % (args.number + 1, len(projectionFiles), projectionFiles[i]))
    if os.path.exists(projectionFiles[args.number][:-4] + "_%s.mat" % (zValstr)):
        print("Already done. Skipping.\n")
        exit
    with h5py.File(projectionFiles[args.number], "r") as f:
        projections = f["projections"][:].T

    projections[~np.isfinite(projections)] = 1e-12
    projections[projections == 0] = 1e-12
    print("Finding Embeddings...")
    zValues, outputStatistics = mmpy.findEmbeddings(
        projections,
        trainingSetData,
        trainingEmbedding,
        parameters,
        projectionFiles[args.number],
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


if __name__ == "__main__":
    main()
