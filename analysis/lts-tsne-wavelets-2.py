import sys

sys.path.append("..")
import argparse
import glob
import os
from datetime import datetime

import h5py
import natsort
import utils.motionmapperpy.motionmapperpy as mmpy

parser = argparse.ArgumentParser(description="Bulk wavelets")
parser.add_argument("--number", type=int)

if __name__ == "__main__":
    args = parser.parse_args()
    i = args.number
    parameters = mmpy.setRunParameters()
    # parameters.projectPath = "20230426-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-medianwin5-gaussian"
    projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
    projectionFiles = natsort.natsorted(projectionFiles)
    print(projectionFiles[0])
    with h5py.File(projectionFiles[0], "r") as f:
        m = f["projections"][:].T

    # %%%%%
    print(m.shape)
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

    if not os.path.exists(tsnefolder + "training_tsne_embedding.mat"):
        print("Calculating wavelets...")
        mmpy.get_wavelets(projectionFiles, parameters, i, ls=True)
        print(datetime.now().strftime("%m-%d-%Y_%H-%M"))
