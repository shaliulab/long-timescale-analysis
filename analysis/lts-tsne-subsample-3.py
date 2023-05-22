import sys

sys.path.append("..")

import glob
import os
from datetime import datetime

import h5py
import natsort
import utils.motionmapperpy.motionmapperpy as mmpy
from tqdm import tqdm

if __name__ == "__main__":
    parameters = mmpy.setRunParameters()
    # parameters.projectPath = "20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks2"
    projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
    print("Found {} projection files".format(len(projectionFiles)))
    print(f"Project path: {parameters.projectPath}")
    projectionFiles = natsort.natsorted(projectionFiles)
    with h5py.File(projectionFiles[0], "r") as f:
        m = f["projections"][:].T

    # %%%%%
    print(m.shape)
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
