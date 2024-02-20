import sys

sys.path.append("..")
import argparse
import glob
import os.path
from datetime import datetime

import h5py
import natsort
import utils.motionmapperpy.motionmapperpy as mmpy

parser = argparse.ArgumentParser(description="Bulk wavelets")
group=parser.add_mutually_exclusive_group(required=True)
group.add_argument("--number", type=int, default=None, help="what file tho?")
group.add_argument("--animal", type=str, default=None, help="what file tho?")
group.add_argument("--files", type=str, default=None, nargs="+")
parser.add_argument("--cache", default=None, type=str, required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    
    parameters = mmpy.setRunParameters()

    if args.files is None:
        with open("lts.txt", "r") as handle:
            experiments=[line.strip("\n") for line in handle.readlines()]

        projectionFiles = glob.glob(parameters.projectPath + "/Projections/*pcaModes.mat")
        projectionFiles = natsort.natsorted(projectionFiles)

        projectionFiles_filtered=[]
        for experiment in experiments:
            projectionFiles_filtered.extend([projectionFile for projectionFile in projectionFiles if os.path.basename(projectionFile).startswith(experiment)])

        projectionFiles=projectionFiles_filtered
    else:
        projectionFiles=args.files

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

    if args.number is None:
        selector=args.animal
    if selector is None and args.files is None:
        selector=args.number
    elif args.files is not None:
        selector=0
    else:
        raise ValueError("Please pass either --number --animal or --files")

    if not os.path.exists(tsnefolder + "training_tsne_embedding.mat"):
        print("Calculating wavelets...")
        mmpy.get_wavelets(projectionFiles, parameters, i=selector, ls=True, cache=args.cache)
        print(datetime.now().strftime("%m-%d-%Y_%H-%M"))
