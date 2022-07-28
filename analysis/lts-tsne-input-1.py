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
import glob, os, pickle
from datetime import datetime
import numpy as np
from scipy.io import loadmat, savemat
import hdf5storage
import utils.motionmapperpy.motionmapperpy as mmpy
from pathlib import Path
import natsort
import argparse

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("analysis_logger")

# video_filenames = ["data/videos/0through23_cam1.mp4","data/videos/0through23_cam2.mp4"]

frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px
base_paths = [ '/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam3']
# %%sb

parser = argparse.ArgumentParser(description='Bulk embeddings')
parser.add_argument("--number",type=int, help="what file tho?")

if __name__ == '__main__':
    args = parser.parse_args()
    number = args.number
    for base_path in base_paths:
        filenames = glob.glob(base_path + '/*.h5')
        filenames = natsort.natsorted(filenames)
        # logger.info(filenames)
        filename=filenames[number]
        logger.info(filename)
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
        # We want the proboscis to capture the deviation from the head -- so here we replace nans
        head_prob_interp = np.where(np.isnan(locations[:,node_names.index('proboscis'),:,:]), locations[:,node_names.index('head'),:,:],locations[:,node_names.index('proboscis'),:,:])
        locations[:,node_names.index('proboscis'),:,:] = head_prob_interp
        # TODO: force y to 0
        # %%


        projectPath = "mmpy_lts_3h"
        mmpy.createProjectDirectory(projectPath)
        instance_count = 4
        for i in range(instance_count):
            data = locations[
                :, :, :, i
            ]  # 0:int(24*60*60*frame_rate)
            data = trx_utils.smooth_median(data, window=5)

            data = trx_utils.normalize_to_egocentric(
                x=data, ctr_ind=node_names.index('thorax'), fwd_ind=node_names.index('head')
            )
            # data = data[np.random.randint(0,data.shape[0],size=32000),:]
            data = np.delete(data, [node_names.index("thorax"),node_names.index("head"),node_names.index("eyeR"),node_names.index("eyeL")],axis = 1)

            print(data.shape)
            mask = np.all(np.isnan(data[:,:,0]) | np.equal(data[:,:,0], 0), axis=1)
            data = data[~mask,:,:]
            print(data.shape)
            data[~np.isfinite(data)] = 1e-12
            data[data == 0] = 1e-12
            # vels = trx_utils.instance_node_velocities(data, 0, data.shape[0]).astype(np.float32)
            # mask = (vels > .1*px_mm).any(axis=1)
            # data = data[mask,:]
            data = data.reshape((data.shape[0], 2 * data.shape[1]))
            # print(data.shape)

            with h5py.File(f'{projectPath}/Projections/{Path(Path(filename).stem).stem}-{i}-pcaModes.mat', 'w') as f:
                dset = f.create_dataset('projections', data=data.T)
            # savemat(
            #     f'{projectPath}/Projections/{Path(filename).stem}-{i}-pcaModes.mat',
            #     {"projections": data},
            # )
