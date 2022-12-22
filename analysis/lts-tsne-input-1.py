# %% [markdown]
# t-SNE
# Remove eyes
# Instead of linearly interpolating, replace all nans with coordinate of head point

# %%
import sys

sys.path.append("..")

import argparse
import glob
import logging
from pathlib import Path

import h5py
import natsort
import numpy as np
import utils.motionmapperpy.motionmapperpy as mmpy
import utils.trx_utils as trx_utils

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("analysis_logger")

# video_filenames = ["data/videos/0through23_cam1.mp4","data/videos/0through23_cam2.mp4"]

frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px
base_paths = ["/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis"]
# %%sb

example_file = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam1/cam1_20220217_0through190_cam1_20220217_0through190_100-tracked.analysis.h5"
with h5py.File(example_file, "r") as f:
    node_names = [n.decode() for n in f["node_names"][:]]

parameters = mmpy.setRunParameters()
mmpy.createProjectDirectory(parameters.projectPath)

parser = argparse.ArgumentParser(description="Bulk embeddings")
parser.add_argument("--number", type=int, help="what file tho?")

if __name__ == "__main__":
    logger.info("Starting...")
    args = parser.parse_args()
    number = args.number
    for base_path in base_paths:
        filenames = glob.glob(base_path + "/*vars.h5")
        filenames = natsort.natsorted(filenames)
        filename = filenames[number]
        logger.info(filename)
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            locations = trx_utils.fill_missing(locations, kind="pchip")
            locations = trx_utils.smooth_median(locations, window=5)
        logger.info("Loaded tracks...")
        # We want the proboscis to capture the deviation from the head -- so here we replace nans
        head_prob_interp = np.where(
            np.isnan(locations[:, node_names.index("proboscis"), :, :]),
            locations[:, node_names.index("head"), :, :],
            locations[:, node_names.index("proboscis"), :, :],
        )
        locations[:, node_names.index("proboscis"), :, :] = head_prob_interp

        instance_count = 4
        for i in range(instance_count):
            data = locations[:, :, :, i]

            data = trx_utils.normalize_to_egocentric(
                x=data,
                ctr_ind=node_names.index("thorax"),
                fwd_ind=node_names.index("head"),
            )
            data = np.delete(
                data,
                [
                    node_names.index("thorax"),
                    node_names.index("head"),
                ],
                axis=1,
            )

            logger.info("Shape before masking: %s", data.shape)
            mask = np.all(np.isnan(data[:, :, 0]) | np.equal(data[:, :, 0], 0), axis=1)
            # with h5py.File(
            #     f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-{i}-missing-data-indices.h5",
            #     "w",
            # ) as f:
            #     dset = f.create_dataset("missing_data_indices", data=mask, compression="lzf")

            # data = data[~mask, :, :]
            # logger.info("Shape after masking: %s", data.shape)
            if data.shape[0] == 0:
                continue
            data[~np.isfinite(data)] = 1e-12
            data[data == 0] = 1e-12
            # vels = trx_utils.instance_node_velocities(data, 0, data.shape[0]).astype(np.float32)
            # mask = (vels > .1*px_mm).any(axis=1)
            # data = data[mask,:]
            reshaped_data = data.reshape((data.shape[0], 2 * data.shape[1]))
            logger.info("Writing fly number %s to file...", i)
            with h5py.File(
                f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-{i}-pcaModes.mat",
                "w",
            ) as f:
                dset = f.create_dataset("projections", data=reshaped_data.T, compression="lzf")
            
            logger.info("Writing fly number %s EGO to file...", i)
            with h5py.File(
                f"{parameters.projectPath}/Ego/{Path(Path(filename).stem).stem}-{i}-pcaModes.h5",
                "w",
            ) as f:
                dset = f.create_dataset("tracks", data=data.T, compression="lzf")
                dset = f.create_dataset("missing_data_indices", data=mask, compression="lzf")
            # savemat(
            #     f'{projectPath}/Projections/{Path(filename).stem}-{i}-pcaModes.mat',
            #     {"projections": data},
            # )
