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
import pandas as pd
import utils.motionmapperpy.motionmapperpy as mmpy
import utils.trx_utils as trx_utils

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


logger = logging.getLogger("analysis_logger")
logger.info("Starting... version v0.0.1")

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
        filenames = glob.glob(base_path + "/*.h5")
        filenames = natsort.natsorted(filenames)
        filename = filenames[number]

        logger.info(filename)
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:]
            if locations.shape[0] < 100:
                locations = locations.T
        logger.info("Loaded tracks...")

        locations[:, 0:13, :] = trx_utils.fill_missing(
            locations[:, 0:13, :], kind="pchip", limit=5
        )

        # logger.info("Filled missing data with median...")
        locations[:, node_names.index("head"), :] = trx_utils.fill_missing(
            locations[:, node_names.index("head"), :], kind="pchip"
        )
        locations[:, node_names.index("thorax"), :] = trx_utils.fill_missing(
            locations[:, node_names.index("thorax"), :], kind="pchip"
        )

        head_prob_interp = np.where(
            np.isnan(locations[:, node_names.index("proboscis"), :]),
            locations[:, node_names.index("head"), :],
            locations[:, node_names.index("proboscis"), :],
        )
        locations[:, node_names.index("proboscis"), :] = head_prob_interp

        # TODO: Not needed with lomb-scargle
        # logging.info("Filling missing data with median...")
        # locations = trx_utils.fill_nan_median(locations)
        locations = trx_utils.smooth_median(locations, window=5)
        locations = trx_utils.smooth_gaussian(locations)
        # # TODO: Re
        logger.info(f"Processing individual in file {filename}!")
        data = locations

        data = trx_utils.normalize_to_egocentric(
            x=data,
            ctr_ind=node_names.index("thorax"),
            fwd_ind=node_names.index("head"),
            fill=False,
        )
        logger.info(
            "Setting proboscis y-coordinate to 0 if less than 0.5 -- post egocentrizing..."
        )
        prob_y = data[:, node_names.index("proboscis"), 1]
        prob_y[prob_y < 0.5] = 0
        data[:, node_names.index("proboscis"), 1] = prob_y

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

        if data.shape[0] == 0:
            continue

        reshaped_data = data.reshape((data.shape[0], 2 * data.shape[1]))
        # reshaped_data = reshaped_data[:, 0:23]
        logger.info("Shape after masking: %s", reshaped_data.shape)
        logger.info("Writing...")
        with h5py.File(
            f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-pcaModes.mat",
            "w",
        ) as f:
            dset = f.create_dataset(
                "projections", data=reshaped_data.T, compression="lzf"
            )

        # edge_file = (
        #     "../data/edge/"
        #     + "-".join(Path(filename).stem.split("-")[:3])
        #     + "_edge.mat"
        # )
        # with h5py.File(edge_file, "r") as hfile:
        #     edge_mask = np.append([False], hfile["edger"][:].T[:, i].astype(bool))

        logger.info("Writing fly number %s EGO to file...")
        with h5py.File(
            f"{parameters.projectPath}/Ego/{Path(Path(filename).stem).stem}-pcaModes.h5",
            "w",
        ) as f:
            dset = f.create_dataset("tracks", data=data.T, compression="lzf")
            dset = f.create_dataset(
                "missing_data_indices", data=mask, compression="lzf"
            )
            # dset = f.create_dataset("edge_calls", data=edge_mask, compression="lzf")

        with open("files_processed_complete.txt", "a") as f:
            f.write(f"{filename}\n")
        continue

# %%
