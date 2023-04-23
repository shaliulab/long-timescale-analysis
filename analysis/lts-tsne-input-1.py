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
import pathlib
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
        filenames = glob.glob(base_path + "/*vars.h5")
        filenames = natsort.natsorted(filenames)
        filename = filenames[number]

        logger.info(filename)
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
        logger.info("Loaded tracks...")

        metadata = pd.read_csv(
            "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data_index.csv"
        )
        metadata = metadata[metadata["Fly id"].notnull()]
        metadata["date"] = pd.to_datetime(
            metadata["Date"], format="%m/%d/%Y"
        ).dt.strftime("%Y%m%d")
        metadata["within_arena_id"] = (metadata["Fly id"].astype(int) - 1) % 4
        metadata["death_day"] = (metadata["Collapse (hours into video)"] // 24).astype(
            int
        )
        metadata["death_frame_in_death_day"] = (
            (metadata["Collapse (hours into video)"] % 24) * 60 * 60 * 99.96
        ).astype(int)
        metadata["cam_num"] = metadata["Camera"].str.replace("Camera ", "").astype(int)

        # get day from filename
        day = int(pathlib.Path(filename).name.split("_")[1].replace("day", ""))
        cam_number = int(
            pathlib.Path(filename).name.split("_")[0].split("-")[2].replace("cam", "")
        )
        date = pathlib.Path(filename).name.split("-")[0]

        locations[:, 0:13, :] = trx_utils.fill_missing(
            locations[:, 0:13, :], kind="pchip", limit=5
        )
        # logger.info("Filled missing data with median...")
        locations[:, node_names.index("head"), :, :] = trx_utils.fill_missing(
            locations[:, node_names.index("head"), :, :], kind="pchip"
        )
        locations[:, node_names.index("thorax"), :, :] = trx_utils.fill_missing(
            locations[:, node_names.index("thorax"), :, :], kind="pchip"
        )

        head_prob_interp = np.where(
            np.isnan(locations[:, node_names.index("proboscis"), :, :]),
            locations[:, node_names.index("head"), :, :],
            locations[:, node_names.index("proboscis"), :, :],
        )
        locations[:, node_names.index("proboscis"), :, :] = head_prob_interp

        # TODO: Not needed with lomb-scargle
        # locations = trx_utils.fill_nan_median(locations)
        locations = trx_utils.smooth_median(locations, window=5)
        locations = trx_utils.smooth_gaussian(locations)
        instance_count = 4

        matching_rows = metadata[
            (metadata["date"].astype(str) == date) & (metadata["cam_num"] == cam_number)
        ]
        if len(matching_rows) == 0:
            raise Exception(f"No matching rows! Something is wrong with {filename}!")

        for i in range(instance_count):
            # with open('files_processed.txt', 'a') as f:
            #     f.write(f"{filename}, {i}\n")
            # continue
            logger.info(f"Processing individual {i} in file {filename}!")
            data = locations[:, :, :, i]
            matching_row = matching_rows[matching_rows["within_arena_id"] == i]
            logger.info(f"Matching metadata: {matching_row}")
            if len(matching_rows) == 0:
                raise Exception(
                    f"No matching rows! Something is wrong with {filename} on individual {i}!"
                )
            if day > (matching_row["death_day"].values[0] + 1):
                logger.info("Skipping because it's after the death day!")
                continue
            elif day == (matching_row["death_day"].values[0] + 1):
                cutoff = matching_row["death_frame_in_death_day"].values[0]
                if cutoff == 0:
                    logger.info("This day the cutoff at 0. Skipping!")
                    continue
                logger.info(
                    f"Cutting off at {cutoff} for individual {i} in file {filename}!"
                )
                data = data[:cutoff, :, :]
            else:
                logger.info(f"Using all frames for individual {i} in file {filename}!")
                pass

            with h5py.File(
                f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-{i}-processed-tracks.h5",
                "w",
            ) as f:
                dset = f.create_dataset("tracks", data=data, compression="lzf")

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
            # missingness_mask = np.sum(np.isnan(data[:, :, 0]) | np.equal(data[:, :, 0], 0), axis=1) > 6
            # with h5py.File(
            #     f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-{i}-missing-data-indices.h5",
            #     "w",
            # ) as f:
            #     dset = f.create_dataset("missing_data_indices", data=mask, compression="lzf")

            # data = data[~mask, :, :]
            # logger.info("Shape after masking: %s", data.shape)
            if data.shape[0] == 0:
                continue
            # After egocentrizing
            # vels = trx_utils.instance_node_velocities(data, 0, data.shape[0]).astype(np.float32)
            # mask = (vels > .1*px_mm).any(axis=1)
            # data = data[mask,:] = np.nan
            # data = trx_utils.fill_missing(data, kind="pchip", limit=10)

            # data[~np.isfinite(data)] = 0
            # data[data == 0] = 1e-12
            reshaped_data = data.reshape((data.shape[0], 2 * data.shape[1]))
            logger.info("Writing fly number %s to file...", i)
            with h5py.File(
                f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-{i}-pcaModes.mat",
                "w",
            ) as f:
                dset = f.create_dataset(
                    "projections", data=reshaped_data.T, compression="lzf"
                )

            edge_file = (
                "../data/edge/"
                + "-".join(Path(filename).stem.split("-")[:3])
                + "_edge.mat"
            )
            with h5py.File(edge_file, "r") as hfile:
                edge_mask = np.append([False], hfile["edger"][:].T[:, i].astype(bool))

            logger.info("Writing fly number %s EGO to file...", i)
            with h5py.File(
                f"{parameters.projectPath}/Ego/{Path(Path(filename).stem).stem}-{i}-pcaModes.h5",
                "w",
            ) as f:
                dset = f.create_dataset("tracks", data=data.T, compression="lzf")
                dset = f.create_dataset(
                    "missing_data_indices", data=mask, compression="lzf"
                )
                dset = f.create_dataset("edge_calls", data=edge_mask, compression="lzf")
            # savemat(
            #     f'{projectPath}/Projections/{Path(filename).stem}-{i}-pcaModes.mat',
            #     {"projections": data},
            # )
            with open("files_processed_complete.txt", "a") as f:
                f.write(f"{filename}, {i}\n")
            continue

# %%
