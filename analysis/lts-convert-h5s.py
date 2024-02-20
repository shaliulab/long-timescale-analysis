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
base_paths = ["/Genomics/ayroleslab2/scott/lts_data"]
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
        filenames = glob.glob(base_path + "/**/*.h5")
        filenames = natsort.natsorted(filenames)
        filename = filenames[number]
        logger.info(filename)
        with h5py.File(filename, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
        logger.info("Loaded tracks...")

        locations=trx_utils.interpolate(locations, node_names)
        locations = trx_utils.fill_nan_median(locations)
        locations = trx_utils.smoothen(locations, window=5)
        folder = Path(filename).parent.name
        final_folder_path = Path(parameters.projectPath) / "for_grace" / folder
        final_folder_path.mkdir(parents=True, exist_ok=True)
        path = final_folder_path / f"{Path(filename).stem}-processed-tracks.h5"
        with h5py.File(
            path,
            "w",
        ) as f:
            dset = f.create_dataset("tracks", data=locations, compression="lzf")
