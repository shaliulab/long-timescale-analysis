# %% [markdown]
# t-SNE
# Remove eyes
# Instead of linearly interpolating, replace all nans with coordinate of head point

# %%
import sys

sys.path.append("../")
import glob
import logging
from pathlib import Path

import h5py
import natsort
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("analysis_logger")

# base_path = "/Genomics/ayroleslab2/scott/grace_metrics"
base_path = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis"
filenames = glob.glob(base_path + "/*.mat")
filenames = natsort.natsorted(filenames)
logger.info(filenames)
frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px

filename = filenames[0]
for filename in tqdm(filenames):
    with h5py.File(filename, "r") as f:
        logger.info("Keys in file: %s", list(f.keys()))
        locations = f["day_raw_tracks"][:].T
        logger.info("Loaded tracks...")
        logger.info("Shape of raw locations: %s", locations.shape)

    logger.info("Shape of locations: %s", locations.shape)

    logger.info("Writing file...")
    with h5py.File(f"{Path(filename).stem}.h5", "w") as f:
        f.create_dataset("tracks", data=locations.T, compression="lzf")
