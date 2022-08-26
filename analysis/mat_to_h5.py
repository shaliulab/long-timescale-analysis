# %% [markdown]
# t-SNE
# Remove eyes
# Instead of linearly interpolating, replace all nans with coordinate of head point

# %%
import sys

sys.path.append("..")
import glob
import logging

import h5py
from pathlib import Path

import natsort
import numpy as np
from tqdm import tqdm

from tqdm import tqdm
import utils.trx_utils as trx_utils

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


logger = logging.getLogger("analysis_logger")
base_path = "/Genomics/ayroleslab2/scott/grace_metrics"
filenames = glob.glob(base_path + "/*.mat")
filenames = natsort.natsorted(filenames)
logger.info(filenames)
frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px

filename = filenames[0]
for filename in tqdm(filenames):
    with h5py.File(filename, "r") as f:
        # logger.info("Keys in file: %s", list(f.keys()))
        locations = f["int_tracks"][:].T
        vels = f["sm_speeds"][:].T

    # logger.info("Writing file...")
    with h5py.File(f"{Path(filename).stem}.h5", "w") as f:
        f.create_dataset("tracks", data=locations.T,compression="lzf")
        f.create_dataset("vels", data=vels.T,compression="lzf")
