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
import natsort
import numpy as np
import utils.trx_utils as trx_utils
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)


def chunker(seq, size):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))


logger = logging.getLogger("analysis_logger")
base_path = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam1"
filenames = glob.glob(base_path + "/*.h5")
filenames = natsort.natsorted(filenames)
logger.info(filenames)
frame_rate = 99.96  # Hz
px_mm = 28.25  # mm/px

# %%
for i, filename_set in tqdm(enumerate(list(chunker(filenames, 12)))):
    logger.info(
        "Building: %s",
        f"20220217-lts-cam1_{((i*12))}through{((i*12)-1) + len(filename_set)}.npy",
    )
    output = None
    for filename in tqdm(filename_set):
        with h5py.File(filename, "r") as f:
            print(filename)
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]
            locations[:, :, 1, :] = -locations[:, :, 1, :]
            assignment_indices, locations, freq = trx_utils.hist_sort(
                locations, ctr_idx=node_names.index("thorax")
            )
            locations[:, :, 1, :] = -locations[:, :, 1, :]
            if isinstance(output, np.ndarray):
                output = np.append(output, locations, axis=0)
            else:
                output = locations
    logger.info("Writing file...")
    np.save(
        f"20220217-lts-cam1_{((i*12))}through{(i*12) + len(filename_set)}.npy", output
    )
