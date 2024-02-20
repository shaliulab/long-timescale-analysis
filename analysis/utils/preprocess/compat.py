import logging
import os.path
from pathlib import Path

import numpy as np
import h5py
import codetiming
import utils.trx_utils as trx_utils
from .utils import load_node_names, list_files
from .main import process_locations

logger=logging.getLogger(__name__)

def process_path(base_path, selector, node_names, parameters, filenames=None, cache=None):

    if parameters.deleted_bodyparts is None:
        deleted_bodyparts=["thorax", "head"]
    else:
        deleted_bodyparts=parameters.deleted_bodyparts



    if filenames is None:
        filenames=list_files(base_path)
        filenames_filtered=[]

        with open("lts.txt", "r") as handle:
            experiments=[line.strip("\n") for line in handle.readlines()]

        if isinstance(selector, int):
            for experiment in experiments:
                filenames_filtered.extend(
                    [filename for filename in filenames if os.path.basename(filename).startswith(experiment)]
                )

            filename=filenames_filtered[selector]
        elif isinstance(selector, str):
            for experiment in experiments:
                filenames_filtered.extend(
                    [filename for filename in filenames if os.path.splitext(os.path.basename(filename))[0] == selector]
                )
            assert len(filenames_filtered)==1
            filename=filenames_filtered[0]
        else:
            raise ValueError("Please pass an integer or a dataset name")
    else:
        experiments=None
        filename=filenames[selector]


    output_file=f"{parameters.projectPath}/Projections/{Path(Path(filename).stem).stem}-pcaModes.mat"
    output_file_ego=f"{parameters.projectPath}/Ego/{Path(Path(filename).stem).stem}-pcaModes.h5"


    if cache is not None:
        final_output_file=os.path.join(cache, f"Projections/{Path(Path(filename).stem).stem}-pcaModes.mat")
        final_output_file_ego=os.path.join(cache, f"Ego/{Path(Path(filename).stem).stem}-pcaModes.mat")
        if os.path.exists(final_output_file) and os.path.exists(final_output_file_ego):
            print(f"Restoring from cache {final_output_file} and {final_output_file_ego}")
            os.symlink(final_output_file, output_file)
            os.symlink(final_output_file_ego, output_file_ego)
            return None

    logger.info("Processing %s", filename)

    node_names_ = load_node_names(filename)

    assert len(node_names_) == len(node_names)
    for i, name in enumerate(node_names_):
        assert name == node_names[i]
        
    logger.info(filename)
    with h5py.File(filename, "r") as f:
        dset_names = list(f.keys())
        locations = f["tracks"][:]
        if locations.shape[0] < 100:
            locations = locations.T

    locations=locations[..., 0]
    if parameters.stride > 1:
        locations=locations[::parameters.stride, :]


    logger.info(f"Processing individual in file {filename}!")
    data, reshaped_data=process_locations(locations, node_names, deleted_bodyparts=deleted_bodyparts, preprocess=parameters.preprocess)
    processed_parts=[part for part in node_names if part not in deleted_bodyparts]


    logger.info("Shape before masking: %s", data.shape)
    mask = np.all(np.isnan(data[:, :, 0]) | np.equal(data[:, :, 0], 0), axis=1)

    logger.info("Shape after masking: %s", reshaped_data.shape)
    logger.info("Writing...")
    with h5py.File(
        output_file,
        "w",
    ) as f:
        dset = f.create_dataset(
            "projections", data=reshaped_data.T, compression="lzf"
        )
        dset = f.create_dataset(
            "node_names", data=processed_parts
        )
    logger.info("Writing fly number %s EGO to file...")
    with h5py.File(
        output_file_ego,
        "w",
    ) as f:
        dset = f.create_dataset("tracks", data=data.T, compression="lzf")
        dset = f.create_dataset(
            "missing_data_indices", data=mask, compression="lzf"
        )
        # dset = f.create_dataset("edge_calls", data=edge_mask, compression="lzf")

    with open("files_processed_complete.txt", "a") as f:
        f.write(f"{filename}\n")
        return
