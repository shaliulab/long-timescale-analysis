import logging
from pathlib import Path

import numpy as np
import h5py
import codetiming
import analysis.utils.trx_utils as trx_utils
from .utils import load_node_names, list_files

logger=logging.getLogger(__name__)


def delete_bodyparts(data, deleted_bodyparts, node_names):
    data = np.delete(
        data, [node_names.index(bp) for bp in deleted_bodyparts],
        axis=1,
    )
    return data

def process_locations(locations, node_names, deleted_bodyparts=["thorax", "head"], preprocess=True):
    """
    Args:

        locations (np.array): frames x bodyparts x 2 of float64 dtype
        node_names (list): name of each bodypart in the axis=1 of the locations
        deleted_bodyparts (list): name of bodyparts to be removed from the dat after egocentric projection
        typically these are the body parts that the rest are centered around
        preprocess (bool): If true, missing data will be interpolated (pd.DataFrame.interpolate) and the data will be smoothened
            (median and gaussian filters)

    Returns:
        data (np.array): frames x bodyparts x 2 of float64 dtype after processing
        reshaped_data (np.array) same as data but reshaped to format frames x bodyparts*2.
            The two features of the first body part come first, then those of the second body part, and so on.
    """

    assert not np.isnan(locations).all()

    logger.info("Loaded tracks...")
    logger.info(
        f"Fraction of prob nan before interpolation {np.mean(np.isnan(locations[:, node_names.index('proboscis'), 0]))}"
    )
    logger.info(
        f"Fraction of head nan before interpolation {np.mean(np.isnan(locations[:, node_names.index('head'), 0]))}"
    )

    if preprocess:
        with codetiming.Timer(text="Interpolating: {:.2f} seconds"):
            locations = trx_utils.interpolate(locations, node_names, limit=20)
        with codetiming.Timer(text="Smoothen: {:.2f} seconds"):
            locations = trx_utils.smoothen(locations)

    data = locations


    with codetiming.Timer(text="Egocentric norm: {:.2f} seconds"):
        data = trx_utils.normalize_to_egocentric(
            x=data,
            ctr_ind=node_names.index("thorax"),
            fwd_ind=node_names.index("head"),
            fill=False,
        )

    data=delete_bodyparts(data, deleted_bodyparts, node_names)

    if data.shape[0] == 0:
        return

    reshaped_data = data.reshape((data.shape[0], 2 * data.shape[1]))
    return data, reshaped_data

