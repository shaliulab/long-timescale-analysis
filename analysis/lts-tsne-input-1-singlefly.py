# %% [markdown]
# t-SNE
# Remove eyes
# Instead of linearly interpolating, replace all nans with coordinate of head point


import argparse
import logging
from analysis.utils.preprocess.compat import process_path, list_files, load_node_names
import joblib
import analysis.utils.motionmapperpy.motionmapperpy as mmpy

logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("analysis_logger")
logger.info("Starting... version v0.0.3")


parser = argparse.ArgumentParser(description="Bulk embeddings")
group=parser.add_mutually_exclusive_group(required=True)
group.add_argument("--number", type=int, default=None, help="what file tho?")
group.add_argument("--animal", type=str, default=None, help="what file tho?")
group.add_argument("--files", type=str, nargs="+", default=None, help="what file tho?")
parser.add_argument("--base-paths", type=str, help="Location of h5 files", nargs="+")
parser.add_argument("--n-jobs", type=int, help="Files processed in parallel")
parser.add_argument("--cache", type=str, default=None)


def main():
    logger.info("Starting...")
    args = parser.parse_args()
    base_paths = args.base_paths
    if base_paths is None:
        base_paths=[None]
    n_jobs = args.n_jobs

    parameters = mmpy.setRunParameters()
    parameters.projectPath="FlyHostel_long_timescale_analysis"


    mmpy.createProjectDirectory(parameters.projectPath)

    if args.files is None:
        filenames = list_files(base_paths[0])
        if args.number is None:
            selector=args.animal
        elif args.files is None:
            selector=args.number

    else:
        filenames=args.files
        selector=0 # not more than 1

    node_names=load_node_names(filenames[0])

    if n_jobs == 1:
        for base_path in base_paths:
            process_path(base_path, selector, node_names, parameters, filenames=filenames, cache=args.cache)
    else:
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                process_path
            )(
                base_path, selector, node_names, parameters, filenames=filenames, cache=args.cache
            )
            for base_path in base_paths
        )






if __name__ == "__main__":
    main()