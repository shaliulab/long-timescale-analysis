from awkde import GaussianKDE
import numpy as np
import numpy as np
import utils.motionmapperpy.motionmapperpy as mmpy
import matplotlib.pyplot as plt
import h5py
from multiprocessing import Pool
import random


training_embeddings_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/training_embedding.mat"
wshedfile = h5py.File(training_embeddings_file, "r")

zValues = wshedfile["trainingEmbedding"][:].T
alpha_range = np.arange(0.5, 1.1, 0.1)  # Range of alpha values
glob_bw_range = np.arange(0.04, 0.0625, 0.0025)  # Range of glob_bw values


def process_combination(alpha, glob_bw):
    kde = GaussianKDE(glob_bw=glob_bw, alpha=alpha, diag_cov=True)
    sample = zValues
    kde.fit(sample)

    minx, maxx = np.amin(sample[:, 0]), np.amax(sample[:, 0])
    miny, maxy = np.amin(sample[:, 1]), np.amax(sample[:, 1])

    x = np.linspace(minx, maxx, 625)
    y = np.linspace(miny, maxy, 625)

    XX, YY = np.meshgrid(x, y)
    grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T

    zz = kde.predict(grid_pts)
    ZZ = zz.reshape(XX.shape)

    dx2, dy2 = (x[1] - x[0]) / 2.0, (y[1] - y[0]) / 2.0
    bx = np.concatenate((x - dx2, [x[-1] + dx2]))
    by = np.concatenate((y - dy2, [y[-1] + dy2]))

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.pcolormesh(bx, by, ZZ, cmap=mmpy.gencmap())
    fig.savefig(
        f"figures/tmp/training_embeddings_density_alpha{round(alpha,3)}_globbw{round(glob_bw,3)}.png"
    )
    plt.close(fig)


# Create a list of parameter combinations for the grid search
param_combinations = [
    (alpha, glob_bw) for alpha in alpha_range for glob_bw in glob_bw_range
]
# param_combinations = [(1, 0.05)]

# Create a Pool of worker processes
with Pool() as pool:
    # Use starmap to pass each parameter combination to the process_combination function
    pool.starmap(process_combination, param_combinations)
