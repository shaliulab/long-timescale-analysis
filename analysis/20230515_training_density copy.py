from awkde import GaussianKDE
import numpy as np
import numpy as np
import utils.motionmapperpy.motionmapperpy as mmpy
import matplotlib.pyplot as plt
import h5py
from multiprocessing import Pool
import matplotlib as mpl
import random
import copy
from skimage.segmentation import watershed
from skimage.filters import roberts


training_embeddings_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/training_embedding.mat"
wshedfile = h5py.File(training_embeddings_file, "r")

zValues = wshedfile["trainingEmbedding"][:].T


def getDensityBounds(density, thresh=1e-6):
    """
    Get the outline for density maps.
    :param density: m by n density image.
    :param thresh: Density threshold for boundaries. Default 1e-6.
    :return: (p by 2) points outlining density map.
    """
    x_w, y_w = np.where(density > thresh)
    x, inv_inds = np.unique(x_w, return_inverse=True)
    bounds = np.zeros((x.shape[0] * 2 + 1, 2))
    for i in range(x.shape[0]):
        bounds[i, 0] = x[i]
        bounds[i, 1] = np.min(y_w[x_w == bounds[i, 0]])
        bounds[x.shape[0] + i, 0] = x[-i - 1]
        bounds[x.shape[0] + i, 1] = np.max(y_w[x_w == bounds[x.shape[0] + i, 0]])
    bounds[-1] = bounds[0]
    bounds[:, [0, 1]] = bounds[:, [1, 0]]
    return bounds.astype(int)


def gencmap():
    """
    Get behavioral map colormap as a matplotlib colormap instance.
    :return: Matplotlib colormap instance.
    """
    colors = np.zeros((64, 3))
    colors[:21, 0] = np.linspace(1, 0, 21)
    colors[20:43, 0] = np.linspace(0, 1, 23)
    colors[42:, 0] = 1.0

    colors[:21, 1] = np.linspace(1, 0, 21)
    colors[20:43, 1] = np.linspace(0, 1, 23)
    colors[42:, 1] = np.linspace(1, 0, 22)

    colors[:21, 2] = 1.0
    colors[20:43, 2] = np.linspace(1, 0, 23)
    colors[42:, 2] = 0.0
    return mpl.colors.ListedColormap(colors)


# zValues = zValues[np.random.choice(zValues.shape[0], 10000, replace=False), :]
def findPointDensity_awkde(zValues, alpha, glob_bw, numPoints, rangeVals):
    """
    findPointDensity finds a Kernel-estimated PDF from a set of 2D data points
    through convolving with a Gaussian function.
    :param zValues: 2d points of shape (m by 2).
    :param alpha: A smoothing factor for Gaussian KDE.
    :param glob_bw: A bandwidth factor for Gaussian KDE.
    :param numPoints: Output density map dimension (n x n).
    :param rangeVals: 1 x 2 array giving the extrema of the observed range
    :return:
        bounds -> Outline of the density map (k x 2).
        xx -> 1 x numPoints array giving the x and y axis evaluation points.
        density -> numPoints x numPoints array giving the PDF values (n by n) density map.
    """
    # zValues = zValues[np.random.choice(zValues.shape[0], 1000, replace=False), :]
    xx = np.linspace(rangeVals[0], rangeVals[1], numPoints)
    yy = copy.copy(xx)

    # Use the same meshgrid as in the other function
    [XX, YY] = np.meshgrid(xx, yy)
    grid_pts = np.array(list(map(np.ravel, [XX, YY]))).T

    density = compute_density_map(zValues, alpha, glob_bw, grid_pts, numPoints)

    # Normalize density so it sums to 1
    density = density / np.sum(density)

    # Find bounds as in the other function
    bounds = getDensityBounds(density)
    return bounds, xx, density


def compute_density_map(zValues, alpha, glob_bw, grid_pts, numPoints):
    kde = GaussianKDE(glob_bw=glob_bw, alpha=alpha, diag_cov=True)
    kde.fit(zValues)

    zz = kde.predict(grid_pts)
    ZZ = zz.reshape(numPoints, numPoints)  # Reshape back to grid size

    return ZZ  # Return the density map for further processing


bounds, xx, density = findPointDensity_awkde(
    zValues,
    alpha=1,
    glob_bw=0.048,
    numPoints=625,
    rangeVals=[-np.abs(zValues).max() - 15, np.abs(zValues).max() + 15],
)


density_copy = copy.copy(density)
density_copy[density_copy < 9e-6] = 0
wshed = watershed(-density_copy, connectivity=10)
wshed[density_copy < 0.6e-5] = 0

numRegs = len(np.unique(wshed)) - 1
for i, wreg in enumerate(np.unique(wshed)):
    wshed[wshed == wreg] = i
wbounds = np.where(roberts(wshed).astype("bool"))
wbounds = (wbounds[1], wbounds[0])
fig, ax = plt.subplots()
ax.imshow(density_copy, origin="lower", cmap=gencmap())
ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
plt.imshow(density_copy, cmap=mmpy.gencmap(), origin="lower")
plt.savefig("figures/tmp/tmp.png")
plt.close()
