import copy

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import utils.motionmapperpy.motionmapperpy as mmpy
from awkde import GaussianKDE
from skimage.filters import roberts
from skimage.segmentation import watershed


training_embeddings_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230522-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/training_embedding.mat"
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


zValues = zValues[mask, :]

bounds, xx, density = mmpy.findPointDensity(
    zValues,
    numPoints=610,
    sigma=0.95,
    rangeVals=[-70, 70],
)

density_copy = copy.copy(density)


from scipy.ndimage import label
import scipy.io as sio
from skimage.measure import find_contours
import matplotlib.pyplot as plt
import numpy as np
import colorsys
from adjustText import adjust_text
import random

# load mat file
path = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data/map_final.mat"
data = sio.loadmat(path)
BW3 = data["BW3"]

labels, numRegs = label(BW3)


# Function to generate maximally distinct colors
def get_distinct_colors(n):
    hue_partitions = np.linspace(0, 1, n + 1)[:-1]
    colors = [colorsys.hsv_to_rgb(h, 1, 1) for h in hue_partitions]
    return colors


colors = get_distinct_colors(numRegs + 1)
# permute colors
colors = random.shuffle(colors, random_state=0)

fig, ax = plt.subplots(figsize=(10, 10), dpi=300, frameon=False)
plt.imshow(
    density_copy, cmap=mmpy.gencmap(), origin="lower", alpha=0.5
)  # Display the density

# Now, we find contours for each region and add labels
regions = np.unique(labels)

texts = []  # Store all the text objects

for region in regions:
    if region != 0:  # We don't need to consider background as a region
        region_coords = np.where(labels == region)
        center_x = int(np.mean(region_coords[1]))
        center_y = int(np.mean(region_coords[0]))

        txt = plt.text(
            center_x, center_y, str(region), color=colors[region], fontsize=12
        )
        texts.append(txt)

        contours = find_contours(labels == region, 0.5)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], linewidth=1, color=colors[region])


# Adjust the positions of the text to minimize overlaps
adjust_text(
    texts,
    force_points=0.2,
    force_text=0.2,
    expand_points=(1, 1),
    expand_text=(1, 1),
    arrowprops=dict(arrowstyle="-", color=colors, lw=0.5),
)
plt.axis("off")

plt.savefig("figures/tmp/20230525_finalmap_adaptive.png", bbox_inches="tight")
plt.close()

bounds = getDensityBounds(density)
xx = np.linspace(-70, 70, 610)
wbounds = np.where(roberts(labels).astype("bool"))

wbounds = (wbounds[1], wbounds[0])
fig, ax = plt.subplots()
ax.imshow(density_copy, origin="lower", cmap=gencmap())
ax.scatter(wbounds[0], wbounds[1], color="k", s=0.1)
plt.imshow(density_copy, cmap=mmpy.gencmap(), origin="lower")
plt.savefig("figures/tmp/tmp.png")
plt.close()

f = h5py.File("figures/tmp/20230525_finalmap_adaptive.h5", "w")
f.create_dataset("density", data=density)
f.create_dataset("wbounds", data=wbounds)
f.create_dataset("xx", data=xx)
f.create_dataset("LL", data=labels.T)
f.close()


f = h5py.File("figures/tmp/20230525_finalmap_adaptive.h5", "r")
