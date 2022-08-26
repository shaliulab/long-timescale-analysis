# %% [markdown]
# t-SNE
# Remove eyes
# Instead of linearly interpolating, replace all nans with coordinate of head point

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
import matplotlib as mpl

import utils.motionmapperpy.motionmapperpy.mmutils as mmutils
import utils.trx_utils as trx_utils
# %%
# %matplotlib inline


mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.titlesize"] = "xx-large"  # medium, large, x-large, xx-large
mpl.style.use("seaborn-deep")

import matplotlib.pyplot as plt
bmapcmap = mmutils.gencmap()
_, xx , density = mmutils.findPointDensity()

plt.imshow(density, origin="lower", cmap=bmapcmap)

plt.savefig("test.png")
