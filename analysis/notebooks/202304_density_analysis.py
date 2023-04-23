import h5py
import numpy as np
import utils.motionmapperpy.motionmapperpy as mmpy

results_path = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230409-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/TSNE/20230418_sigma1_3_minregions75_zVals_wShed_groups.mat"
results = h5py.File(results_path, "r")
density = results["density"][:]
density[density > 0.005] = 0.005
density[density < 0.0005] = 0.0

import matplotlib.pyplot as plt
import seaborn as sns

plt.close()
# sns.histplot(density.flatten(), bins=1000)
# plt.yscale("log")
plt.imshow(density.T, cmap=mmpy.gencmap(), origin="lower")
plt.savefig("tmp.png")
