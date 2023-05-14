import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import utils.motionmapperpy.motionmapperpy as mmpy
import utils.motionmapperpy.motionmapperpy.mmutils as mmutils

# results_path = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230409-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/TSNE/20230418_sigma1_3_minregions75_zVals_wShed_groups.mat"
results_path = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian/TSNE/training_embedding.mat"
# results_path = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230426-mmpy-lts-all-headprobinterp-missingness-pchip5-fillnanmedian-medianwin5-gaussian/TSNE/training_embedding.mat"
results = h5py.File(results_path, "r")
zValues = results["trainingEmbedding"][:]
m = np.abs(zValues).max()


sigma = 1.2
_, xx, density = mmpy.findPointDensity(zValues.T, sigma, 611, [-m - 15, m + 15])
# density = results["density"][:]
# density[density > 0.005] = 0.005
# density[density < 0.0005] = 0.0


# plt.close()
# sns.histplot(density.flatten(), bins=1000)
# plt.yscale("log")
# density[density < 1e-3] = 0
plt.imshow(density.T, cmap=mmpy.gencmap(), origin="lower")

plt.savefig("tmp.png")

# # plot distribution of image intensities
# density = h5py.File("/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian/TSNE/20230501_sigma0_45_minregions30_zVals_wShed_groups.mat")["density"][:]
# density[density < 1e-3] = 0
# plt.imshow(density.T, cmap=mmpy.gencmap(), origin="lower")
# plt.savefig("tmp.png")
# plt.close()
# sns.histplot(image.flatten(), bins=1000)
# plt.xlim(0, 0.001)
# plt.ylim(0, 10000)
# # plt.yscale("log")
# plt.savefig("tmp.png")
