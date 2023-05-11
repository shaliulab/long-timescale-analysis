import h5py
import numpy as np
import matplotlib.pyplot as plt

example_file = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam1/cam1_20220217_0through190_cam1_20220217_0through190_100-tracked.analysis.h5"
with h5py.File(example_file, "r") as f:
    node_names = [n.decode() for n in f["node_names"][:]]

projections = h5py
node_num = 10
start_time = 10000
length = 2000

f_ls = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian/Wavelets/20220217-lts-cam1_day1_24hourvars-0-pcaModes-wavelets.mat"
)
wavelets_ls = f_ls["wavelets"][
    (start_time) : (start_time + length), 25 * node_num : 25 * (node_num + 1)
]

plt.close()
plt.imshow(wavelets_ls.T, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar()
plt.savefig("tmp_ls.png", dpi=300)

f = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-day1-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/20220217-lts-cam1_day1_24hourvars-0-pcaModes-wavelets.mat"
)
wavelets = f["wavelets"][
    (start_time) : (start_time + length), 25 * node_num : 25 * (node_num + 1)
]
plt.close()
plt.imshow(wavelets.T, aspect="auto", origin="lower", cmap="viridis")
plt.colorbar()
plt.savefig("tmp_morlet.png", dpi=300)
plt.close()

import seaborn as sns

f_projs = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian/Projections/20220217-lts-cam1_day1_24hourvars-0-pcaModes.mat"
)
projs = f_projs["projections"][:, (start_time) : (start_time + length)]
sns.lineplot(np.linspace(0, 1, projs.shape[1]), projs[10, :], color="red")
plt.savefig("tmp.png", dpi=300)
