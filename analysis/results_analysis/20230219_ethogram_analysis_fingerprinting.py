import time
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_lts_all_filtered/TSNE/20221130_sigma1_55_regions50_zVals_wShed_groups.mat"
z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230522-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/20230523_minregions100_zVals_wShed_groups_prevregions.mat"
with h5py.File(z_vals_file, "r") as f:
    z_val_names_dset = f["zValNames"]
    references = [
        f[z_val_names_dset[dset_idx][0]]
        for dset_idx in range(z_val_names_dset.shape[0])
    ]
    z_val_names = ["".join(chr(i) for i in obj[:]) for obj in references]
    z_lens = [l[0] for l in f["zValLens"][:]]

wshedfile = h5py.File(z_vals_file, "r")
wregs = wshedfile["watershedRegions"][:].flatten()
# ethogram = np.zeros((wregs.max() + 1, len(wregs)))

# for wreg in range(1, wregs.max() + 1):
#     ethogram[wreg, np.where(wregs == wreg)[0]] = 1.0
ethogram_split = np.split(wregs, np.cumsum(wshedfile["zValLens"][:].flatten())[:-1])

ethogram_dict = {k: v for k, v in zip(z_val_names, ethogram_split)}

# Ethogram fingerprinting
from collections import defaultdict

output = defaultdict(list)
N_REGIONS = 98
# /Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian/TSNE/20230501_sigma0_45_minregions30_zVals_wShed_groups_finalsave.mat
ROOT_DIR = Path(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230522-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/Wavelets"
)

freqs = None
for file in tqdm(list(ethogram_dict.keys())[0:12]):
    wavelet_file_abs = Path.joinpath(ROOT_DIR, f"{file}-wavelets.mat")
    with h5py.File(wavelet_file_abs, "r") as f:
        wavelet = f["wavelets"][:]
        if freqs is None:
            freqs = f["f"][:]
    for region in np.arange(0, N_REGIONS):
        region_idx = np.where(ethogram_dict[file] == region)[0]
        print(f"Appending output -- region {region}")
        if output[region] == []:
            print(f"Appending output -- region {region} in file {file} -- first time")
            output[region].append(wavelet[region_idx])
            continue
        if (
            output[region] is not []
            and np.concatenate(output[region]).shape[0] > 1000000
        ):
            print(
                f"Skipping region {region} because it already has 10000000 samples -- {np.concatenate(output[region]).shape[0]}"
            )
            continue
        else:
            output[region].append(wavelet[region_idx])


# Setup info
example_file = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam1/cam1_20220217_0through190_cam1_20220217_0through190_100-tracked.analysis.h5"
with h5py.File(example_file, "r") as f:
    node_names = [n.decode() for n in f["node_names"][:]]

node_names = [n for n in node_names if n not in ["thorax", "head"]]
wlet_nodes = np.repeat(node_names, 2)

mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.titlesize"] = "xx-large"  # medium, large, x-large, xx-large
mpl.style.use("seaborn-deep")

# Plot wavelets
for region, list_of_wlets in tqdm(output.items()):
    print(f"Plotting region {region}")
    region_wavelets = np.concatenate(list_of_wlets)
    mean_amps = np.mean(region_wavelets, axis=0)
    split_to_nodes = np.array_split(mean_amps, 24)
    resulting_numpy = np.zeros((24, 25))
    for i, node_mean_amps in enumerate(split_to_nodes):
        resulting_numpy[i, :] = node_mean_amps
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(resulting_numpy, aspect="auto")
    ax.set_xticks([0, 5, 10, 15, 20, 24])
    ax.set_xticklabels(["%0.1f" % freqs[j] for j in [0, 5, 10, 15, 20, 24]])
    ax.set_yticks(np.arange(24))
    ax.set_yticklabels([wlet_nodes[j] for j in np.arange(24)])
    ax.colorbar()
    figures_path = Path("figures/fingerprints/20230522-ls-example/")
    figures_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        figures_path / f"region{region}-wavelet-example-by-part.png",
        dpi=600,
        bbox_inches="tight",
    )
    # color_bar
    plt.close()
