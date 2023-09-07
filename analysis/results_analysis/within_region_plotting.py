from collections import defaultdict

import dill
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.style
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["figure.dpi"] = 150
mpl.rcParams["savefig.dpi"] = 600
mpl.rcParams["savefig.transparent"] = True
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
mpl.rcParams["axes.titlesize"] = "xx-large"  # medium, large, x-large, xx-large

mpl.style.use("seaborn-deep")

from itertools import groupby


def encode_list(s_list):
    return [[len(list(group)), key[0]] for key, group in groupby(s_list)]


import sys

sys.path.append("../")
import analysis.utils.trx_utils as trx_utils

filename = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_tracks/20220217-lts-cam1/cam1_20220217_0through190_cam1_20220217_0through190_1-tracked.analysis.h5"
import h5py
import numpy as np

with h5py.File(filename, "r") as f:
    dset_names = list(f.keys())
    node_names = [n.decode() for n in f["node_names"][:]]


z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_lts_1d_subset/TSNE/zVals_wShed_groups_20.mat"
f = h5py.File(z_vals_file, "r")
z_val_names_dset = f["zValNames"]
references = [
    f[z_val_names_dset[dset_idx][0]] for dset_idx in range(z_val_names_dset.shape[0])
]
z_val_names = ["".join(chr(i) for i in obj[:]) for obj in references]
z_lens = [l[0] for l in f["zValLens"][:]]

d = {}
for z_val_name_idx in range(len(z_val_names)):
    d[z_val_names[z_val_name_idx]] = [
        np.sum(z_lens[:(z_val_name_idx)]),
        np.sum(z_lens[: (z_val_name_idx + 1)]),
    ]
d[z_val_names[0]][0] = 0

metafile = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20220217-lts-cam1_day1_24hourvars.h5"


import importlib

importlib.reload(trx_utils)

# videofile = "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_videos/20220217-lts-cam1/20220217-lts-cam1-0000.mp4"
min_length = 25
f = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_lts_1d_subset/TSNE/zVals_wShed_groups_20.mat"
)


with h5py.File(metafile, "r") as hfile:
    vels = hfile["vels"][:]
    print(vels.shape)
running_list = defaultdict(lambda: defaultdict(dict))
for fly_idx in tqdm(range(4)):
    fly_id_mm = fly_idx
    fly_id_trx = fly_idx
    rle_list = encode_list(
        f["watershedRegions"][
            d[f"20220217-lts-cam1_day1_24hourvars-{fly_id_mm}-pcaModes"][0] : d[
                f"20220217-lts-cam1_day1_24hourvars-{fly_id_mm}-pcaModes"
            ][1]
        ]
    )
    dict_rle = {"number": [p[1] for p in rle_list], "length": [p[0] for p in rle_list]}
    df = pd.DataFrame(dict_rle)
    # Get the endasd

    df["end"] = np.cumsum(df.length)
    # Get the start
    df["start"] = df["end"] - df.length

    for region in range(0, np.unique(f["watershedRegions"][:]).shape[0]):
        running_list[fly_idx][region] = list()
        try:
            subset = df[(df.number == region)]
            for section in subset.iterrows():
                # print(section)
                start = section[1]["start"]
                end = section[1]["end"]
                # print(f"Frames: {start},{end}")
                try:
                    running_list[fly_idx][region].extend(
                        vels[fly_idx, start:end].tolist()
                    )
                except:
                    print(f"Failed to append {start},{end}")
        except:
            print(f"Failed to find velocities in {region}")
        # print(f"{fly_idx},{region} completed with {len(running_list)} examples")
output = pd.DataFrame(columns=["fly_idx", "region", "mean_velocity"])
for idx in tqdm(range(len(running_list))):
    for region in range(len(running_list[idx])):
        output.loc[len(output.index)] = [
            idx,
            region,
            np.mean(running_list[idx][region]),
        ]
        print(
            f"{idx},{region} completed with {len(running_list[idx][region])} examples"
        )

output.to_csv("wtf.csv")

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import ConvergenceWarning

cleaned_output = output[~np.isnan(output["mean_velocity"])]
# md = smf.mixedlm("mean_velocity ~ 1 + fly_idx ", cleaned_output, groups=cleaned_output['region'])
# mdf = md.fit(method=["lbfgs"])
model = smf.ols(formula="mean_velocity ~ C(region)", data=cleaned_output)
model_fit = model.fit()
print(model_fit.summary())
# anova_table = sm.stats.anova_lm(model_fit, typ=2)
# print(anova_table)

plt.figure()
cleaned_output.groupby("region").mean()["mean_velocity"].plot(kind="bar")
plt.savefig("test.png")
