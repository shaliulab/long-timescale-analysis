import pathlib

import pandas as pd

metadata = pd.read_csv(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data_index.csv"
)
metadata = metadata[metadata["Fly id"].notnull()]
metadata["date"] = pd.to_datetime(metadata["Date"], format="%m/%d/%Y").dt.strftime(
    "%Y%m%d"
)
metadata["within_arena_id"] = (metadata["Fly id"].astype(int) - 1) % 4
metadata["death_day"] = (metadata["Collapse (hours into video)"] // 24).astype(int)
metadata["death_frame_in_death_day"] = (
    (metadata["Collapse (hours into video)"] % 24) * 60 * 60 * 99.96
).astype(int)
metadata["cam_num"] = metadata["Camera"].str.replace("Camera ", "").astype(int)
frames = (
    metadata["death_day"] * 24 * 60 * 60 * 99.96 + metadata["death_frame_in_death_day"]
)

metadata[
    [
        "date",
        "Fly id",
        "Collapse (hours into video)",
        "death_day",
        "death_frame_in_death_day",
    ]
]

filename = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20220312-lts-cam3_day5_24hourvars-3-pcaModes.mat"

# get day from filename
day = int(pathlib.Path(filename).name.split("_")[1].replace("day", ""))
cam_number = int(
    pathlib.Path(filename).name.split("_")[0].split("-")[2].replace("cam", "")
)
date = pathlib.Path(filename).name.split("-")[0]

matching_rows = metadata[
    (metadata["date"].astype(str) == date) & (metadata["cam_num"] == cam_number)
]

id = 3
i = id
matching_row = matching_rows[matching_rows["within_arena_id"] == i]
print(f"Matching metadata: {matching_row}")
if len(matching_rows) == 0:
    raise Exception(
        f"No matching rows! Something is wrong with {filename} on individual {i}!"
    )
if day > (matching_row["death_day"].values[0] + 1):
    print("Skipping because it's after the death day!")
elif day == (matching_row["death_day"].values[0] + 1):
    cutoff = matching_row["death_frame_in_death_day"].values[0]
    if cutoff == 0:
        print("This day the cutoff at 0. Skipping!")
    print(f"Cutting off at {cutoff} for individual {i} in file {filename}!")
else:
    print(f"Using all frames for individual {i} in file {filename}!")
    pass


import h5py
import matplotlib.pyplot as plt

node_num = 4
start_time = 10000
length = 6000

f_ls = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230421-mmpy-lts-all-headprobinterp-missingness-pchip5-medianwin5-gaussian/Wavelets/20220217-lts-cam1_day1_24hourvars-0-pcaModes-wavelets.mat"
)
wavelets_ls = f_ls["wavelets"][
    (start_time) : (start_time + length), 25 * node_num : 25 * (node_num + 1)
]

plt.imshow(wavelets_ls.T, aspect="auto", origin="lower", cmap="viridis")
plt.savefig("tmp_ls.png", dpi=300)

f = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230210-mmpy-lts-day1-headprobinterp-missingness-pchip5-fillnanmedian-setnonfinite0-removegt1missing/Wavelets/20220217-lts-cam1_day1_24hourvars-0-pcaModes-wavelets.mat"
)
wavelets = f["wavelets"][
    (start_time) : (start_time + length), 25 * node_num : 25 * (node_num + 1)
]

plt.imshow(wavelets.T, aspect="auto", origin="lower", cmap="viridis")
plt.savefig("tmp_morlet.png", dpi=300)

import mne
import numpy as np

omega0 = 5
f = 0.01
number_of_cycles = omega0 / (2 * np.pi * 1)

python

import numpy as np
from scipy.signal import lombscargle

minT = 1.0 / 50
maxT = 1.0 / 1
Ts = minT * (2 ** ((np.arange(25) * np.log(maxT / minT)) / (np.log(2) * (25 - 1))))
f = (1.0 / Ts)[::-1]

omega0 = 20
scales = (omega0 + np.sqrt(2 + omega0**2)) / (4 * np.pi * f)
window_sizes = scales * 100
np.round(window_sizes).astype(int)


#


import h5py
import numpy as np

z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230509-mmpy-lts-all-pchip5-headprobinterp-medianwin5-gaussian-lombscargle-win50-singleflysampledtracks-noyprob/UMAP/20230509_sigma2_55_minregions30_zVals_wShed_groups_finalsave.mat"

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
ethogram = np.zeros((wregs.max() + 1, len(wregs)))
ethogram_split = np.split(wregs, np.cumsum(wshedfile["zValLens"][:].flatten())[:-1])
ethogram_dict = {k: v for k, v in zip(z_val_names, ethogram_split)}


import h5py
import numpy as np

# Open the h5 file
file = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/20230515_sigma1_05_minregions100_zVals_wShed_groups_finalsave.mat",
    "r",
)

# Access the dataset
wbounds_dataset = file["wbounds"]

# Extract the object references
object_refs = wbounds_dataset[:]

# Unpack the object references
x_ref = object_refs[0, 0]
y_ref = object_refs[1, 0]

# Access the actual data
x_data = file[x_ref][:]
y_data = file[y_ref][:]

# Close the file
file.close()

# Access the unpacked data
print("x values:", x_data)
print("y values:", y_data)
