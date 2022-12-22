import h5py
import numpy as np

z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_lts_all_filtered/TSNE/20221207_sigma1_8_minregions40_zVals_wShed_groups_finalsave.mat"

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

for wreg in range(1, wregs.max() + 1):
    ethogram[wreg, np.where(wregs == wreg)[0]] = 1.0

ethogram_split = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

ethogram_split_arr = np.split(ethogram, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

ethogram_dict = {k: v for k, v in zip(z_val_names, ethogram_split)}