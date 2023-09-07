import h5py
import matplotlib.pyplot as plt
import numpy as np

training_data_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/training_data.mat"
training_data = h5py.File(training_data_file, "r")
normalized_wavelets = training_data["trainingSetData"][:].T

training_amps_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230514-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/training_amps.mat"
training_amps = h5py.File(training_amps_file, "r")
amps = training_amps["trainingSetAmps"][:].T


# plot row wise sum histogram

plot_data = normalized_wavelets * amps
plot_data = np.sum(plot_data, axis=1)
plot_data = plot_data[plot_data < 1e5]
fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
plt.hist(plot_data, bins=100)
plt.xscale("log")
plt.yscale("log")
plt.title("Histogram of sum of wavelet amps")
plt.savefig("figures/wavelet_total_amp.png")
plt.close()
