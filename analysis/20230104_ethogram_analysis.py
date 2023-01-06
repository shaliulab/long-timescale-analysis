import h5py
import numpy as np
from click import echo

z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_lts_all_filtered/TSNE/20221130_sigma1_55_regions50_zVals_wShed_groups.mat"

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

# for wreg in range(1, wregs.max() + 1):
    # ethogram[wreg, np.where(wregs == wreg)[0]] = 1.0
ethogram_split = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])
# ethogram_split_interp = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

# ethogram_split_arr = np.split(ethogram.T, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

ethogram_dict = {k: v for k, v in zip(z_val_names, ethogram_split)}


# 
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
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

plt.figure(figsize=(30, 10))
plt.plot(ethogram_split_interp[0][0:100000])
plt.savefig("figures/ethogram_split_interp_0to100000.png")

plt.figure(figsize=(30, 10))
plt.plot(ethogram_split_basic[0][:100000])
plt.savefig("figures/ethogram_split_basic_0to100000.png")


#  Noise

from collections import defaultdict
keys = list(ethogram_dict.keys())
cams = [grp.split("-")[2].split("_")[0] for grp in keys]
dates = [grp.split("-")[0] for grp in keys]
fly_nums = [int(grp.split("-")[3]) for grp in keys]
days = [int(grp.split("_")[1][3]) for grp in keys]

videos_dict = {
    "20220217-cam1": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_videos/20220217-lts-cam1/20220217-lts-cam1-0000through0189.mkv",
    "20220217-cam1": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_videos/20220217-lts-cam1/",
    "20220217-cam1": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_videos/20220217-lts-cam1/",
    "20220217-cam1": "/Genomics/ayroleslab2/scott/long-timescale-behavior/data/organized_videos/20220217-lts-cam1/",
}

FPS=99.96


start_time_dict = {
    "20220217": 755*60*FPS,
    "20220312": 1277*60*FPS,
    "20220326": 890*60*FPS,
    "20220418": 846*60*FPS,
}

experimental_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(str)))))

for key, date, cam, fly_num, day in zip(keys, dates, cams, fly_nums, days):
    # if date != "20220217":
        # continue
    experimental_dict[date][cam][fly_num][day]['base_name'] = key
    experimental_dict[date][cam][fly_num][day]['projections'] = f"Projections/{key}.mat"
    experimental_dict[date][cam][fly_num][day]['wavelets'] = f"Wavelets/{key}-wavelets.mat"
    experimental_dict[date][cam][fly_num][day]['egocentric'] = f"Ego/{key}.h5"
    experimental_dict[date][cam][fly_num][day]['zvals']  = ethogram_dict[key]

hour_of_day_wise_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(np.array))))
for date, date_dict in experimental_dict.items():
    for cam, cam_dict in date_dict.items():
        for fly, fly_dict in cam_dict.items():
            # for day, day_dict in fly_dict.items():
            day_keys = sorted(fly_dict.keys(), reverse=False)
            focal_dset_list = []
            
            for key in day_keys:
                focal_dset_list.append(fly_dict[key])
                
            zvals_indiv = [dset['zvals'] for dset in focal_dset_list]
            zvals_arr = np.concatenate(zvals_indiv, axis=0)
            hour_wise = np.array_split(zvals_arr, 24*len(zvals_indiv))
            for i in range(24):
                item_cts = np.unique(np.concatenate(hour_wise[i::24]), return_counts=True)
                item_relative_freq = (item_cts[1] / np.sum(item_cts[1]))*100
                hour_of_day_wise_dict[date][cam][fly][i] = item_relative_freq
                            

plt.figure(figsize=(20, 5))
zt_agg = []
freq_agg = []
for date, date_dict in hour_of_day_wise_dict.items():
    for cam, cam_dict in date_dict.items():
        for fly, fly_dict in cam_dict.items():
            for i in [44]:
                list_of_freqs = []
                for hour, hour_arr in fly_dict.items():
                    list_of_freqs.append(hour_arr[i])
                zt = (np.arange(24) + ((start_time_dict[date]/(60*FPS))/60)) % 24
                list_of_freqs = [x for _, x in sorted(zip(zt, list_of_freqs))]
                zt = sorted(zt)
                print(zt)
                zt_agg.extend(zt)
                freq_agg.extend(list_of_freqs)
                plt.plot(zt , list_of_freqs, label=f"{date}-{cam}-{fly}-{hour}")
freq_agg = [x for _, x in sorted(zip(zt_agg, freq_agg))]
zt_agg = sorted(zt_agg)
sns.histplot(x = zt_agg, y = freq_agg, bins=24, stat="probability")
plt.savefig("figures/hour_of_day_wise_indiv.png")

df = pd.DataFrame(data={'x': zt_agg, 'y': freq_agg})
bins = range(0,25,1)
df['bin'] = pd.cut(df['x'], bins)
agg_df = df.groupby(by='bin').mean()

# this is the important step. We can obtain the interval index from the categorical input using this line.
mids = pd.IntervalIndex(agg_df.index.get_level_values('bin')).mid

# to apply for plots:
plt.figure()
plt.xlabels([])
# plt.yticks([])
sns.barplot(x=mids, y=agg_df['y'],color="#202C59")
sns.despine()
plt.savefig("figures/agg_df.png")

df = pd.DataFrame.from_dict(hour_of_day_wise_dict, orient="index")


df.plot(kind="bar", stacked=True, figsize=(10, 10),legend=None)
plt.savefig("figures/hour_of_day.png")
plt.close()

for i in range(df.shape[0]):
    df.iloc[:, i].plot(kind="bar", stacked=True, figsize=(10, 10),legend=None)
    plt.savefig(f"figures/hour_of_day_region{i}.png")
    plt.close()
    
    
    

import scipy.io as sio
caroline_state_transitions = sio.loadmat("/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/data/behavioral_state_transitions.mat")

caroline_ethogram = caroline_state_transitions['states'][2][0]
plt.figure(figsize=(30, 10))
plt.plot(caroline_ethogram[:100000])
plt.savefig("figures/caroline_plot_0to100000.png")


item_cts = np.unique(caroline_ethogram, return_counts=True)
item_relative_freqs = (item_cts[1] / np.sum(item_cts[1]))*100
item_relative_freqs

item_cts = np.unique(ethogram_split_basic, return_counts=True)
item_relative_freqs = (item_cts[1] / np.sum(item_cts[1]))*100
item_relative_freqs

item_cts = np.unique(ethogram_split_interp, return_counts=True)
item_relative_freqs = (item_cts[1] / np.sum(item_cts[1]))*100
item_relative_freqs

item_relative_freqs[np.where(item_cts[0] == 15)]

# 
import hdf5storage
wshedfile = hdf5storage.loadmat(z_vals_file)

import utils.motionmapperpy.motionmapperpy as mmpy
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt

parameters = mmpy.setRunParameters() 

try:
    tqdm._instances.clear()
except:
    pass
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 1, figsize=(10,10))
zValues = wshedfile['zValues']
m = np.abs(zValues).max()


sigma=1.55
_, xx, density = mmpy.findPointDensity(zValues, sigma, 511, [-m-10, m+10])
axes.imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')
axes.axis('off')
axes.set_title('Method : %s'%parameters.method)
sc = axes.scatter([],[],marker='o', color='k', s=100)

h5ind = 0
tstart = 0
# connections = [np.arange(6,10), np.arange(10,14), np.arange(14,18), np.arange(18,22), np.arange(22,26), np.arange(26,30),
#               [2,0,1],[0,3,4,5], [31,3,30]]


def animate(t):
    t = int(t*99.96)+tstart
#   axes[1].clear()
#   im = axes[1].imshow(clips[h5ind].get_frame(t/clips[h5ind].fps), cmap='Greys', origin='lower')
#   for conn in connections:
#       axes[1].plot(h5s[h5ind][t, conn, 0], h5s[h5ind][t, conn, 1], 'k-')
#   axes[1].axis('off')
    sc.set_offsets(zValues[t])
    return mplfig_to_npimage(fig) #im, ax


anim = VideoClip(animate, duration=60) # will throw memory error for more than 100.
plt.close()
anim.write_videofile('figures/tsne-animation.mp4', fps=20, codec='libx264', bitrate='20m', audio=False)



import utils.motionmapperpy.motionmapperpy as mmpy
from moviepy.editor import VideoClip, VideoFileClip
from moviepy.video.io.bindings import mplfig_to_npimage
import matplotlib.pyplot as plt
import h5py
# z_vals_file =  "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221216_sigma1_55_minregions50_zVals_wShed_groups.mat"
# z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221216_sigma1_05_minregions50_zVals_wShed_groups.mat"
z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221219_sigma1_55_minregions50_zVals_wShed_groups_finalsave.mat"
wshedfile = h5py.File(z_vals_file, "r")
wregs = wshedfile["watershedRegions"][:].flatten()
ethogram_split = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

item_cts_base = np.unique(np.concatenate(ethogram_split), return_counts=True)
item_relative_freq = (item_cts[1] / np.sum(item_cts[1]))*100
fig, ax = plt.subplots(1, 1, figsize=(5,5))
import numpy as np
# training_embeddings_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20221208_mmpy_lts_all_filtered/TSNE/training_embedding.mat"
# wshedfile = h5py.File(training_embeddings_file, "r")
zValues = wshedfile['zValues'][:].T
m = np.abs(zValues).max()

z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20221208_mmpy_lts_all_filtered/Projections/20220418-lts-cam1_day2_24hourvars-1-pcaModes_zVals.mat"
wshedfile = h5py.File(z_vals_file, "r")
zValues = wshedfile["zValues"][:].T
sigma=1.55
_, xx, density = mmpy.findPointDensity(zValues, sigma, 611, [-m-15, m+15])
ax.imshow(density, cmap=mmpy.gencmap(), extent=(xx[0], xx[-1], xx[0], xx[-1]), origin='lower')
ax.axis('off')
# axes[0].set_title('Method : %s'%parameters.method)
sc = ax.scatter([],[],marker='o', color='k', s=500)
plt.savefig("figures/tsne_density.png")



z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221219_sigma1_55_minregions50_zVals_wShed_groups_finalsave_1e5.mat"
wshedfile = h5py.File(z_vals_file, "r")
wregs = wshedfile["watershedRegions"][:].flatten()
ethogram_split = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

item_cts = np.unique(np.concatenate(ethogram_split), return_counts=True)
item_relative_freq = (item_cts[1] / np.sum(item_cts[1]))*100
print(item_relative_freq)

# 1e-100
z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221219_sigma1_55_minregions50_zVals_wShed_groups_finalsave.mat"
wshedfile = h5py.File(z_vals_file, "r")
wregs = wshedfile["watershedRegions"][:].flatten()
ethogram_split = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

item_cts = np.unique(np.concatenate(ethogram_split), return_counts=True)
item_relative_freq = (item_cts[1] / np.sum(item_cts[1]))*100
print(item_relative_freq)


# 
# z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221219_sigma1_55_minregions50_zVals_wShed_groups_finalsave.mat"
# z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221219_sigma1_55_minregions50_zVals_wShed_groups.mat"
# z_vals_file = "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/mmpy_1fly_interp/TSNE/20221219_sigma1_55_minregions50_zVals_wShed_groups_finalsave.mat"
wshedfile = h5py.File(z_vals_file, "r")
wregs = wshedfile["watershedRegions"][:].flatten()
ethogram_split = np.split(wregs, np.cumsum(wshedfile['zValLens'][:].flatten())[:-1])

item_cts = np.unique(np.concatenate(ethogram_split), return_counts=True)
item_relative_freq = (item_cts[1] / np.sum(item_cts[1]))*100
print(item_relative_freq)