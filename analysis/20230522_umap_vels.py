from awkde import GaussianKDE
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import h5py
from multiprocessing import Pool
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

wshed_file = h5py.File(
    "/Genomics/ayroleslab2/scott/git/lts-manuscript/analysis/20230522-mmpy-lts-all-pchip5-headprobinterpy0xhead-medianwin5-gaussian-lombscargle-dynamicwinomega020-singleflysampledtracks/UMAP/20230523_minregions100_zVals_wShed_groups_prevregions.mat"
)

z_vals = wshed_file["zValues"][:].T


def compute_velocity(points):
    # Calculate differences between consecutive points
    diffs = np.diff(points, axis=0)

    # Calculate Euclidean distances and divide by time interval
    velocities = np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2)

    # Append NA to match the original points shape
    velocities = np.append(velocities, np.nan)

    return velocities


# Assuming `points` is your (50739250, 2) numpy array
velocities = compute_velocity(z_vals)

# Compute mean velocity (ignoring NAs)

mean_velocity = np.nanmean(velocities)
print("Mean velocity:", mean_velocity)

# Define mask for finite velocities (i.e., not NaN)
finite_velocities_mask = np.isfinite(velocities)

# Apply mask to points and velocities
finite_points = z_vals[finite_velocities_mask]
finite_velocities = velocities[finite_velocities_mask]

# Compute histogram and normalize by number of velocities per bin
heatmap, xedges, yedges = np.histogram2d(
    finite_points[:, 0], finite_points[:, 1], bins=512, weights=finite_velocities
)

counts, _, _ = np.histogram2d(finite_points[:, 0], finite_points[:, 1], bins=512)

# Avoid division by zero
mean_velocities = np.divide(
    heatmap, counts, out=np.zeros_like(heatmap), where=counts != 0
)

mean_velocities = gaussian_filter(mean_velocities, sigma=1)  # Smooth out the heatmap
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(mean_velocities.T, extent=extent, origin="lower")
plt.colorbar(label="Mean Velocity")
plt.savefig("figures/tmp/velocity.png")
plt.show()

# histogram of velocities
plt.clf()
plt.hist(velocities, bins=100)
plt.yscale("log")
plt.savefig("figures/tmp/velocity_hist.png")
plt.close()
