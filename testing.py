from tqdm.auto import tqdm
import utillities as ut
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

import quaternions as qpy
import rotations
import utillities as ut

eu, ids, s = ut.read_ang("E:/rolled_al/merged_1x1.ang", "E:/rolled_al/FeatureIDs.npy")
dd_l1_new = np.log10(np.clip(np.load("E:/rolled_al/dd_l1.npy").mean(axis=0)[0], 1, None))
dd_l2_new = np.log10(np.clip(np.load("E:/rolled_al/dd_l2.npy").mean(axis=0)[0], 1, None))
dd_l2_old = np.log10(np.clip(np.load("E:/rolled_al/dd_l2_old.npy").mean(axis=-1)[0], 1, None))
misori = np.rad2deg(np.load("E:/rolled_al/misorientation.npy").mean(axis=-1)[0])

with open("E:/rolled_al/osm_values.txt", "r") as f:
    osm = f.read()
osm = np.array(osm.replace(" "*6, " ").replace(" "*5, " ").strip().replace("\n", "").split(" "), dtype=float).reshape(687, 725)


# ut.view_simple(osm, 'RdBu', "E:/rolled_Al/results/osm_trancated.tif")
# ut.view(osm, "OSM", 'RdBu')
# vmin, vmax = np.percentile(dd_l1_new, [0.5, 99.5])
# ut.view_simple(dd_l1_new, "RdBu_r", "E:/rolled_Al/results/l1_truncated.tif", vmin=vmin, vmax=vmax)
# ut.view(dd_l1_new, r"L1 minimization, $\rho^{GND}$ $[m^{-2}]$", 'RdBu_r', log=True, vmin=vmin, vmax=vmax)
# vmin, vmax = np.percentile(dd_l2_new, [0.5, 99.5])
# ut.view_simple(dd_l2_new, "RdBu_r", "E:/rolled_Al/results/l2_truncated.tif", vmin=vmin, vmax=vmax)
# ut.view(dd_l2_new, r"L2 minimization, $\rho^{GND}$ $[m^{-2}]$", 'RdBu_r', log=True, vmin=vmin, vmax=vmax)


# ut.view(misori, "Misorientation $[^\circ]$", 'jet', vmin=0, vmax=4)
# ut.view(dd_l1_new, r"L1 minimization, $\rho^{GND}$ $[m^{-2}]$", 'RdBu_r', log=True, vmin=11.5, vmax=15)
# ut.view(dd_l2_new, r"L2 minimization, $\rho^{GND}$ $[m^{-2}]$", 'RdBu_r', log=True, vmin=11.5, vmax=15)
# ut.view(ids[0], "Feature ID", 'cividis')
# ut.view_simple(misori, 'Greys', "E:/rolled_al/Finite-Difference-Misorientation.tif", vmin=0, vmax=4)
# ut.view_simple(dd_l1_new, 'Greys', "E:/rolled_al/L1-GND-Density.tif", vmin=11.5, vmax=15)
# ut.view_simple(dd_l2_new, 'Greys', "E:/rolled_al/L2-GND-Density.tif", vmin=11.5, vmax=15)

range = ((osm.min(), osm.max()),
         (dd_l1_new[dd_l1_new > 0].min(), dd_l1_new.max()))
extent = range[0] + range[1]

# plot the GND density vs OSM as a scatter plot
# on the x axis, also show the histogram of the OSM
# on the y axis, also show the histogram of the GND density

# fig, ax = plt.subplots(1, 1, figsize=(9, 8))
# ax.tick_params(axis='both', which='major', labelsize=16)
# h, edges = np.histogram(dd_l1_new.flatten(), bins=200, range=range[1], density=True)
# ax.plot(edges[:-1], h, color=(0.9, 0.2, 0.2, 1.0))
# ax.set_xlim(range[1])
# ax.set_xlabel(r"L1 minimization, $\rho^{GND}$ $[m^{-2}]$", color=(0.9, 0.2, 0.2, 1.0), fontsize=20, labelpad=10)
# ax.set_ylabel("Density", fontsize=20, labelpad=10)
# ax.tick_params(axis='x', labelcolor=(0.9, 0.2, 0.2, 1.0), labelsize=16)
# ut.make_axis_log(ax, 'x')
# axt = ax.twiny()
# h, edges = np.histogram(osm.flatten(), bins=200, range=range[0], density=True)
# edges = edges[:-1]
# mask = h > 1e-2
# mask[:50] = h[:50] > 1e-3
# mask[-10:] = True
# axt.plot(edges[mask], h[mask], color=(0.2, 0.2, 0.9, 1.0))
# axt.set_xlim(range[0])
# axt.set_xlabel("OSM", color=(0.2, 0.2, 0.9, 1.0), fontsize=20, labelpad=10)
# axt.tick_params(axis='x', labelcolor=(0.2, 0.2, 0.9, 1.0), labelsize=16)
# plt.tight_layout()
# plt.show()

mask = dd_l1_new > 0
fig, ax = plt.subplots(1, 1, figsize=(9, 7.5))
# _, _, _, im = ax.hist2d(osm[mask], dd_l1_new[mask], bins=40, range=range, cmap='Blues_r', density=True)
ax.scatter(osm[mask], dd_l1_new[mask], s=1.0, c='blue', alpha=0.2, marker="s")
ut.make_axis_log(ax, 'y')
ut.standardize_axis(ax)
ax.set_xlabel("OSM", fontsize=20, labelpad=10)
ax.set_ylabel(r"L1 minimization, $\rho^{GND}$ $[m^{-2}]$", fontsize=20, labelpad=10)
ax.tick_params(axis='both', which='major', labelsize=16)
plt.tight_layout()
plt.subplots_adjust(left=0.15, right=0.82, top=0.98, bottom=0.1)
# l = ax.get_position()
# cax = fig.add_axes([l.x1 + 0.04, l.y0, 0.02, l.height])
# fig.colorbar(im, cax=cax)
# cax.set_ylabel("Density", fontsize=20, labelpad=10)
# cax.tick_params(axis='both', which='major', labelsize=14)
plt.show()