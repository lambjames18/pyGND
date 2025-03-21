import h5py
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage import io

import quaternions as qpy
import rotations
import utillities as ut



path = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_EBSD.dream3d"
h5 = h5py.File(path, "r")
gnd = h5["DataStructure/ImageDataContainer/CellData/GND"][...]
fdam = h5["DataStructure/ImageDataContainer/CellData/FDAM"][...]
h5.close()

print(gnd.shape, fdam.shape)

gnd = np.log10(np.clip(gnd, 1, None))

print(gnd.min(), gnd[gnd > 0].min(), gnd.max())
print(fdam.min(), fdam[gnd > 0].min(), fdam.max())

fig, ax = plt.subplots(1, 2, figsize=(13, 13 * 9/16), sharex=True, sharey=True)

im0 = ax[0].imshow(gnd[0, ..., 0], cmap='RdBu_r', vmin=11, vmax=16)
ax[0].set_title("GND")
ax[0].axis('off')

im1 = ax[1].imshow(fdam[0, ..., 0], cmap='jet')
ax[1].set_title("Misorientation")
ax[1].axis('off')

plt.tight_layout()
fig.subplots_adjust(right=0.92, wspace=0.2)
l0 = ax[0].get_position()
l1 = ax[1].get_position()
cbar_ax = fig.add_axes([l0.x1 + 0.01, l0.y0, 0.02, l0.height])
fig.colorbar(im0, cax=cbar_ax)
cbar_ax.set_ylabel("GND density")
ut.make_axis_log(cbar_ax, "y")
cbar_ax = fig.add_axes([l1.x1 + 0.01, l1.y0, 0.02, l1.height])
fig.colorbar(im1, cax=cbar_ax)
cbar_ax.set_ylabel("FDAM")

plt.show()