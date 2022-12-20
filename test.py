import timeit
import numpy as np
import matplotlib.pyplot as plt

def cut_dataset(num_cuts, cut_axis, verbose=True):
    cut_width = vol.shape[cut_axis] // num_cuts

    # Divide the volume into sub volumes based on the cut locations
    print("Dividing the volume into sub volumes based on the cut locations")
    cuts_ids = []
    taken_ids = []
    for i in range(num_cuts):
        slc = tuple(slice(None) if _ != cut_axis else slice(cut_width*i, cut_width*(i+1)) for _ in range(vol.ndim))
        temp = np.copy(vol)
        temp[slc] = 0
        if verbose:
            plt.imshow(temp[50])
            plt.imshow(vol[50], alpha=0.5)
            plt.title("ROI for cut {}".format(i))
            plt.show()
            plt.close("all")
        current_ids = np.unique(vol[slc])
        unique_current_ids = np.setdiff1d(current_ids, np.array(taken_ids))
        cuts_ids.append(unique_current_ids)
        taken_ids.extend(cuts_ids[i])

    print("Assigning each voxel to a subvolume")
    subvolume = np.zeros_like(vol)
    for i in range(num_cuts):
        mask = np.isin(vol, cuts_ids[i])
        subvolume[mask] = i

    print("Finding the bounds of each subvolume")
    bounds = np.zeros((num_cuts, 2), dtype=int)
    for i in range(num_cuts):
        mask = np.isin(subvolume, i)
        mn = np.array(np.where(mask)).min(axis=1)[cut_axis]
        mx = np.array(np.where(mask)).max(axis=1)[cut_axis]
        bounds[i] = mn, mx
        if verbose:
            plt.imshow(mask.sum(axis=0).astype(bool))
            plt.axhline(mn, color="r")
            plt.axhline(mx, color="r")
            plt.title("Bounds for cut {}".format(i))
            plt.show()
            plt.close("all")

    if verbose:
        fig = plt.figure(figsize=(12, 4))
        for i in range(num_cuts):
            ax = fig.add_subplot(1, num_cuts, i+1)
            ax.imshow(np.where(subvolume[50] == i, vol[50], 0))
        plt.show()
    
    return bounds


seed = np.random.seed(1230)
locs = np.random.randint(0, 100, (256, 3))
indices = np.swapaxes(np.indices((100, 100, 100)).reshape(3, -1), 0, 1)

dists = np.zeros((256, 100 * 100 * 100), dtype=np.float32)
for i in range(256):
    dists[i] = np.linalg.norm(indices - locs[i], axis=1)

IDs = np.argmin(dists, axis=0)
vol = IDs.reshape((100, 100, 100, 1))
indices = np.moveaxis(np.indices((100, 100, 100), dtype=np.float32), 0, -1)

num_cuts = 3
cut_axis = 1

bounds = cut_dataset(num_cuts, cut_axis)
print(bounds)
