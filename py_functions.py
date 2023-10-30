import numpy as np
import matplotlib.pyplot as plt


def cut_dataset(vol, num_cuts, cut_axis, verbose=True):
    """Cut a dataset into subvolumes using the feature IDs 3D data."""
    verbose_slice = vol.shape[0] // 2
    cut_width = vol.shape[cut_axis] // num_cuts

    # Divide the volume into sub volumes based on the cut locations
    print("Dividing the volume into sub volumes based on the cut locations")
    cuts_ids = []
    taken_ids = []
    for i in range(num_cuts):
        slc = tuple(slice(None) if _ != cut_axis else slice(cut_width*i, cut_width*(i+1)) for _ in range(vol.ndim))
        current_ids = np.unique(vol[slc])
        unique_current_ids = np.setdiff1d(current_ids, np.array(taken_ids))
        cuts_ids.append(unique_current_ids[unique_current_ids != 0])
        taken_ids.extend(cuts_ids[i])

    print("Assigning each voxel to a subvolume")
    subvolume = np.zeros_like(vol)
    for i in range(num_cuts):
        mask = np.isin(vol, cuts_ids[i])
        subvolume[mask] = i + 1

    print("Finding the bounds of each subvolume")
    bounds = np.zeros((num_cuts, 2), dtype=int)
    for i in range(num_cuts):
        mask = np.isin(subvolume, i + 1)
        mn = np.array(np.where(mask)).min(axis=1)[cut_axis]
        mx = np.array(np.where(mask)).max(axis=1)[cut_axis] + 1
        bounds[i] = mn, mx
        if verbose:
            plt.imshow(mask.sum(axis=0).astype(bool))
            if cut_axis == 1:
                plt.axhline(mn, color="r")
                plt.axhline(mx, color="r")
            elif cut_axis == 2:
                plt.axvline(mn, color="r")
                plt.axvline(mx, color="r")
            plt.title("Bounds for cut {}".format(i))
            plt.show()
            plt.close("all")

    if verbose:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.imshow(vol[verbose_slice])
        ax.grid()
        ax.set_xticks([])
        if cut_axis == 1:
            for i in range(1, num_cuts):
                ax.axhline(cut_width * i, color="r")
        elif cut_axis == 2:
            for i in range(1, num_cuts):
                ax.axvline(cut_width * i, color="r")
        fig = plt.figure(figsize=(18, 18 / num_cuts))
        for i in range(num_cuts):
            ax = fig.add_subplot(1, num_cuts, i + 1)
            ax.set_title("Subvolume {}".format(i + 1))
            ax.imshow(np.where(subvolume[verbose_slice] == i + 1, vol[verbose_slice], 0))
            ax.grid()
            ax.set_xticks([])
        plt.tight_layout()
        plt.show()

    return bounds, subvolume


def determine_slicing(featIDs, verbose=False):
    mask = featIDs > 0
    p0 = mask.sum(axis=0) > 0
    p1 = mask.sum(axis=1) > 0
    p2 = mask.sum(axis=2) > 0
    mn0 = np.array(np.where(mask)).min(axis=1)[0]
    mx0 = np.array(np.where(mask)).max(axis=1)[0] + 1
    mn1 = np.array(np.where(mask)).min(axis=1)[1]
    mx1 = np.array(np.where(mask)).max(axis=1)[1] + 1
    mn2 = np.array(np.where(mask)).min(axis=1)[2]
    mx2 = np.array(np.where(mask)).max(axis=1)[2] + 1
    if mn0 == 0: mn0 = None
    if mn1 == 0: mn1 = None
    if mn2 == 0: mn2 = None
    if mx0 == mask.shape[0]: mx0 = None
    if mx1 == mask.shape[1]: mx1 = None
    if mx2 == mask.shape[2]: mn2 = None
    if verbose:
        fig = plt.figure()
        ax0 = fig.add_subplot(131)
        ax1 = fig.add_subplot(132)
        ax2 = fig.add_subplot(133)
        ax0.imshow(p0)
        if mn1 is not None: ax0.axhline(mn1, color="r")
        if mx1 is not None: ax0.axhline(mx1, color="r")
        if mn2 is not None: ax0.axvline(mn2, color="r")
        if mx2 is not None: ax0.axvline(mx2, color="r")
        ax1.imshow(p1)
        if mn0 is not None: ax1.axhline(mn0, color="r")
        if mx0 is not None: ax1.axhline(mx0, color="r")
        if mn2 is not None: ax1.axvline(mn2, color="r")
        if mx2 is not None: ax1.axvline(mx2, color="r")
        ax2.imshow(p2)
        if mn0 is not None: ax2.axhline(mn0, color="r")
        if mx0 is not None: ax2.axhline(mx0, color="r")
        if mn1 is not None: ax2.axvline(mn1, color="r")
        if mx1 is not None: ax2.axvline(mx1, color="r")
        ax0.set_title("1 and 2 axes")
        ax1.set_title("0 and 2 axes")
        ax2.set_title("0 and 1 axes")
        plt.tight_layout()
        plt.show()
    slice_x1 = slice(mn0, mx0)
    slice_x2 = slice(mn1, mx1)
    slice_x3 = slice(mn2, mx2)
    return (slice_x1, slice_x2, slice_x3)
