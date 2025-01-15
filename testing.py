import numpy as np
import timeit
from typing import Tuple




def _neighbors(x1, x2, x3, featIDs):
    x1_min, x1_max = max(0, x1-1), min(featIDs.shape[0]-1, x1+1) + 1
    x2_min, x2_max = max(0, x2-1), min(featIDs.shape[1]-1, x2+1) + 1
    x3_min, x3_max = max(0, x3-1), min(featIDs.shape[2]-1, x3+1) + 1
    sub_volume = np.pad(featIDs[x1_min:x1_max, x2_min:x2_max, x3_min:x3_max], 1, 'constant')
    temp = np.copy(featIDs[x1_min:x1_max, x2_min:x2_max, x3_min:x3_max])
    temp[x1-x1_min, x2-x2_min, x3-x3_min] = -1
    center = np.array(np.where(temp == -1))[:, 0] + 1
    x1_neighbors = sub_volume[center[0]-1:center[0]+2, center[1], center[2]]
    x2_neighbors = sub_volume[center[0], center[1]-1:center[1]+2, center[2]]
    x3_neighbors = sub_volume[center[0], center[1], center[2]-1:center[2]+2]
    return (x1_neighbors, x2_neighbors, x3_neighbors)


def analyze_neighborhoods2(grain_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Initialize output arrays
    shape = grain_ids.shape
    x_diffs = np.zeros(shape, dtype=np.int8)
    y_diffs = np.zeros(shape, dtype=np.int8)
    z_diffs = np.zeros(shape, dtype=np.int8)

    # Loop over all voxels
    for idx in np.ndindex(grain_ids.shape):
        x1, x2, x3 = idx
        ref_id = grain_ids[x1, x2, x3]
        if ref_id == 0:
            continue
        x1_n, x2_n, x3_n = _neighbors(x1, x2, x3, grain_ids)
        complete = [False, False, False]

        # Quick check for the most common case
        if np.allclose(x1_n, ref_id):
            x_diffs[x1, x2, x3] = 3
            complete[0] = True
        if np.allclose(x2_n, ref_id):
            y_diffs[x1, x2, x3] = 3
            complete[1] = True
        if np.allclose(x3_n, ref_id):
            z_diffs[x1, x2, x3] = 3
            complete[2] = True

        # Check which ones need to be completed, x1 then x2, then x3
        if not complete[0]:
            # Check if the next or previous voxel is the same as the reference
            # If neither are the same, set as central
            if x1_n[0] == ref_id:
                x_diffs[x1, x2, x3] = 2
            elif x1_n[-1] == ref_id:
                x_diffs[x1, x2, x3] = 1
            else:
                x_diffs[x1, x2, x3] = 0
        if not complete[1]:
            if x2_n[0] == ref_id:
                y_diffs[x1, x2, x3] = 2
            elif x2_n[-1] == ref_id:
                y_diffs[x1, x2, x3] = 1
            else:
                y_diffs[x1, x2, x3] = 0
        if not complete[2]:
            if x3_n[0] == ref_id:
                z_diffs[x1, x2, x3] = 2
            elif x3_n[-1] == ref_id:
                z_diffs[x1, x2, x3] = 1
            else:
                z_diffs[x1, x2, x3] = 0

    return x_diffs, y_diffs, z_diffs


# Example usage:
if __name__ == "__main__":
    # Create a small test dataset
    shapes = [(10, 10, 10), (20, 20, 20), (50, 50, 50), (100, 100, 100), (200, 200, 200), (500, 500, 500)]
    times1 = []
    # times2 = []
    for i, shape in enumerate(shapes):
        test_data = np.random.randint(0, sum(shape) // 10, shape)
        print(f"Testing shape {shape}")
        times1.append(timeit.timeit("analyze_neighborhoods1(test_data)", globals=globals(), number=1))
        # times2.append(timeit.timeit("analyze_neighborhoods2(test_data)", globals=globals(), number=1))

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([np.prod(shape) for shape in shapes], times1, label="Method 1")
    # ax.plot([np.prod(shape) for shape in shapes], times2, label="Method 2")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of voxels")
    ax.set_ylabel("Calculation Time (s)")
    ax.legend()
    plt.show()
