import timeit
import numpy as np
import matplotlib.pyplot as plt
import py_functions as pf



seed = np.random.seed(1230)
locs = np.random.randint(0, 100, (256, 3))
indices = np.swapaxes(np.indices((100, 100, 100)).reshape(3, -1), 0, 1)

dists = np.zeros((256, 100 * 100 * 100), dtype=np.float32)
for i in range(256):
    dists[i] = np.linalg.norm(indices - locs[i], axis=1)

IDs = np.argmin(dists, axis=0)
vol = IDs.reshape((100, 100, 100, 1))
vol = np.pad(vol[:, :, :, 0], 5, 'constant', constant_values=0)
vol = vol.reshape(vol.shape + (1,))

num_cuts = 3
cut_axis = 1

bounds = pf.cut_dataset(vol, num_cuts, cut_axis)
print(bounds)
