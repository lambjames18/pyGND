import timeit
import numpy as np
import matplotlib.pyplot as plt
import py_functions as pf


vol_size = (100, 100, 200)
num_features = 400

seed = np.random.seed(1230)
locs = np.random.randint(0, 200, (num_features, 3))
indices = np.swapaxes(np.indices(vol_size).reshape(3, -1), 0, 1)

dists = np.zeros((num_features, np.prod(vol_size)), dtype=np.float32)
for i in range(num_features):
    dists[i] = np.linalg.norm(indices - locs[i], axis=1)

IDs = np.argmin(dists, axis=0)
vol = IDs.reshape(vol_size + (1,))
vol += 1
vol = np.pad(vol[:, :, :, 0], 5, 'constant', constant_values=0)
vol = vol.reshape(vol.shape + (1,))

num_cuts = 5
cut_axis = 2

plt.imshow(vol[vol.shape[0]//2])
plt.show()
plt.close("all")
bounds = pf.cut_dataset(vol, num_cuts, cut_axis)
print(bounds)
