import numpy as np

data = np.zeros((5, 5, 5))

x, y, z = np.indices(data.shape)
xyz0 = np.hstack((x, y, z))
