
import h5py
import numpy as np
import matplotlib.pyplot as plt

d3d_folder = "./"
name = "TaAMSpalled_mini"

# Read in the dream3d file
h5 = h5py.File(d3d_folder + name + ".dream3d", 'r')

# Get the needed arrays
Ids = np.squeeze(h5["DataContainers/ImageDataContainer/CellData/FeatureIds"][...])
euler = np.squeeze(h5["DataContainers/ImageDataContainer/CellData/EulerAngles"][...])

# Create an indices array
x1, x2, x3 = np.indices(Ids.shape)
coords = np.stack((x1, x2, x3), axis=-1)

# Reshape the arrays
Ids = Ids.reshape(-1, 1)
euler = euler.reshape(-1, 3)
coords = coords.reshape(-1, 3)
# print(coords[:1000])
# exit()

# Join the arrays
data = np.hstack((coords, euler))

np.savetxt(name + "_GrainIDs.csv", Ids, fmt="%d", delimiter=",")
np.savetxt(name + "_Data.csv", data, fmt="%f", delimiter=",")
