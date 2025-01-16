import numpy as np
import h5py


gnd_path = "./output_data/TaAMSpalled_mini_GND_SR.npy"
d3d_path = "D:/Research/scripts/TriBeam_GND/python-matlab-test/TaAMSpalled_mini.dream3d"
d3d_array_str = "DataContainers/ImageDataContainer/CellData/GND_Python"

sr = np.load(gnd_path)
h = h5py.File(d3d_path, "r+")
assert h[d3d_array_str].dtype == sr.dtype
assert sr.shape + (1,) == h[d3d_array_str].shape
h[d3d_array_str][...] = sr.reshape(sr.shape + (1,))
h.close()
