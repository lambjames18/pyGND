import h5py
import numpy as np


d3d_path = "D:/Research/scripts/TriBeam_GND/TaAMSpalled_mini.dream3d" 
d3d_array_str = "DataContainers/ImageDataContainer/CellData/GND_Matlab3"
matlab_output_paths = ["D:/Research/scripts/TriBeam_GND/TaAMSpalled_mini_Data_output_GND110_.mat",
                       "D:/Research/scripts/TriBeam_GND/TaAMSpalled_mini_Data_output_GND112_.mat",
                       "D:/Research/scripts/TriBeam_GND/TaAMSpalled_mini_Data_output_GND123_.mat",
                       "D:/Research/scripts/TriBeam_GND/TaAMSpalled_mini_Data_output_GND_s_.mat"]

d = np.zeros((100, 100, 100))
for matlab_output_path in matlab_output_paths:
    h = h5py.File(matlab_output_path, "r")
    d += h[list(h.keys())[0]][...]
    h.close()

h = h5py.File(d3d_path, "r+")
h[d3d_array_str][...] = d.reshape(h[d3d_array_str].shape)
h.close()
print("Done.")