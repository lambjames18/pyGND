import shutil

strains001 = ["000", "002", "010", "019", "031", "040", "050", "059", "071", "080", "090"]
strains111 = ["000", "002", "010", "020", "030", "040", "050", "060", "070", "080", "090"]

"""
for strain in strains001:
    shutil.copy("Ta001Simulation.ini", f"Ta001_{strain}.ini")
    with open(f"Ta001_{strain}.ini", "r") as f:
        lines = f.readlines()
    with open(f"Ta001_{strain}.ini", "w") as f:
        for line in lines:
            f.write(line.replace("000", strain))

for strain in strains111:
    shutil.copy("Ta111Simulation.ini", f"Ta111_{strain}.ini")
    with open(f"Ta111_{strain}.ini", "r") as f:
        lines = f.readlines()
    with open(f"Ta111_{strain}.ini", "w") as f:
        for line in lines:
            f.write(line.replace("000", strain))
"""


import numpy as np
import h5py

for strain in strains001:
    name = "Ta001_" + strain
    sr = np.load(f"./output_data/{name}_GND_SR.npy")
    h = h5py.File(f"D:/Research/Ta/Data/3D/AMSpall/Simulation/Ta001_{strain}.dream3d", "r+")
    assert h["DataContainers/ImageDataContainer/CellData/GND"].dtype == sr.dtype
    assert sr.shape + (1,) == h["DataContainers/ImageDataContainer/CellData/GND"].shape
    h["DataContainers/ImageDataContainer/CellData/GND"][...] = sr.reshape(sr.shape + (1,))
    h.close()

for strain in strains111:
    name = "Ta111_" + strain
    sr = np.load(f"./output_data/{name}_GND_SR.npy")
    h = h5py.File(f"D:/Research/Ta/Data/3D/AMSpall/Simulation/Ta111_{strain}.dream3d", "r+")
    assert h["DataContainers/ImageDataContainer/CellData/GND"].dtype == sr.dtype
    assert sr.shape + (1,) == h["DataContainers/ImageDataContainer/CellData/GND"].shape
    h["DataContainers/ImageDataContainer/CellData/GND"][...] = sr.reshape(sr.shape + (1,))
    h.close()
