import numpy as np
import h5py

name = "R2S10S5"
name = "CoNiS29S2"
name = "R2S9S4"
sr = np.load(f"./output_data/{name}_GND_SR.npy")
ss = np.load(f"./output_data/{name}_GND_SS.npy")
ms = np.load(f"./output_data/{name}_misori.npy")

sr = np.log10(sr, where=sr > 0)

sample = "NiAlMo_APS"
fname = name + "_Isotropic"
h = h5py.File(f"D:/Research/{sample}/Data/3D/{fname}.dream3d", "r+")
assert h["DataContainers/ImageDataContainer/CellData/GND"].dtype == sr.dtype
assert h["DataContainers/ImageDataContainer/CellData/Misori"].dtype == ms.dtype
assert sr.shape + (1,) == h["DataContainers/ImageDataContainer/CellData/GND"].shape
assert ms.shape + (1,) == h["DataContainers/ImageDataContainer/CellData/Misori"].shape
h["DataContainers/ImageDataContainer/CellData/GND"][...] = sr.reshape(sr.shape + (1,))
h["DataContainers/ImageDataContainer/CellData/Misori"][...] = ms.reshape(ms.shape + (1,))
h.close()
