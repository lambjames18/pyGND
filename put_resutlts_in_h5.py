import numpy as np
import h5py
import matplotlib.pyplot as plt

# name = "TaAMSpalled_mini"
# sr = np.load(f"./output_data/TaAMSpall-Test_GND_SR.npy")
# # sr = np.log10(sr, where=sr > 0)
# bins = np.logspace(9, 15, 100)
# plt.hist(sr[sr > 0].flatten(), bins=bins, density=True)
# plt.xscale("log")
# plt.title("Subvolume GND SR")
# plt.show()
# print("{:.4e}, {:.4e}".format(sr[sr > 0].min(), sr.max()))
# # exit()
# print(np.isnan(sr).sum())
# h = h5py.File(f"TaAMSpalled_mini.dream3d", "r+")
# h["DataContainers/ImageDataContainer/CellData/GND_Python2"][...] = sr.reshape(sr.shape + (1,))
# exit()

# name = "R2S9S4"
# name = "R2S10S5"
# name = "CoNiS29S2"
name = "CoNiS29S2End"
# name = "CoNi90"
# name = "IN718EBM"
# name = "CoNi16"
# name = "TaSpalled"
# name = "TaAMSpalled"
sr = np.load(f"./output_data/{name}_GND_SR.npy")
# ss = np.load(f"./output_data/{name}_GND_SS.npy")
# ms = np.load(f"./output_data/{name}_misori.npy")

# sr = np.log10(sr, where=sr > 0)

# sample = "R2S9S4"
# sample = "R2S10S5"
sample = "Co_APS"
# sample = "CoNi_90"
# sample = "IN718"
# sample = "CoNi_16"
# sample = "Ta"
# fname = "Raster_Start_Isotropic"
fname = "Raster_End_Isotropic"
# fname = name
h = h5py.File(f"D:/Research/{sample}/Data/3D/{fname}.dream3d", "r+")
# print(h["DataContainers/ImageDataContainer/CellData"].keys())
# exit()
assert h["DataContainers/ImageDataContainer/CellData/GND"].dtype == sr.dtype
assert sr.shape + (1,) == h["DataContainers/ImageDataContainer/CellData/GND"].shape
h["DataContainers/ImageDataContainer/CellData/GND"][...] = sr.reshape(sr.shape + (1,))
# assert h["DataContainers/ImageDataContainer/CellData/Misori"].dtype == ms.dtype
# assert ms.shape + (1,) == h["DataContainers/ImageDataContainer/CellData/Misori"].shape
# h["DataContainers/ImageDataContainer/CellData/Misori"][...] = ms.reshape(ms.shape + (1,))
h.close()
