import timeit
import numpy as np
import matplotlib.pyplot as plt
import py_functions as pf
import h5py


h = h5py.File("D:/Research/CoNi_90/Data/3D/CoNi90.dream3d", "r+")
h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/DIMENSIONS"][...] = np.array([1052, 438, 318])
h.close()