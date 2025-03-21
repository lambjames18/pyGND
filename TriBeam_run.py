# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import os
import numpy as np

import utillities as utils
import py_functions as pf
import GND


if __name__ == '__main__':
    path = "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/stitched_EBSD.dream3d"
    burgers = 2.48e-10
    n_cpus = 5
    cs = 1
    minimization = 'l2'
    progress_bar = True
    spacing = np.array([1.5, 1.5, 1.5]) * 1e-6

    euler, ids = utils.read_dream3d(path, ids_path="DataStructure/ImageDataContainer/CellData/FeatureIds", euler_path="DataStructure/ImageDataContainer/CellData/EulerAngles")
    dd, mis = GND.calculate(euler, ids, cs, burgers, spacing, minimization, n_cpus, progress_bar)

    import h5py
    h5 = h5py.File(path, "r+")
    h5["DataStructure/ImageDataContainer/CellData/GND"][...] = dd[minimization].sum(axis=0).reshape(ids.shape + (1,))
    h5["DataStructure/ImageDataContainer/CellData/FDAM"][...] = mis.mean(axis=0).reshape(ids.shape + (1,))
    h5.close()
