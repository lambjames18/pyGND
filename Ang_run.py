# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import os
import numpy as np

import utillities as utils
import GND


if __name__ == "__main__":
    # Data
    path = ...
    ids_path = ...
    burgers = ...
    spacing = np.array([1.5, 1.5, 1.5]) * 1e-6
    # Parameters
    n_cpus = 5
    cs = 1
    minimization = "l2"
    progress_bar = True

    euler, ids = utils.read_ang(
        path,
        ids_path,
    )
    dd, mis = GND.calculate(
        euler, ids, cs, burgers, spacing, minimization, n_cpus, progress_bar
    )

    import h5py

    h5 = h5py.File(path, "r+")
    h5["DataStructure/ImageDataContainer/CellData/GND"][...] = (
        dd[minimization].sum(axis=0).reshape(ids.shape + (1,))
    )
    h5["DataStructure/ImageDataContainer/CellData/FDAM"][...] = mis.mean(
        axis=0
    ).reshape(ids.shape + (1,))
    h5.close()
