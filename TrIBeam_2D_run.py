# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import os
import numpy as np
import h5py

import utillities as utils
import py_functions as pf
import GND


if __name__ == "__main__":
    path = "E:/CoNi90-thin/old_d3d/CoNi90-thin.dream3d"
    burgers = 2.48e-10
    n_cpus = 15
    cs = 1
    minimization = "l2"
    progress_bar = True
    spacing = np.array([0.1, 0.1, 1.0]) * 1e-6
    chunk_size = 1000

    h5 = h5py.File(path, "r")
    euler = h5["DataContainers/ImageDataContainer/CellData/EulerAngles"][...]
    ids = h5["DataContainers/ImageDataContainer/CellData/FeatureIds"][..., 0]
    h5.close()
    print(f"Euler angles shape: {euler.shape}")
    print(f"Feature IDs shape: {ids.shape}")

    dd = np.zeros_like(euler[..., 0], dtype=np.float32)
    mis_avg = np.zeros_like(euler[..., 0], dtype=np.float32)
    mis_max = np.zeros_like(euler[..., 0], dtype=np.float32)
    for i in range(euler.shape[0]):
        print(f"Processing slice {i + 1}/{euler.shape[0]}")
        dd_i, mis_i = GND.calculate(
            euler[i : i + 1],
            ids[i : i + 1],
            cs,
            burgers,
            spacing,
            minimization,
            n_cpus,
            progress_bar,
            chunk_size,
        )
        dd[i] = dd_i[minimization].sum(axis=0)
        mis_avg[i] = mis_i.mean(axis=0)
        mis_max[i] = mis_i.max(axis=0)

    np.save(os.path.join(os.path.dirname(path), "CoNi90-thin_GND.npy"), dd)
    np.save(
        os.path.join(os.path.dirname(path), "CoNi90-thin_FDAM_avg.npy"),
        mis_avg,
    )
    np.save(
        os.path.join(os.path.dirname(path), "CoNi90-thin_FDAM_max.npy"),
        mis_max,
    )
