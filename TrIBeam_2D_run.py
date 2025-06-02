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
    path = (
        "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin_basic.dream3d"
    )
    burgers = 2.48e-10
    n_cpus = 12
    cs = 1
    minimization = "l2"
    progress_bar = True
    spacing = np.array([0.1, 0.1, 0.1]) * 1e-6
    chunk_size = 100

    h5 = h5py.File(path, "r")
    euler = h5["DataStructure/ImageGeom/Cell Data/EulerAngles"][...]
    h5.close()
    ids = np.ones_like(euler[:1, :, :, 0], dtype=bool)
    print(f"Euler angles shape: {euler.shape}")
    print(f"Feature IDs shape: {ids.shape}")

    dd = np.zeros_like(euler[..., 0], dtype=np.float32)
    mis_avg = np.zeros_like(euler[..., 0], dtype=np.float32)
    mis_max = np.zeros_like(euler[..., 0], dtype=np.float32)
    for i in range(euler.shape[0]):
        dd_i, mis_i = GND.calculate(
            euler[i : i + 1],
            ids,
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

    np.save(os.path.join(os.path.dirname(path), "CoNi90-thin_basic_GND_3D.npy"), dd)
    np.save(
        os.path.join(os.path.dirname(path), "CoNi90-thin_basic_FDAM_3D_avg.npy"),
        mis_avg,
    )
    np.save(
        os.path.join(os.path.dirname(path), "CoNi90-thin_basic_FDAM_3D_max.npy"),
        mis_max,
    )
