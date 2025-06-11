# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import os
import numpy as np

import utillities as utils
import py_functions as pf
import GND


if __name__ == "__main__":
    # path = "/Users/jameslamb/Documents/research/data/Wrought-DIC/EBSD.dream3d"
    # cell_data_path = "DataStructure/ImageGeometry/Cell Data"
    # spacing_path = "DataStructure/ImageGeometry"
    # path = "E:/CoNi90-thin/old_d3d/CoNi90-thin.dream3d"
    path = "/Users/jameslamb/Documents/research/data/CoNi90-thin/CoNi90-thin.dream3d"
    cell_data_path = "DataContainers/ImageDataContainer/CellData"
    spacing_path = "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"
    dream3d_nx = False

    burgers = 2.48e-10
    n_cpus = 15
    cs = 1
    minimization = "l2"
    progress_bar = True
    chunk_size = 1000

    euler, ids = utils.read_dream3d(
        path,
        ids_path=f"{cell_data_path}/FeatureIds",
        euler_path=f"{cell_data_path}/EulerAngles",
    )
    spacing = utils.read_dream3d_spacing(
        path, spacing_path=spacing_path, dream3d_nx=dream3d_nx
    )

    euler1, euler2 = euler[:101], euler[99:]
    ids1, ids2 = ids[:101], ids[99:]
    del euler, ids  # Free memory

    calc = False
    if calc:
        dd1, mis1 = GND.calculate(
            euler1,
            ids1,
            cs,
            burgers,
            spacing,
            minimization,
            n_cpus,
            progress_bar,
            chunk_size,
        )
        np.save("dd1.npy", dd1[minimization])
        np.save("mis1.npy", mis1)
        dd2, mis2 = GND.calculate(
            euler2,
            ids2,
            cs,
            burgers,
            spacing,
            minimization,
            n_cpus,
            progress_bar,
            chunk_size,
        )
        np.save("dd2.npy", dd2[minimization])
        np.save("mis2.npy", mis2)
    else:
        dd1 = np.load("dd1.npy")
        mis1 = np.load("mis1.npy")
        dd2 = np.load("dd2.npy")
        mis2 = np.load("mis2.npy")

    print("dd1:", dd1.shape, "mis1:", mis1.shape)
    print("dd2:", dd2.shape, "mis2:", mis2.shape)
    dd = {
        minimization: np.empty(
            (dd1.shape[0], dd1.shape[1] + dd2.shape[1] - 2, dd1.shape[2], dd1.shape[3]),
            dtype=dd1.dtype,
        )
    }
    dd[minimization][:, : dd1.shape[1] - 1] = dd1[:, :-1]
    dd[minimization][:, dd1.shape[1] - 1 :] = dd2[:, 1:]
    del dd1, dd2  # Free memory
    print("dd:", dd[minimization].shape)

    mis = np.empty(
        (
            mis1.shape[0],
            mis1.shape[1] + mis2.shape[1] - 2,
            mis1.shape[2],
            mis1.shape[3],
        ),
        dtype=mis1.dtype,
    )
    mis[:, : mis1.shape[1] - 1] = mis1[:, :-1]
    mis[:, mis1.shape[1] - 1 :] = mis2[:, 1:]
    del mis1, mis2  # Free memory
    print("mis:", mis.shape)

    import h5py

    try:
        h5 = h5py.File(path, "r+")
        h5[f"{cell_data_path}/GND"][...] = (
            dd[minimization].sum(axis=0).reshape(mis.shape[1:] + (1,))
        )
        h5[f"{cell_data_path}/FDM_avg"][...] = mis.mean(axis=0).reshape(
            mis.shape[1:] + (1,)
        )
        h5[f"{cell_data_path}/FDM_max"][...] = mis.max(axis=0).reshape(
            mis.shape[1:] + (1,)
        )
        h5.close()
    except Exception as e:
        print("Failed to write to HDF5 file.")
        print(e)
        h5.close()
