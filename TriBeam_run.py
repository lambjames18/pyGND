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
    path = "E:/CoNi90-thin/old_d3d/CoNi90-thin.dream3d"
    cell_data_path = "DataContainers/ImageDataContainer/CellData"
    spacing_path = "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"
    dream3d_nx = False

    burgers = 2.48e-10
    n_cpus = 8
    cs = 1
    minimization = "l2"
    progress_bar = True
    chunk_size = 500

    euler, ids = utils.read_dream3d(
        path,
        ids_path=f"{cell_data_path}/FeatureIds",
        euler_path=f"{cell_data_path}/EulerAngles",
    )
    spacing = utils.read_dream3d_spacing(
        path, spacing_path=spacing_path, dream3d_nx=dream3d_nx
    )

    dd, mis = GND.calculate(
        euler, ids, cs, burgers, spacing, minimization, n_cpus, progress_bar, chunk_size
    )

    import h5py

    try:
        h5 = h5py.File(path, "r+")
        h5[f"{cell_data_path}/GND"][...] = (
            dd[minimization].sum(axis=0).reshape(ids.shape + (1,))
        )
        h5[f"{cell_data_path}/FDM_avg"][...] = mis.mean(axis=0).reshape(
            ids.shape + (1,)
        )
        h5[f"{cell_data_path}/FDM_max"][...] = mis.max(axis=0).reshape(ids.shape + (1,))
        h5.close()
    except Exception as e:
        print("Failed to write to HDF5 file.")
        print(e)
        h5.close()
