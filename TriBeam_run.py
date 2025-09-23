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
    path = "C:/Users/Pollock-GPU/Documents/Tri Beam/CoNi90_Wrought/CoNi90_Wrought_aligned_mut_final.dream3d"
    cell_data_path = "DataContainers/ImageDataContainer/CellData"
    spacing_path = "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"
    dream3d_nx = False

    # Burgers vector magnitude in m
    burgers = 2.48e-10
    # Number of CPU cores to use
    n_cpus = 4
    # Crystal structure, 1 = FCC, 2 = BCC, 3 = HCP
    cs = 1 
    # "l2" or "l1" (where l1 is the absolute value); l2 is faster, l1 may be more accurate
    minimization = "l2"
    progress_bar = True
    # How many data points to process in one chunk (decrease if memory issues)
    chunk_size = 1000




    euler, ids = utils.read_dream3d(
        path,
        ids_path=f"{cell_data_path}/FeatureIds",
        euler_path=f"{cell_data_path}/EulerAngles",
    )
    spacing = utils.read_dream3d_spacing(
        path, spacing_path=spacing_path, dream3d_nx=dream3d_nx
    )

    calc = True
    if calc:
        dd, mis = GND.calculate(
            euler,
            ids,
            cs,
            burgers,
            spacing,
            minimization,
            n_cpus,
            progress_bar,
            chunk_size,
        )
        np.save("dd.npy", dd[minimization])
        np.save("mis.npy", mis)
    else:
        print("Reading in calculated data from .npy files")
        dd = np.load("dd.npy")
        mis = np.load("mis.npy")
        dd = {minimization: dd}

    # print("dd1:", dd1.shape, "mis1:", mis1.shape)
    # print("dd2:", dd2.shape, "mis2:", mis2.shape)
    # dd = {
    #     minimization: np.empty(
    #         (dd1.shape[0], dd1.shape[1] + dd2.shape[1] - 2, dd1.shape[2], dd1.shape[3]),
    #         dtype=dd1.dtype,
    #     )
    # }
    # dd[minimization][:, : dd1.shape[1] - 1] = dd1[:, :-1]
    # dd[minimization][:, dd1.shape[1] - 1 :] = dd2[:, 1:]
    # del dd1, dd2  # Free memory
    # print("dd:", dd[minimization].shape)

    # mis = np.empty(
    #     (
    #         mis1.shape[0],
    #         mis1.shape[1] + mis2.shape[1] - 2,
    #         mis1.shape[2],
    #         mis1.shape[3],
    #     ),
    #     dtype=mis1.dtype,
    # )
    # mis[:, : mis1.shape[1] - 1] = mis1[:, :-1]
    # mis[:, mis1.shape[1] - 1 :] = mis2[:, 1:]
    # del mis1, mis2  # Free memory
    # print("mis:", mis.shape)



    # Save the results back to the DREAM3D file
    import h5py

    try:
        h5 = h5py.File(path, "r+")
        h5[f"{cell_data_path}/GND"][...] = (
            dd[minimization].sum(axis=0).reshape(mis.shape[1:] + (1,))
        )
    except Exception as e:
        print("Failed to write GND data to DREAM3D file. Check that a 'GND' data array exists in the DREAM3D file.")
        print(e)
    try:
        h5[f"{cell_data_path}/FDM_avg"][...] = mis.mean(axis=0).reshape(
            mis.shape[1:] + (1,)
        )
    except Exception as e:
        print("Failed to write FDM_avg data to DREAM3D file. Check that a 'FDM_avg' data array exists in the DREAM3D file.")
        print(e)
    try:
        h5[f"{cell_data_path}/FDM_max"][...] = mis.max(axis=0).reshape(
            mis.shape[1:] + (1,)
        )
    except Exception as e:
        print("Failed to write FDM_max data to DREAM3D file. Check that a 'FDM_max' data array exists in the DREAM3D file.")
        print(e)
    
    h5.close()
    print("Finished")
