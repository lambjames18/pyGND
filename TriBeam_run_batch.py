# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import os
import numpy as np
import h5py
import time

import utillities as utils
import py_functions as pf
import GND


if __name__ == "__main__":
    paths = [
        "D:/Research/Co_APS/Data/3D/Raster_End_Isotropic.dream3d",
        "D:/Research/Co_APS/Data/3D/Raster_Start_Isotropic.dream3d",
        "D:/Research/R2S9S4/Data/3D/R2S9S4_Isotropic.dream3d",
        "D:/Research/R2S10S1/Data/3D/R2S10S1_Isotropic.dream3d",
        "D:/Research/R2S10S5/Data/3D/R2S10S5.dream3d",
        "D:/Research/CoNi_90/Data/3D/CoNi90.dream3d",
        "D:/Research/CoNi67/Data/3D/CoNi67.dream3d",
        "D:/Research/CoNi_16/Data/3D/new/CoNi16.dream3d",
    ]
    burgers = [
        2.48e-10,  # Co_APS
        2.48e-10,  # Co_APS
        2.5e-10,  # R2S9S4
        2.5e-10,  # R2S10S1
        2.5e-10,  # R2S10S5
        2.48e-10,  # CoNi_90
        2.48e-10,  # CoNi_67
        2.48e-10,  # CoNi_16
    ]
    calculate = [
        False,  # Co_APS
        False,  # Co_APS
        False,  # R2S9S4
        False,  # R2S10S1
        True,  # R2S10S5
        True,  # CoNi_90
        True,  # CoNi_67
        True,  # CoNi_16
    ]
    n_cpus = 15
    cs = 1
    minimization = "l2"
    progress_bar = True
    chunk_size = 50

    for i in range(len(paths)):
        path = paths[i]
        burger = burgers[i]
        print("*" * 50)
        print(path)
        print("Burgers vector:", burger)

        euler, ids = utils.read_dream3d(
            path,
            ids_path="DataContainers/ImageDataContainer/CellData/FeatureIds",
            euler_path="DataContainers/ImageDataContainer/CellData/EulerAngles",
        )
        spacing = (
            utils.read_dream3d_spacing(
                path, "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING", False
            )
            * 1e-6
        )
        print("Shape:", ids.shape)
        print("Spacing:", spacing)

        if calculate[i]:
            t0 = time.time()
            dd, mis = GND.calculate(
                euler,
                ids,
                cs,
                burger,
                spacing,
                minimization=minimization,
                n_cpus=n_cpus,
                progress_bar=progress_bar,
                chunk_size=chunk_size,
            )
            dd = dd[minimization]
            np.save(path.replace(".dream3d", "_GND.npy"), dd)
            np.save(path.replace(".dream3d", "_MIS.npy"), mis)
            print("Time:", time.time() - t0)
        else:
            dd = np.load(path.replace(".dream3d", "_GND.npy"))
            print("---> Loaded GND data")

        h5 = h5py.File(path, "r+")
        h5["DataContainers/ImageDataContainer/CellData/GND"][...] = dd.sum(
            axis=0
        ).reshape(ids.shape + (1,))
        h5.close()
        print("*" * 50)
