# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import os
import numpy as np
import matplotlib.pyplot as plt

import utillities as utils
import py_functions as pf
import GND


if __name__ == "__main__":
    idx = 5
    ang_path = [
        "E:/cells/CoNi90_full.ang",
        "E:/cells/CoNi90_parallel-110.ang",
        "E:/cells/CoNi90_parallel-001.ang",
        "E:/cells/CoNi90_ortho-110.ang",
        "E:/cells/CoNi90_ortho-001.ang",
        "C:/Users/lambj/Downloads/SX_CoNi_0degLineA_Rescan_cropped.ang",
    ][idx]
    ids_path = [
        "E:/cells/CoNi90_full_grains.txt",
        "E:/cells/CoNi90_parallel-110_grains.txt",
        "E:/cells/CoNi90_parallel-001_grains.txt",
        "E:/cells/CoNi90_ortho-110_grains.txt",
        "E:/cells/CoNi90_ortho-001_grains.txt",
        "C:/Users/lambj/Downloads/SX_CoNi_0degLineA_Rescan_cropped.txt",
    ][idx]

    burgers = 2.48e-10
    n_cpus = 10
    cs = 1
    minimization = "l2"
    progress_bar = True
    chunk_size = 100
    calc = True

    folder = os.path.dirname(ang_path)
    filename = os.path.basename(ang_path).split(".")[0]
    if calc:
        euler, ids, spacing = utils.read_ang(
            ang_path,
            ids_path=ids_path,
        )

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
        dd = dd[minimization]

        np.save(os.path.join(folder, filename + "_dd.npy"), dd)
        np.save(os.path.join(folder, filename + "_mis.npy"), mis)
    else:
        dd = np.load(os.path.join(folder, filename + "_dd.npy"))
        mis = np.load(os.path.join(folder, filename + "_mis.npy"))

    dd[dd <= 0] = 1.0
    dd = np.log10(dd.sum(axis=(0))[0])
    mis = mis.mean(axis=(0))[0]

    utils.view(dd, "Dislocation Density", "RdBu_r", log=True, vmin=7, vmax=11)
    utils.view(mis, "Misorientation", "jet", vmin=0, vmax=5)
