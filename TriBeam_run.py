# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# Designed to read in and write out to DREAM3D files, but can be adapted for other formats

import os
import numpy as np

import utillities as utils
import py_functions as pf
import GND


#############################################
path = "F:/Haydn/CoNi90_Top/CoNi90_Top_Final.dream3d"
cell_data_path = "DataContainers/ImageDataContainer/CellData"
spacing_path = "DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"
dream3d_nx = False

# Burgers vector magnitude in m
burgers = 2.48e-10
# Number of CPU cores to use
n_cpus = 30
# Crystal structure, 1 = FCC, 2 = BCC, 3 = HCP
cs = 1
# "l2" or "l1" (where l1 is the absolute value); l2 is faster, l1 may be more accurate
minimization = "l2"
# Whether to show a progress bar
progress_bar = True
# How many data points to process in one chunk (decrease if memory issues)
chunk_size = 1000
# Spacing units, ensures that the spacing is converted to meters correctly
units = "um"
# Whether to perform the calculation (True) or read in previously calculated data from .npy files (False)
calc = True
#############################################


if __name__ == "__main__":

    # Read in the data from the DREAM3D file
    euler, ids = utils.read_dream3d(
        path,
        ids_path=f"{cell_data_path}/FeatureIds",
        euler_path=f"{cell_data_path}/EulerAngles",
    )
    spacing = utils.read_dream3d_spacing(
        path, spacing_path=spacing_path, dream3d_nx=dream3d_nx
    )
    # Convert spacing to meters
    if units == "nm" or units == "nanometer" or units == "nanometers":
        spacing *= 1e-9
    elif (
        units == "um"
        or units == "micron"
        or units == "microns"
        or units == "micrometer"
        or units == "micrometers"
        or "µm"
    ):
        spacing *= 1e-6
    elif units == "mm" or units == "millimeter" or units == "millimeters":
        spacing *= 1e-3
    elif units == "m" or units == "meter" or units == "meters":
        pass
    else:
        raise ValueError("units must be one of 'nm', 'um', 'mm', or 'm'")

    # Get densities of geometrically necessary dislocations
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
        print("Finished calculations")
    else:
        print("Reading in calculated data from .npy files")
        dd = np.load("dd.npy")
        mis = np.load("mis.npy")
        dd = {minimization: dd}
        print("Finished reading in data")

    # Save the results back to the DREAM3D file
    import h5py

    try:
        h5 = h5py.File(path, "r+")
        h5[f"{cell_data_path}/GND"][...] = (
            dd[minimization].sum(axis=0).reshape(mis.shape[1:] + (1,))
        )
    except Exception as e:
        print(
            "Failed to write GND data to DREAM3D file. Check that a 'GND' data array exists in the DREAM3D file."
        )
        print(e)
    try:
        h5[f"{cell_data_path}/FDM_avg"][...] = mis.mean(axis=0).reshape(
            mis.shape[1:] + (1,)
        )
    except Exception as e:
        print(
            "Failed to write FDM_avg data to DREAM3D file. Check that a 'FDM_avg' data array exists in the DREAM3D file."
        )
        print(e)
    try:
        h5[f"{cell_data_path}/FDM_max"][...] = mis.max(axis=0).reshape(
            mis.shape[1:] + (1,)
        )
    except Exception as e:
        print(
            "Failed to write FDM_max data to DREAM3D file. Check that a 'FDM_max' data array exists in the DREAM3D file."
        )
        print(e)

    h5.close()
    print("Finished")
