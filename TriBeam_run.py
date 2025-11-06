# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# Designed to read in and write out to DREAM3D files, but can be adapted for other formats

import os
import numpy as np
import h5py

import utillities as utils
import py_functions as pf
import GND


#############################################
path = "/Users/jameslamb/Documents/research/data/IN718SS_Justine/Testing_Cropped_IN718SS.dream3d"
cell_data_path = "DataContainers/HEDM-IN718SS/HEDM-CellData"
spacing_path = "DataContainers/HEDM-IN718SS/_SIMPL_GEOMETRY/SPACING"
dream3d_nx = False

# Burgers vector magnitude in m
burgers = 2.57e-10

# Number of CPU cores to use
n_cpus = 4

# Crystal structure, 1 = FCC, 2 = BCC, 3 = HCP
cs = 1

# Slip systems
# (FCC) - unused, always 'all'
# (BCC) - 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'screw+110+123', 'screw+112+123', 'all'
# (HCP) - 'basal', 'prismatic', 'pyramidal', 'basal+prismatic', 'basal+pyramidal', 'prismatic+pyramidal', 'all'
slip_systems = "all"

# "l2" or "l1" (where l1 is the absolute value); l2 is faster, l1 may be more accurate
minimization = "l2"

# Whether to show a progress bar
progress_bar = True

# How many data points to process in one chunk (decrease if memory issues)
chunk_size = 100

# Spacing units, ensures that the spacing is converted to meters correctly
units = "um"

# Whether to perform the calculation (True) or read in previously calculated data from .npy files (False)
calc = True
#############################################


if __name__ == "__main__":

    # Read in the data from the DREAM3D file
    euler, ids, spacing = utils.read_dream3d(
        path,
        ids_name="FeatureIds",
        euler_name="EulerAngles",
        spacing_units=units,
    )

    # Get densities of geometrically necessary dislocations
    if calc:
        dd, mis = GND.calculate(
            euler,
            ids,
            cs,
            slip_systems,
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

    data_shape = mis.shape[1:] + (1,)
    dd = dd[minimization].sum(axis=0).reshape(data_shape)
    fdm_avg = mis.mean(axis=0).reshape(data_shape)
    fdm_max = mis.max(axis=0).reshape(data_shape)

    # Save the results back to the DREAM3D file
    gnd_written, fdm_avg_written, fdm_max_written = False, False, False
    try:
        gnd_path = utils.extract_path_from_h5(path, "GND")
        h5 = h5py.File(path, "r+")
        h5[gnd_path][...] = dd
        h5.close()
        gnd_written = True
    except Exception as e:
        h5.close()
        print(
            "Failed to write GND data to DREAM3D file. Check that a 'GND' data array exists in the DREAM3D file."
        )
        print(e)

    try:
        fdm_avg_path = utils.extract_path_from_h5(path, "FDM_avg")
        h5 = h5py.File(path, "r+")
        h5[fdm_avg_path][...] = fdm_avg
        h5.close()
        fdm_avg_written = True
    except Exception as e:
        h5.close()
        print(
            "Failed to write FDM_avg data to DREAM3D file. Check that a 'FDM_avg' data array exists in the DREAM3D file."
        )
        print(e)

    try:
        fdm_max_path = utils.extract_path_from_h5(path, "FDM_max")
        h5 = h5py.File(path, "r+")
        h5[fdm_max_path][...] = fdm_max
        h5.close()
        fdm_max_written = True
    except Exception as e:
        h5.close()
        print(
            "Failed to write FDM_max data to DREAM3D file. Check that a 'FDM_max' data array exists in the DREAM3D file."
        )
        print(e)

    print("\nResults summary:")
    print("----------------")
    print(f"- GND max: {dd.max():.3e} m\u207b\u00b2")
    print(f"- GND min (non-zero): {dd[dd > 0].min():.3e} m\u207b\u00b2")
    print(f"- FDM_avg max: {fdm_avg.max():.3f}\u00b0")
    print(f"- FDM_max max: {fdm_max.max():.3f}\u00b0")
    print("----------------")
    cwd = os.getcwd()
    if gnd_written:
        print("- GND data written to DREAM3D successfully")
    else:
        print(
            f"- GND data NOT written to DREAM3D. Cached in {os.path.join(cwd, 'dd.npy')}"
        )
    if fdm_avg_written:
        print(
            "- Average finite difference misorientation (FDM_avg) data written to DREAM3D successfully"
        )
    else:
        print(
            f"- Average finite difference misorientation (FDM_avg) data NOT written to DREAM3D. FDM data cached in {os.path.join(cwd, 'mis.npy')}"
        )
    if fdm_max_written:
        print(
            "- Maximum finite difference misorientation (FDM_max) data written to DREAM3D successfully"
        )
    else:
        print(
            f"- Maximum finite difference misorientation (FDM_max) data NOT written to DREAM3D. FDM data cached in {os.path.join(cwd, 'mis.npy')}"
        )
    print("----------------")
