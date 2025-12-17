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
# Path to the DREAM3D file
path = "/Users/jameslamb/coding/research/TriBeam_GND/demo_data/CoNi.dream3d"

# Name of the DataArray containing the Feature IDs
ids_name = "FeatureIds"

# Name of the DataArray containing the Euler angles
euler_name = "EulerAngles"

# Burgers vector magnitude in m
burgers = 2.48e-10

# Number of CPU cores to use
n_cpus = 5

# Crystal structure, 1 = FCC, 2 = BCC, 3 = HCP
cs = 1

# Slip systems
# (FCC) - unused, always 'all'
# (BCC) - 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'screw+110+123', 'screw+112+123', 'all'
# (HCP) - 'basal', 'prismatic', 'pyramidal', 'basal+prismatic', 'basal+pyramidal', 'prismatic+pyramidal', 'all'
slip_systems = "all"

# "l2" or "l1" (where l1 is the absolute value); l2 is faster, l1 may be more accurate
# can be a list/tuple of both as well, e.g. minimization = ["l1", "l2"]
minimization = ["l2", "l1"]

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
    euler, ids, spacing = utils.read_dream3d(
        path,
        ids_name=ids_name,
        euler_name=euler_name,
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
        for m in dd:
            np.save(f"dd_{m}.npy", dd[m])
        np.save("mis.npy", mis)
        print("Finished calculations, temporary data saved to .npy files")
        for m in dd:
            print(f"- {os.path.abspath(f'dd_{m}.npy')}")
        print(f"- {os.path.abspath('mis.npy')}")
    else:
        try:
            print("Reading in calculated data from .npy files")
            dd = {m: np.load(f"dd_{m}.npy") for m in minimization}
            mis = np.load("mis.npy")
            dd = {minimization: dd}
            print("Finished reading in data")
        except Exception as e:
            raise RuntimeError(
                "Error reading in .npy files. Make sure the files exist and match the specified minimization method."
            ) from e

    print("\nResults summary:")
    print("----------------")
    for m in dd:
        print(f"- {m} GND max: {dd[m].max():.3e} m\u207b\u00b2")
        print(f"- {m} GND min (non-zero): {dd[m][dd[m] > 0].min():.3e} m\u207b\u00b2")
    print(f"- FDM_avg max: {mis.mean(axis=0).max():.3f}\u00b0")
    print(f"- FDM_max max: {mis.max(axis=0).max():.3f}\u00b0")
    print("----------------")

    # Write out the results
    utils.save_to_dream3d(path, ids_name, dd, mis)
    print(f"Results saved to DREAM3D file: {os.path.abspath(path)}")
