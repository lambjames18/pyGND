# GND calculations in a 3D TriBeam dataset
# Author: James Lamb (GND calculations originally from Wyatt Witzen)
# All parameters should be stated within the config.ini file

import sys
import os
import numpy as np
import h5py
import mpire
import configparser
import argparse

import py_functions as pf
import GND

parser = argparse.ArgumentParser(prog='3D GND Calculations',
                                 description='Determine the spatial distribution of GND densities in 3D EBSD datasets using crystallography and misorientation.',
                                 epilog='Original GND calculations were developed by Wyatt Wytzen, the python adaptation was developed by James Lamb.')
parser.add_argument("-c", "--config", required=True,  help="The .ini file containing the calculation configuration")
parser.add_argument("-t", "--test", action='store_true', help="Terminate the code prior to excecuting parallel computations. Useful for guaranteeing that the script is setup properly.")
args = parser.parse_args()

config = configparser.ConfigParser()
config.read(args.config)
path = config["PARAMS"]["DREAM3D File"]
directory = config["PARAMS"]["Save Folder"]
ID = config["PARAMS"]["ID"]
burgers = float(config["PARAMS"]["Burgers"])
cs = config["PARAMS"]["Crystallography"]
slip_systems = config["PARAMS"]["Slip Systems"]

if cs.lower() == 'fcc': cs = 1
elif cs.lower() == 'bcc': cs = 2
elif cs.lower() == 'hcp': cs = 3
else: raise ValueError("The crystallography entry in the config file was not one of the following ['fcc', 'bcc', 'hcp'].")


def main(path, burgers, cs):
    # Read data
    h = h5py.File(path)
    spacing = np.squeeze(h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...]) * 1e-6
    featIDs = np.squeeze(h["DataContainers/ImageDataContainer/CellData/FeatureIds"][...])
    try:
        euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/EulerAngles"][...])
    except KeyError:
        try:
            euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/Euler"][...])
        except KeyError:
            raise ValueError("The euler angles cannot be found in the DREAM3D file. Attempted to find 'EulerAngles' and 'Euler'.")

    h.close()
    print("\tSpacing:", spacing)
    print("\tTotal number of points:", featIDs.size)
    print("\tDataset shape:", featIDs.shape)
    full_shape = featIDs.shape

    # Apply a global cropping
    slice_x1, slice_x2, slice_x3 = pf.determine_slicing(featIDs, verbose=False)
    # slice_x1, slice_x2, slice_x3 = (slice(None, 100), slice(None, 100), slice(None, 100))
    featIDs = featIDs[slice_x1, slice_x2, slice_x3]
    euler = euler[slice_x1, slice_x2, slice_x3]
    print("\tNumber of points (after crop):", featIDs.size)
    print("\tDataset shape (after crop):", featIDs.shape)
    cropped_shape = featIDs.shape

    # Get xyz
    x1, x2, x3 = np.indices(featIDs.shape)
    coordinates = np.stack((x1, x2, x3), axis=-1)
    coordinates = coordinates.reshape(-1, 3)
    return full_shape, cropped_shape, (slice_x1, slice_x2, slice_x3), featIDs, euler, coordinates, spacing


def get_num_jobs():
    # Determine what the operating system is
    operating_system = sys.platform
    if operating_system in ["darwin", "win32", "cygwin", "msys"]:
        n_cpus = int(os.cpu_count() - 2)
        n_cpus = 5
    else:
        n_cpus = len(os.sched_getaffinity(0))
    print("\tThere are {} processors available, all will be utilized.".format(n_cpus))
    return n_cpus


if __name__ == '__main__':
    print("\n*** GND Calculations Python Script ***")
    print("Inputs:\n\tConfig File: {}\n\tTest Run: {}\n\tDREAM3D File: {}\n\tID: {}\n\tBurgers Vector: {}\n\tCrystallography: {}\n\tSave Folder: {}".format(args.config, args.test, path, ID, burgers, cs, directory))
    config_exists = os.path.isfile(args.config)
    d3d_exists = os.path.isfile(path)
    save_exists = os.path.isdir(directory)
    if not config_exists: raise ValueError("The config file does not exist.")
    if not d3d_exists: raise ValueError("The DREAM3D file does not exist.")
    if not save_exists: raise ValueError("The save directory does not exist.")
    print("\t-> All inputs are valid.")
    print("\n-> Calling setup function")
    full_shape, cropped_shape, (slice_x1, slice_x2, slice_x3), featIDs, euler, coordinates, spacing = main(path, burgers, cs)

    print("\n-> Creating GND object")
    gnd = GND.GND(cs, burgers, slip_systems)
    gnd.set_data(coordinates, euler, featIDs, spacing)
    gnd.enforce_mask_on_input(gnd.featIDs == 0)
    print("\tNumber of slip systems:", gnd.numSlip)
    coords = list(coordinates)

    # Make the output arrays
    print("\n-> Creating output arrays")
    gnd_sr = np.zeros(full_shape, dtype=np.float32)
    gnd_ms = np.zeros(full_shape, dtype=np.float32)
    gnd_ss = np.zeros(full_shape + (gnd.numSlip,), dtype=np.float32)
    print("\tOutput shape:", gnd_ss.shape)

    print("\n-> Starting parallel computation")
    n_processors = get_num_jobs()
    if args.test:
        print("Terminating the calculation, this was a test run.")
        exit()

    # print("---")
    # v = gnd.compute(coords[0], verbose=True)
    # exit()
    
    with mpire.WorkerPool(n_jobs=n_processors, start_method="spawn") as pool:
        results = pool.map(gnd.compute, coords, progress_bar=True)

    # exit()
    print("\t-> Calculation complete. Unpacking results...")
    gnd.unpack_data(results)
    
    # Now pack in the data, accounting for the slices
    gnd_sr[slice_x1, slice_x2, slice_x3] = gnd.GND_SR
    gnd_ms[slice_x1, slice_x2, slice_x3] = gnd.misori
    gnd_ss[slice_x1, slice_x2, slice_x3] = gnd.GND_SS
        
    print("\t-> Saving data.")
    np.save(directory + ID + "_GND_SR.npy", gnd_sr)
    np.save(directory + ID + "_misori.npy", gnd_ms)
    np.save(directory + ID + "_GND_SS.npy", gnd_ss)
    print("Complete!")
