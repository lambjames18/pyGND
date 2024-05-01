import sys
import os
import numpy as np
import h5py
import mpire
import pickle

import py_functions as pf
import GND


burgers = 2.86
cs = 2
slip_systems = "screw + 110 + 112"

def main(path):
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
    # slice_x1, slice_x2, slice_x3 = (slice(None), slice(None), slice(None))
    print("\tSlicing:", slice_x1, slice_x2, slice_x3)
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
        print("\t-> This is a mac or windows system, determining the number of available processors.")
        n_cpus = int(os.cpu_count() / 2)
        # n_cpus = 2
    else:
        print("\t-> This is a linux system, determining the number of available processors.")
        n_cpus = len(os.sched_getaffinity(0))
    print("\t{} processors will be utilized.".format(n_cpus))
    return n_cpus


if __name__ == '__main__':
    paths = ["D:/Research/Ta/Data/3D/AMSpall/Simulation/Ta001.pkl", "D:/Research/Ta/Data/3D/AMSpall/Simulation/Ta111.pkl"]
    for path in paths:
        print(f"\n\nProcessing {path}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        for key in data.keys():
            print(f"{key}")
            if data[key].shape[-1] == 6:
                data[key] = data[key][:, :, :, :4]
            euler = data[key][:, :, :, :3]
            featIDs = data[key][:, :, :, 3]
            full_shape = featIDs.shape
            x1, x2, x3 = np.indices(featIDs.shape)
            coordinates = np.stack((x1, x2, x3), axis=-1)
            coordinates = coordinates.reshape(-1, 3)
            spacing = np.array([1.5, 1.5, 1.5]) * 1e-6

            print("-> Creating GND object")
            gnd = GND.GND(cs, burgers, slip_systems)
            gnd.set_data(coordinates, euler, featIDs, spacing)
            gnd.enforce_mask_on_input(gnd.featIDs == 0)
            print("\tNumber of slip systems:", gnd.numSlip)
            coords = list(coordinates)

            print("\n-> Starting parallel computation")
            n_processors = get_num_jobs()

            with mpire.WorkerPool(n_jobs=n_processors, start_method="spawn") as pool:
                results = pool.map(gnd.compute, coords, progress_bar=True)

            gnd.unpack_data(results)
            print("\t-> Calculation complete.")

            # Save
            gnd_sr = gnd.GND_SR.reshape(full_shape)
            gnd_mis = gnd.misori.reshape(full_shape)
            additions_to_data = np.stack([gnd_sr, gnd_mis], axis=-1)
            data[key] = np.concatenate([data[key], additions_to_data], axis=-1)
            pickle.dump(data, open(path, "wb"))
            print("\t-> Data saved.")
