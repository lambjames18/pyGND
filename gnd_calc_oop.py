import matplotlib.pyplot as plt
import numpy as np
import h5py
import mpire
from rich.progress import track

import py_functions as pf
import GND

directory = "./output_data/"
# Name output file
ID = "R2S9S4"

# Convert Burgers to m
burgers = 2.5
cs = 1

# Read data
h = h5py.File(f"D:/Research/NiAlMo_APS/Data/3D/{ID}_Isotropic.dream3d")
spacing = np.squeeze(h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...]) * 1e-6
featIDs = np.squeeze(h["DataContainers/ImageDataContainer/CellData/FeatureIds"][...])
euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/EulerAngles"][...])
iq = np.squeeze(h["DataContainers/ImageDataContainer/CellData/IQ"][...])
print("Spacing:", spacing)
print("Number of points:", featIDs.size)
print("Dataset shape:", featIDs.shape)
full_shape = featIDs.shape

# Remove planes with no grains
# For R2S9S4
# slice_x1 = slice(None, 447)  # For R2S10S5
# slice_x2 = slice(26, 374)  # For R2S10S5
# slice_x3 = slice(25, None)  # For R2S10S5
slice_x1 = slice(None)  # For R2S9S4
slice_x2 = slice(16, 314)  # For R2S9S4
slice_x3 = slice(None, 407)  # For R2S9S4
# slice_x1 = slice(None)  # For CoNiS29S2
# slice_x2 = slice(None)  # For CoNiS29S2
# slice_x3 = slice(None)  # For CoNiS29S2
featIDs = featIDs[slice_x1, slice_x2, slice_x3]
euler = euler[slice_x1, slice_x2, slice_x3]
print("Number of points (after crop):", featIDs.size)
print("Dataset shape (after crop):", featIDs.shape)

# Get xyz
x1, x2, x3 = np.indices(featIDs.shape)
coordinates = np.stack((x1, x2, x3), axis=-1)
coordinates = coordinates.reshape(-1, 3)

if __name__ == '__main__':
    gnd = GND.GND(cs, burgers)
    gnd.set_data(coordinates, euler, featIDs, spacing)
    gnd.enforce_mask_on_input(gnd.featIDs == 0)
    coords = list(coordinates)
    # Non parallel:
    # results = []
    # for i in track(range(len(coords)), "Calculating GND"):
    #     results.append(gnd.compute(coords[i], verbose=False))
    # Parallel:
    with mpire.WorkerPool(n_jobs=10) as pool:
        results = pool.map(gnd.compute, coords, progress_bar=True, max_tasks_active=11)

    # exit()
    print("Calculation complete. Unpacking results...")
    gnd.unpack_data(results)
    
    # Make the data 3D again, first by creating a 3D array of zeros
    gnd_sr = np.zeros(full_shape, dtype=np.float32)
    gnd_misori = np.zeros(full_shape, dtype=np.float32)
    gnd_ss = np.zeros(full_shape + (gnd.GND_SS.shape[-1],), dtype=np.float32)
    # Now pack in the data, accounting for the slices
    gnd_sr[slice_x1, slice_x2, slice_x3] = gnd.GND_SR
    gnd_misori[slice_x1, slice_x2, slice_x3] = gnd.misori
    gnd_ss[slice_x1, slice_x2, slice_x3] = gnd.GND_SS
        
    print("Saving data.")
    np.save(directory + ID + "_GND_SR.npy", gnd_sr)
    np.save(directory + ID + "_misori.npy", gnd_misori)
    np.save(directory + ID + "_GND_SS.npy", gnd_ss)
    print("Complete")
