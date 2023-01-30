import numpy as np
import h5py
import mpire

import py_functions as pf
import GND

directory = "./output_data/"
# Name output file
# ID = "R2S9S4"
# ID = "CoNi16"
ID = "CoNi90"

# Convert Burgers to m
burgers = 2.5
cs = 1


def main():
    # Read data
    # h = h5py.File(f"D:/Research/NiAlMo_APS/Data/3D/{ID}_Isotropic.dream3d")
    # h = h5py.File(f"D:/Research/CoNi_16/Data/3D/{ID}.dream3d")
    h = h5py.File(f"D:/Research/CoNi_TFS/Data/3D/{ID}.dream3d")
    spacing = np.squeeze(h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...]) * 1e-6
    featIDs = np.squeeze(h["DataContainers/ImageDataContainer/CellData/FeatureIds"][...])
    euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/EulerAngles"][...])
    h.close()
    print("Spacing:", spacing)
    print("Total number of points:", featIDs.size)
    print("Dataset shape:", featIDs.shape)
    full_shape = featIDs.shape

    # Apply a global cropping
    slice_x1, slice_x2, slice_x3 = pf.determine_slicing(featIDs, verbose=False)
    # slice_x1 = slice(None, 50)
    # slice_x2 = slice(100, 150)
    # slice_x3 = slice(None)
    # Remove planes with no grains
    # slice_x1 = slice(None, 447)  # For R2S10S5
    # slice_x2 = slice(26, 374)  # For R2S10S5
    # slice_x3 = slice(25, None)  # For R2S10S5
    # slice_x1 = slice(None)  # For R2S9S4
    # slice_x2 = slice(16, 314)  # For R2S9S4
    # slice_x3 = slice(None, 407)  # For R2S9S4
    # slice_x1 = slice(None)  # For CoNi16
    # slice_x2 = slice(None, 472)  # For CoNi16
    # slice_x3 = slice(39, 601)  # For CoNi16
    # slice_x1 = slice(None)  # For CoNiS29S2
    # slice_x2 = slice(None)  # For CoNiS29S2
    # slice_x3 = slice(None)  # For CoNiS29S2
    featIDs = featIDs[slice_x1, slice_x2, slice_x3]
    euler = euler[slice_x1, slice_x2, slice_x3]
    print("Number of points (after crop):", featIDs.size)
    print("Dataset shape (after crop):", featIDs.shape)
    cropped_shape = featIDs.shape

    num_cuts = 21
    cut_axis = 2
    bounds, volume = pf.cut_dataset(featIDs, num_cuts, cut_axis, verbose=False)
    print(volume.shape)
    runs = {}
    for i in range(num_cuts):
        mask = np.zeros(cropped_shape, dtype=bool)
        mask[volume == i + 1] = True
        slc = tuple(slice(None) if _ != cut_axis else slice(int(bounds[i, 0]), int(bounds[i, 1])) for _ in range(3))
        featIDs_temp = np.copy(featIDs)
        featIDs_temp[volume != i + 1] = 0
        featIDs_temp = featIDs_temp[slc]
        euler_temp = np.copy(euler)
        euler_temp[volume != i + 1] = 0
        euler_temp = euler_temp[slc]
        x1, x2, x3 = np.indices(featIDs_temp.shape)
        coordinates = np.stack((x1, x2, x3), axis=-1)
        coordinates = coordinates.reshape(-1, 3)
        gnd = GND.GND(cs, burgers)
        gnd.set_data(coordinates, euler_temp, featIDs_temp, spacing)
        gnd.enforce_mask_on_input(gnd.featIDs == 0)
        runs[i] = dict(bounds=bounds,
                       mask=mask,
                       spacing=spacing,
                       featIDs=featIDs_temp,
                       euler=euler_temp,
                       coordinates=list(coordinates),
                       cs=cs,
                       burgers=burgers,
                       gnd=gnd)
        print(f" -> Number of points for subvolume {i}: {featIDs_temp.size}, with shape {featIDs_temp.shape}.")
    return full_shape, (slice_x1, slice_x2, slice_x3), runs, bounds, cut_axis
    

# Get xyz
# x1, x2, x3 = np.indices(featIDs.shape)
# coordinates = np.stack((x1, x2, x3), axis=-1)
# coordinates = coordinates.reshape(-1, 3)
if __name__ == '__main__':
    print("\n*** Starting setup ***")
    full_shape, (slice_x1, slice_x2, slice_x3), runs, bounds, cut_axis = main()
    print("\n*** Starting computations ***")
    # Make the output arrays
    gnd_sr = np.zeros(full_shape, dtype=np.float32)
    gnd_ms = np.zeros(full_shape, dtype=np.float32)
    gnd_ss = np.zeros(full_shape + (runs[0]["gnd"].numSlip,), dtype=np.float32)

    # gnd = GND.GND(cs, burgers)
    # gnd.set_data(coordinates, euler, featIDs, spacing)
    # gnd.enforce_mask_on_input(gnd.featIDs == 0)
    # coords = list(coordinates)
    for i, key in enumerate(runs.keys()):
        with mpire.WorkerPool(n_jobs=19) as pool:
            results = pool.map(runs[key]['gnd'].compute, runs[key]['coordinates'], progress_bar=True, max_tasks_active=17)
        runs[key]['gnd'].unpack_data(results)
        slc = tuple(slice(None) if _ != cut_axis else slice(int(bounds[i, 0]), int(bounds[i, 1])) for _ in range(3))
        # print(runs[key]['gnd'].GND_SR.shape)
        # print(gnd_sr[slice_x1, slice_x2, slice_x3].shape)
        # print(gnd_sr[slice_x1, slice_x2, slice_x3][slc].shape)
        # print((runs[key]['mask'][slc] == True).shape)

        gnd_sr[slice_x1, slice_x2, slice_x3][slc][runs[key]['mask'][slc]] = runs[key]["gnd"].GND_SR[runs[key]['mask'][slc]]
        gnd_ms[slice_x1, slice_x2, slice_x3][slc][runs[key]['mask'][slc]] = runs[key]["gnd"].misori[runs[key]['mask'][slc]]
        gnd_ss[slice_x1, slice_x2, slice_x3][slc][runs[key]['mask'][slc]] = runs[key]["gnd"].GND_SS[runs[key]['mask'][slc]]
        
    # Non parallel:
    # results = []
    # for i in track(range(len(coords)), "Calculating GND"):
    #     results.append(gnd.compute(coords[i], verbose=False))
    # Parallel:
    # with mpire.WorkerPool(n_jobs=2) as pool:
    #     results = pool.map(gnd.compute, coords, progress_bar=True, max_tasks_active=3)

    # exit()
    print("Calculation complete. Unpacking results...")
    # gnd.unpack_data(results)
    
    # Now pack in the data, accounting for the slices
    # gnd_sr[slice_x1, slice_x2, slice_x3] = gnd.GND_SR
    # gnd_ms[slice_x1, slice_x2, slice_x3] = gnd.misori
    # gnd_ss[slice_x1, slice_x2, slice_x3] = gnd.GND_SS
        
    print("Saving data.")
    np.save(directory + ID + "_GND_SR.npy", gnd_sr)
    np.save(directory + ID + "_misori.npy", gnd_ms)
    np.save(directory + ID + "_GND_SS.npy", gnd_ss)
    print("Complete")
