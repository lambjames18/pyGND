import os
import h5py
import numpy as np

def main(path, note):
    # Read
    h = h5py.File(path)
    featIDs = np.squeeze(h["DataContainers/ImageDataContainer/CellData/FeatureIds"][...]).astype(np.int32)
    euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/EulerAngles"][...]).astype(np.float32)
    spacing = np.squeeze(h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...])
    h.close()

    # Get details about the dataset
    shape = featIDs.shape
    npoints = featIDs.size
    nfeatures = np.unique(featIDs[featIDs > 0]).size

    # Get xyz
    x1, x2, x3 = np.indices(featIDs.shape)
    coordinates = np.stack((x1, x2, x3), axis=-1)
    coordinates = coordinates.reshape(-1, 3).astype(np.float32)

    # Reshape the featureIDs and euler angles
    featIDs = featIDs.reshape(-1)
    euler = euler.reshape(-1, 3)

    # Store the coordinates and the euler angles in one array
    data = np.hstack((coordinates, euler))

    # Save the data
    filename_micro = os.path.basename(path).split(".")[0] + "_MatlabInput-Microstructure.csv"
    filename_ids = os.path.basename(path).split(".")[0] + "_MatlabInput-FeatureIDs.csv"
    filename_notes = os.path.basename(path).split(".")[0] + "_MatlabInput-Notes.txt"
    filename_d3d = os.path.basename(path)
    np.savetxt(filename_micro, data, fmt="%f", delimiter=",")
    np.savetxt(filename_ids, featIDs, fmt="%d", delimiter=",")
    with open(filename_notes, "w") as f:
        f.write(f"DETAILS ABOUT THE DATASET\n\n")
        f.write(f"Note: {note}\n")
        f.write(f"Microstructure data: {filename_micro}\n")
        f.write(f"Feature IDs data: {filename_ids}\n")
        f.write(f"Dream3D file: {filename_d3d}\n")
        f.write(f"Shape: {shape}\n")
        f.write(f"Number of points: {npoints}\n")
        f.write(f"Number of features: {nfeatures}\n")
        f.write(f"Spacing: {spacing}\n")


if __name__ == "__main__":
    path = "D:/Research/scripts/TriBeam_GND/TaAMSpalled_mini.dream3d"
    sample_description = "A small subvolume of a spalled Ta (BCC) microstructure."
    main(path, sample_description)
    print("Done.")
