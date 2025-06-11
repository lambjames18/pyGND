import h5py
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from orix.quaternion import Orientation, symmetry
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL

Oh = symmetry.Oh

import quaternions as qpy
import rotations
import utillities as ut


N = 243765769
chunk_size = 1000
out_shape = (169, 1201, 1201)

distances = (
    np.random.rand(N, 3).astype(np.float32) * 1e-6
)  # Random distances for testing
aa = np.random.rand(N, 3, 4).astype(np.float32)
out = np.array_split(aa, N // chunk_size, axis=0)

quats_disorientation = np.empty((N, 3, 4), dtype=np.float32)

start_idx = 0
for chunk in out:
    end_idx = start_idx + chunk.shape[0]
    quats_disorientation[start_idx:end_idx] = chunk
    start_idx = end_idx
print("Quaternions loaded.")

quats_disorientation = quats_disorientation.transpose(1, 0, 2)
print("Quaternions transposed.")

# Convert quaternions to rotation vectors
rot_vectors = qpy.qu_log(quats_disorientation) * 2
del quats_disorientation  # Free memory
print("Quaternions converted to rotation vectors.")

# Get the misorientations from the rotation vectors
misorientation = np.linalg.norm(rot_vectors, axis=-1)
print("Misorientations calculated.")

# Get the orientation gradients
with np.errstate(divide="ignore", invalid="ignore"):
    gradient_tensors = np.where(
        (misorientation == 0).reshape(3, -1, 1),
        0,
        rot_vectors / distances.T[..., None],
    )
print("Gradient tensors calculated.")

# Reshape the output
gradient_tensors = gradient_tensors.transpose(1, 0, 2).reshape(out_shape + (3, 3))
print("Gradient tensors reshaped.")
misorientation = misorientation.T.reshape(out_shape + (3,))
print("Misorientation reshaped.")
