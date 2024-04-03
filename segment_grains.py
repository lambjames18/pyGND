import matplotlib.pyplot as plt
import numpy as np
import rotations as R
from skimage.segmentation import flood_fill

# Quaternion stuff
R2 = 0.7071067811865475244008443621048490392848359376884740365883398689
R3 = 0.8660254037844386467637231707529361834714026269051903140279034897
LAUE_O = np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [R2, 0, 0, R2],
        [R2, 0, 0, -R2],
        [0, R2, R2, 0],
        [0, -R2, R2, 0],
        [0.5, 0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5, 0.5],
        [R2, R2, 0, 0],
        [R2, -R2, 0, 0],
        [R2, 0, R2, 0],
        [R2, 0, -R2, 0],
        [0, R2, 0, R2],
        [0, -R2, 0, R2],
        [0, 0, R2, R2],
        [0, 0, -R2, R2],
    ],
    dtype=np.float64,
)

def quaternion_raw_multiply(a: np.array, b: np.array, verbose:bool=False) -> np.array:
    a_shape, b_shape = a.shape, b.shape
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError("The last dimension of both arrays must be 4.")
    if a.ndim == b.ndim + 1:
        b = b.reshape((1,) + b.shape)
    elif a.ndim + 1 == b.ndim:
        a = a.reshape((1,) + a.shape)
    elif a.ndim != b.ndim:
        raise ValueError("The two arrays must have the same number of dimensions or one more dimension in one of the arrays.")
    # Make sure we have an array of quaternions, not just a quaternion
    if a.ndim == 1:
        a = a.reshape((1, -1))
        b = b.reshape((1, -1))
    # Now we setup broadcasting
    a = a.reshape(a.shape[:1] + (1,) + a.shape[1:])  # this sets up broadcasting for the number of unique quaternions in a
    b = b.reshape((1,) + b.shape)  # this sets up broadcasting for the number of unique quaternions in b
    if a.ndim > 3:
        a = a.reshape(a.shape[:-1] + (1,) + a.shape[-1:])
        b = b.reshape(b.shape[:-2] + (1,) + b.shape[-2:])
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    stack = np.stack((ow, ox, oy, oz), -1)
    if verbose:
        print(" . input:", a_shape, b_shape, "convert:", a.shape, b.shape, "output:", stack.shape)
    return standardize_qu(np.squeeze(stack))

def standardize_qu(q: np.array) -> np.array:
    q_out = q / np.linalg.norm(q, axis=-1, keepdims=True)
    q_out = np.where(q_out[..., :1] < 0, -q_out, q_out)
    q_out += 0
    return q_out

def inverse_qu(qu: np.array):
    qu[..., 1:] = -qu[..., 1:]
    return qu

def misorientation(q1, q2):
    q1s = quaternion_raw_multiply(q1, LAUE_O)
    q2s = quaternion_raw_multiply(q2, LAUE_O)
    mis = quaternion_raw_multiply(q1s, q2s)
    axangle = R.qu2ax(mis)
    angles = np.abs(axangle[..., 3])
    return angles

def segment(eulerangles, angle_threshold=5):
    # Pred the 2D data
    q = R.eu2qu(eulerangles)
    q_pad = np.pad(q, ((1, 1), (1, 1), (0, 0)), mode="constant")
    mis = np.zeros(q.shape[:2])
    for i in range(1, q_pad.shape[0] - 1):
        for j in range(1, q_pad.shape[1] - 1):
            q_ref = q_pad[i, j]
            qs = q_pad[i - 1:i + 2, j - 1:j + 2].reshape(-1, 4)
            m = misorientation(q_ref, qs)
            mis[i - 1, j - 1] = m.max()
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(mis)
    ax[1].imshow(mis > angle_threshold)
    plt.show()
    exit()
    # print(f"\nSegmented into {grain_count} grains")
    # return featIDs
