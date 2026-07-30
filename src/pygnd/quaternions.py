"""

Unit normal quaternions (points that sit on the surface of the 3-sphere with
unit radius in 4D Euclidean space) are used to represent 3D rotations. This
module provides a set of operations for working with quaternions in general.
Often times only the angle of the rotation is needed for comparison amongst
quaternions, so separate functions are provided for accelerating this common
operation. The quaternion (w, x, y, z) is used to represent a rotation that is
indistinguishable from the quaternion (-w, x, y, z), so the standardization
function is provided to make the real part non-negative by conjugation, limiting
the hypervolume we work with to the positive w hemisphere of the 3-sphere.

For more information on quaternions, see:

https://en.wikipedia.org/wiki/Quaternion

Adopted from Pynp3D

https://github.com/facebookresearch/pynp3d

"""

import numpy as np
from tqdm import tqdm
from pygnd.rotations import epsijk
from joblib import Parallel, delayed


# Basic operations


def qu_std(qu: np.ndarray) -> np.ndarray:
    """
    Standardize unit quaternion to have non-negative real part.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        Standardized quaternions as array of shape (..., 4).
    """
    return np.where(qu[..., 0:1] >= 0, qu, -qu)


def qu_norm(qu: np.ndarray) -> np.ndarray:
    """
    Normalize quaternions to unit norm.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        np.ndarray of normalized quaternions.
    """
    norms = np.linalg.norm(qu, axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(norms > 0, qu / norms, 0)


def qu_norm_std(qu: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion to unit norm and make real part non-negative.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        np.ndarray of normalized and standardized quaternions.
    """
    return qu_std(qu_norm(qu))


def qu_conj(qu: np.ndarray) -> np.ndarray:
    """
    Get the unit quaternions for the inverse action.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The inverse, a array of quaternions of shape (..., 4).
    """
    scaling = np.array([1, -1, -1, -1], dtype=qu.dtype)
    return qu * scaling


def qu_angle(qu: np.ndarray) -> np.ndarray:
    """
    Compute angles of rotation for quaternions.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        array of shape (..., ) of rotation angles.
    """
    out = np.where(np.isclose(qu[..., 0], 1.0), 0.0, 2 * np.arccos(qu[..., 0]))
    return out


def qu_axis(qu: np.ndarray) -> np.ndarray:
    """
    Compute the axis of rotation for quaternions.

    Args:
        qu: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        array of shape (..., 3) of rotation axes.
    """
    mag = np.linalg.norm(qu[..., 1:], axis=-1, keepdims=True)
    return np.where(np.isclose(mag, 0.0), [0.0, 0.0, 1.0], qu[..., 1:] / mag)


# Multiplication operations


def qu_prod_raw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions.
    Usual np rules for broadcasting apply.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        The product of a and b, a array of quaternions shape (..., 4).
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    ow = aw * bw - ax * bx - (ay * by + az * bz)
    ow = aw * bw - ax * bx - (ay * by + az * bz)
    ox = aw * bx + ax * bw + epsijk * (ay * bz - az * by)
    oy = aw * by + ay * bw + epsijk * (az * bx - ax * bz)
    oz = aw * bz + az * bw + epsijk * (ax * by - ay * bx)

    return np.stack((ow, ox, oy, oz), -1)


def qu_prod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication, then make real part non-negative.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b np.ndarray shape (..., 4) of the quaternion product.

    """
    ab = qu_prod_raw(a, b)
    return qu_std(ab)


def qu_prod_axis(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return the axis of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b np.ndarray shape (..., 3) of quaternion product axes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ox = aw * bx + ax * bw + epsijk * ay * bz - epsijk * az * by
    oy = aw * by - epsijk * ax * bz + ay * bw + epsijk * az * bx
    oz = aw * bz + epsijk * ax * by - epsijk * ay * bx + az * bw

    return np.stack((ox, oy, oz), -1)


def qu_prod_pos_real(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Return only the magnitude of the real part of the quaternion product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b np.ndarray shape (..., ) of quaternion product real part magnitudes.
    """
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    return np.abs(ow)


def qu_triple_prod_pos_real(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Return only the magnitude of the real part of the quaternion triple product.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        c: shape (..., 4) quaternions in form (w, x, y, z)

    Returns:
        a*b*c np.ndarray shape (..., ) of quaternion triple product real part magnitudes.
    """
    return qu_prod_pos_real(a, qu_prod(b, c))


# Misorientations


def qu_misorientation(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Calculates the misorientation quaternion between two quaternions.

    Args:
        q1: shape (..., 4) quaternions
        q2: shape (..., 4) quaternions

    Returns:
        The misorientation quaternion, a np.ndarray of shape (..., 4)
    """
    return qu_prod(q1, qu_conj(q2))


def qu_disorientation(
    quats1: np.ndarray, quats2: np.ndarray, laue_id_1: int, laue_id_2: int, naive=True
):
    """

    Return the disorientation quaternion between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id_1: laue group ID of quats1
        laue_id_2: laue group ID of quats2
        naive: whether to use the naive method or the more accurate method

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # get the important shapes
    data_shape = quats2.shape

    # check that the shapes are the same
    if data_shape == (4,):
        data_shape = (1, 4)
        quats1 = quats1.reshape(data_shape)
    elif data_shape == (1, 4):
        pass
    elif data_shape == quats1.shape:
        pass
    else:
        raise ValueError(
            f"quats1 and quats2 must have the same data shape, or quats1 must be a single quaternion, but got {data_shape} and {quats2.shape}"
        )
    if not ((quats1.dtype == np.float64) or (quats1.dtype == np.float32)):
        raise ValueError("Quaternions must be of type float32 or float64")
    if not ((quats2.dtype == np.float64) or (quats2.dtype == np.float32)):
        raise ValueError("Quaternions must be of type float32 or float64")

    # multiply by inverse of second (without symmetry)
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # find the number of quaternions (generic input shapes are supported)
    N = int(np.prod(np.array(data_shape[:-1])))

    # retrieve the laue group elements for the first quaternions
    laue_group_1 = laue_elements(laue_id_1)
    # laue_group_1 = np.vstack((laue_group_1, qu_conj(laue_group_1)))

    # if the laue groups are the same, then the second laue group is the same as the first
    if laue_id_1 == laue_id_2:
        laue_group_2 = laue_group_1
    else:
        laue_group_2 = laue_elements(laue_id_2)
        # laue_group_2 = np.vstack((laue_group_2, qu_conj(laue_group_2)))

    # pre / post mult by Laue operators of the second and first symmetry groups respectively
    # broadcasting is done so that the output is of shape (N, |laue_group_2|, |laue_group_1|, 4)
    equivalent_quaternions = qu_prod_raw(
        laue_group_2.reshape(1, -1, 1, 4),
        qu_prod_raw(misori_quats.reshape((N, 1, 1, 4)), laue_group_1.reshape((1, 1, -1, 4))),
    )

    # flatten along the laue group dimensions
    equivalent_quaternions = equivalent_quaternions.reshape(N, -1, 4)
    equivalent_quaternions = np.unique(equivalent_quaternions, axis=1)

    # find the quaternion with the largest real part value (smallest angle)
    row_maximum_indices = np.argmax(
        np.abs(equivalent_quaternions[..., 0]),
        axis=-1,
    )

    # if naive, just grab the first quaternion that has the largest real part value
    if naive:
        output = equivalent_quaternions[np.arange(N), row_maximum_indices]

    # if not naive, then we find the quaternion with the largest real part and with the axis in the fundamental sector
    else:
        output = np.zeros((N, 4), dtype=quats1.dtype)
        abs_scalars = np.abs(equivalent_quaternions[..., 0])
        for i in tqdm(range(N)):
            mask = np.isclose(
                abs_scalars[i],
                abs_scalars[i, row_maximum_indices[i]],
                rtol=1e-6,
                atol=1e-6,
            )
            out_equivalent = qu_norm_std(equivalent_quaternions[i][mask]) + 0.0
            if [1.0, 0.0, 0.0, 0.0] in out_equivalent.tolist():
                print("Found identity quaternion")
                output[i] = [1.0, 0.0, 0.0, 0.0]
                continue
            angles = qu_angle(out_equivalent)
            axes = qu_axis(out_equivalent)
            angles[axes[..., 2] < 0] *= -1
            axes[axes[..., 2] < 0] *= -1
            mask1 = (axes[..., 0] >= 0) & (axes[..., 1] >= 0) & (axes[..., 2] >= 0)  # all positive
            mask2 = (axes[..., 2] >= axes[..., 1]) & (
                axes[..., 1] >= axes[..., 0]
            )  # ascending order
            mask3 = mask1 & mask2
            if mask3.sum() == 0:
                print(out_equivalent)
                print("Number of minimum angle quaternions", out_equivalent.shape[0])
                print(axes)
                print("mask1", mask1.sum())
                print("mask2", mask2.sum())
                print("mask all", mask3.sum())
                print(equivalent_quaternions[i][mask][mask3])
                print(out_equivalent[mask3])
                print(axes[mask3])
                print(angles[mask3])
                raise ValueError("No equivalent quaternion found in the fundamental sector")
            output[i] = out_equivalent[mask3][0]

    return qu_norm_std(output.reshape(data_shape))


def qu_disorientation_directional(quats1: np.ndarray, quats2: np.ndarray, laue_id: int):
    """

    Return the disorientation quaternion between the given quaternions.

    Args:
        quats1: quaternions of shape (..., 4)
        quats2: quaternions of shape (..., 4)
        laue_id: laue group ID of quats

    Returns:
        disorientation quaternion of shape (..., 4)

    """

    # get the important shapes
    data_shape = quats2.shape

    # check that the shapes are the same
    if data_shape == (4,):
        data_shape = (1, 4)
        quats1 = quats1.reshape(data_shape)
    elif data_shape == (1, 4):
        pass
    elif data_shape == quats1.shape:
        pass
    else:
        raise ValueError(
            f"quats1 and quats2 must have the same data shape, or quats1 must be a single quaternion, but got {data_shape} and {quats2.shape}"
        )
    if not ((quats1.dtype == np.float64) or (quats1.dtype == np.float32)):
        raise ValueError("Quaternions must be of type float32 or float64")
    if not ((quats2.dtype == np.float64) or (quats2.dtype == np.float32)):
        raise ValueError("Quaternions must be of type float32 or float64")

    # find the number of quaternions (generic input shapes are supported)
    N = int(np.prod(np.array(data_shape[:-1])))

    # retrieve the laue group elements for the first quaternions
    laue_group = laue_elements(laue_id)

    # symmetrize the quaternions
    quats2 = qu_prod_raw(laue_group, quats2.reshape(N, 1, 4))

    # get misorientation quaternions
    misori_quats = qu_prod(quats1, qu_conj(quats2))

    # Collapse to only the unique quaternions
    misori_quats = np.unique(misori_quats, axis=1)

    # find the quaternion with the largest real part value (smallest angle)
    row_maximum_indices = np.argmax(np.abs(misori_quats[..., 0]), axis=-1)

    # if naive, just grab the first quaternion that has the largest real part value
    disori_quats = misori_quats[np.arange(N), row_maximum_indices]
    return qu_std(disori_quats.reshape(data_shape))


# Advanced / Unique operations


def qu_slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        a: shape (..., 4) quaternions in form (w, x, y, z)
        b: shape (..., 4) quaternions in form (w, x, y, z)
        t: interpolation parameter between 0 and 1

    Returns:
        The interpolated quaternions, a array of shape (..., 4).
    """
    a = qu_norm(a)
    b = qu_norm(b)
    cos_theta = np.sum(a * b, axis=-1)
    angle = np.acos(cos_theta)
    sin_theta = np.sin(angle)
    w1 = np.sin((1 - t) * angle) / sin_theta
    w2 = np.sin(t * angle) / sin_theta
    return a * w1[..., None] + b * w2[..., None]


def qu_log(q: np.ndarray, tol=1e-6) -> np.ndarray:
    """Logarithm of a quaternion.
    log(q) = [0, theta*n] where q = [cos(theta), sin(theta)*n]
    quaternion should be scalar first, vector second.
    """
    # Make sure the quaternion is a unit quaternion
    q = qu_norm_std(q)
    # Separate into scalar and vector components
    s, v = q[..., 0], q[..., 1:]
    # Get the angle
    theta = np.arccos(s)
    # Use the angle to get the rotation vector
    norm_v = np.linalg.norm(v, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        qlog = v * np.where(norm_v > tol, theta / norm_v, 0).reshape(q.shape[:-1] + (1,))
    return qlog


def qu_avg(q: np.ndarray, laue_id, chunk_size=None) -> np.ndarray:
    """Calculates the average quaternion from a set of quaternions.

    Args:
        q: shape (N, 4) quaternions
        laue_id: integer between inclusive [1, 11]

    Returns:
        The average quaternion, a np.ndarray of shape (4,)
    """
    S = laue_elements(laue_id)
    q = qu_norm_std(q)
    q0, qn = q[0], q[1:]
    if chunk_size is not None:
        chunk_size = min(chunk_size, qn.shape[0])
        idx = np.random.choice(qn.shape[0], size=chunk_size, replace=False)
        qn = qn[idx]
    qn_sym = qu_prod(S[None], qn[:, None])
    dots = np.abs(q0.dot(qn_sym.transpose(0, 2, 1)))
    idx = dots.argmax(axis=1)
    qn_close = qn_sym[np.arange(qn_sym.shape[0]), idx]
    return qn_close.mean(axis=0)


# Rotations


def qu_apply(qu: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    Rotate 3D points by unit quaternions.

    Args:
        qu: shape (..., 4) of quaternions in the form (w, x, y, z)
        point: shape (..., 3) of 3D points.

    Returns:
        np.ndarray of rotated points of shape (..., 3).
    """
    point_as_quaternion = np.concatenate([np.zeros_like(point[..., :1]), point], axis=-1)
    qu_i = qu_prod_raw(qu, point_as_quaternion)
    qu_o = qu_prod_raw(qu_i, qu_conj(qu))
    v_out = qu_o[..., 1:]  # Extract the vector part

    return v_out


def qu_rotate_sets_sphere(points_start: np.ndarray, points_finish) -> np.ndarray:
    """
    Determine the quaternions that rotate the points_start to the points_finish.
    All points are assumed to be on the unit sphere. The cross product is used
    as the axis of rotation, but there are an infinite number of quaternions that
    fulfill the requirement as the points can be rotated around their axis by
    an arbitrary angle, and they will still have the same latitude and longitude.

    Args:
        points_start: Starting points as array of shape (..., 3).
        points_finish: Ending points as array of shape (..., 3).

    Returns:
        The quaternions, as array of shape (..., 4).

    """
    # determine mask for numerical stability
    valid = np.abs(np.sum(points_start * points_finish, axis=-1)) < 0.999999
    # get the cross product of the two sets of points
    cross = np.cross(points_start[valid], points_finish[valid], axis=-1)
    # get the dot product of the two sets of points
    dot = np.sum(points_start[valid] * points_finish[valid], axis=-1)
    # get the angle
    angle = np.atan2(np.linalg.norm(cross, axis=-1), dot)
    # add tau to the angle if the cross product is negative
    angle[angle < 0] += 2 * np.pi
    # set the output
    out = np.zeros((points_start.shape[0], 4), dtype=points_start.dtype)
    out[valid, 0] = np.cos(angle / 2)
    out[valid, 1:] = np.sin(angle / 2)[..., None] * (
        cross / np.linalg.norm(cross, axis=-1, keepdims=True)
    )
    out[~valid, 0] = 1
    out[~valid, 1:] = 0
    return out


def ori_to_fz_laue(quats: np.ndarray, laue_id: int) -> np.ndarray:
    """
    This function moves the given quaternions to the fundamental zone of the
    given Laue group. This computes the orientation fundamental zone, not the
    misorientation fundamental zone.

    Args:
        quats: quaternions to move to fundamental zone of shape (..., 4)
        laue_id: laue group of quaternions to move to fundamental zone

    Returns:
        orientations in fundamental zone of shape (..., 4)

    Notes:

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

    """
    # get the important shapes
    data_shape = quats.shape
    N = np.prod(np.array(data_shape[:-1]))
    laue_group = laue_elements(laue_id)
    card = laue_group.shape[0]

    # reshape so that quaternions is (N, 1, 4) and laue_group is (1, card, 4) then use broadcasting
    equivalent_quaternions_real = qu_prod_pos_real(
        quats.reshape(N, 1, 4), laue_group.reshape(card, 4)
    )

    # find the quaternion with the largest w value
    row_maximum_indices = np.argmax(equivalent_quaternions_real, axis=-1)

    # gather the equivalent quaternions with the largest w value for each equivalent quaternion set
    output = qu_prod(quats.reshape(N, 4), laue_group[row_maximum_indices])

    return output.reshape(data_shape)


def get_preferred_rotation_axis(q_dis, chunk_size=None) -> np.ndarray:
    """Calculate the preferred rotation axis from a set of disorientation quaternions.
    The preferred rotation axis is the eigenvector of the Q tensor corresponding to the largest eigenvalue.
    The preferred rotation axis is used to calculate the sign carrying disorientation angle.

    Args:
        q_dis: shape (N, 4) of disorientation quaternions

    Returns:
        r_star: shape (3,) of the preferred rotation axis
    """
    Q = get_Q_tensor(q_dis, chunk_size=chunk_size)
    evals, evecs = np.linalg.eig(Q)
    evecs = np.real(evecs[:, np.argsort(evals)[::-1]])
    evals = np.real(evals[np.argsort(evals)[::-1]])
    r_star = evecs[:, 0]
    r_star = r_star / np.linalg.norm(r_star)
    return r_star


def get_Q_tensor(q_dis, chunk_size=None) -> np.ndarray:
    """Calculate the Q tensor from a set of disorientation quaternions.
    The Q tensor is a 3x3 symmetric tensor of the second order central moments of disorientations.
    The Q tensor is used to calculate the reference axis of disorientation.

    Args:
        q_dis: shape (N, 4) of disorientation quaternions

    Returns:
        Q: shape (3, 3) of the Q tensor
    """
    if chunk_size is not None:
        chunk_size = min(chunk_size, q_dis.shape[0])
        idx = np.random.choice(q_dis.shape[0], size=chunk_size, replace=False)
        q_dis = q_dis[idx]
    if q_dis.shape[0] > 10000:
        q_dis_s = np.array_split(q_dis, q_dis.shape[0] // 1000, axis=0)
        Q_s = Parallel(n_jobs=5)(
            delayed(np.einsum)("ij,ik->jk", q_dis_s[i][..., 1:], q_dis_s[i][..., 1:])
            for i in range(len(q_dis_s))
        )
        Q = np.sum(Q_s, axis=0) / q_dis.shape[0]
    else:
        v = q_dis[..., 1:]
        Q = np.einsum("ij,ik->jk", v, v) / q_dis.shape[0]
    return Q


def get_sign_carrying_disorientation_angle(q_dis, chunk_size=None, r_star=None) -> np.ndarray:
    """Calculate the sign carrying disorientation angle from a set of disorientation quaternions.
    As opposed to the disorientation angle, the sign carrying disorientation angle is maintains
    the direction around the rotation axis, using a global preferred rotation axis.

    Args:
        q_dis: shape (N, 4) of disorientation quaternions
        method: method to use for calculating the angle, either "refaxis" or "axangle"
            "refaxis": use the preferred rotation axis to calculate the angle
            "axangle": use the axis-angle representation to calculate the angle

    Returns:
        angles: shape (N,) of the sign carrying disorientation angles
    """
    if r_star is None:
        r_star = get_preferred_rotation_axis(q_dis, chunk_size=chunk_size)
    with np.errstate(divide="ignore", invalid="ignore"):
        angles = (
            np.arccos(q_dis[..., 0]) / np.sqrt(1 - q_dis[..., 0] ** 2) * q_dis[..., 1:].dot(r_star)
        )
    angles[np.isnan(angles)] = 0
    angles[np.isinf(angles)] = 0

    return angles


# Symmetry


def symmetrize(q: np.ndarray, laue_id: int) -> np.ndarray:
    """Symmetrizes a quaternion using the Laue group.

    Args:
        q: shape (N, 4) quaternions
        laue_id: integer between inclusive [1, 11]

    Returns:
        The symmetrized quaternions, a np.ndarray of shape (N, |Laue group|, 4)
    """
    S = laue_elements(laue_id)
    q = qu_norm_std(q)
    q = q.reshape((-1, 1, 4))
    S = S.reshape((1, -1, 4))
    q_sym = qu_prod(S[None], q[:, None]).reshape((q.shape[0], -1, 4))
    return q_sym


def laue_elements(laue_id: int) -> np.ndarray:
    """
    Generators for Laue group specified by the laue_id parameter. The first
    element is always the identity.

    1) Laue C1       Triclinic: 1-, 1
    2) Laue C2      Monoclinic: 2/m, m, 2
    3) Laue D2    Orthorhombic: mmm, mm2, 222
    4) Laue C4  Tetragonal low: 4/m, 4-, 4
    5) Laue D4 Tetragonal high: 4/mmm, 4-2m, 4mm, 422
    6) Laue C3    Trigonal low: 3-, 3
    7) Laue D3   Trigonal high: 3-m, 3m, 32
    8) Laue C6   Hexagonal low: 6/m, 6-, 6
    9) Laue D6  Hexagonal high: 6/mmm, 6-m2, 6mm, 622
    10) Laue T       Cubic low: m3-, 23
    11) Laue O      Cubic high: m3-m, 4-3m, 432

    Args:
        laue_id: integer between inclusive [1, 11]

    Returns:
        np array of shape (cardinality, 4) containing the elements of the

    Notes:

    https://en.wikipedia.org/wiki/Space_group

    """

    # sqrt(2) / 2 and sqrt(3) / 2
    R2 = 1.0 / (2.0**0.5)
    R3 = (3.0**0.5) / 2.0

    LAUE_O = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [0.0, 0.0, 0.0, 1.0],
            [-R2, 0.0, 0.0, R2],  #
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, R2, R2],
            [-0.5, -0.5, 0.5, 0.5],  #
            [-R2, -R2, 0.0, 0.0],  #
            [0.0, 1.0, 0.0, 0.0],
            [0.0, R2, R2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -R2, R2, 0.0],
            [-0.5, 0.5, 0.5, -0.5],  #
            [0.0, 0.0, R2, -R2],  #
            [0.5, -0.5, 0.5, -0.5],
            [R2, -R2, 0.0, 0.0],
            [0.0, R2, 0.0, R2],
            [-0.5, 0.5, 0.5, 0.5],  #
            [-R2, 0.0, R2, 0.0],  #
            [-0.5, -0.5, 0.5, -0.5],  #
            [0.0, R2, 0.0, -R2],  #
            [0.5, 0.5, 0.5, -0.5],
            [R2, 0.0, R2, 0.0],
            [0.5, -0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    LAUE_O_alt = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, -R2],  #
            [0.5, 0.5, 0.5, 0.5],
            [0.0, 0.0, R2, R2],
            [0.5, 0.5, -0.5, -0.5],  #
            [R2, R2, 0.0, 0.0],  #
            [0.0, 1.0, 0.0, 0.0],
            [0.0, R2, R2, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.5, -0.5, -0.5, 0.5],  #
            [0.0, 0.0, -R2, R2],  #
            [0.5, -0.5, 0.5, -0.5],
            [R2, -R2, 0.0, 0.0],
            [0.0, R2, 0.0, R2],
            [0.5, -0.5, -0.5, -0.5],  #
            [R2, 0.0, -R2, 0.0],  #
            [0.5, 0.5, -0.5, 0.5],  #
            [0.0, -R2, 0.0, R2],  #
            [0.5, 0.5, 0.5, -0.5],
            [R2, 0.0, R2, 0.0],
            [0.5, -0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )
    LAUE_O_ = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [R2, R2, 0.0, 0.0],
            [R2, 0.0, R2, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, -R2, 0.0, 0.0],
            [R2, 0.0, -R2, 0.0],
            [R2, 0.0, 0.0, -R2],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
            [0.0, 0.0, R2, R2],
            [0.0, 0.0, -R2, R2],
            [0.0, R2, 0.0, R2],
            [0.0, -R2, 0.0, R2],
        ],
        dtype=np.float64,
    )
    LAUE_T = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.5, 0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5, 0.5],
        ],
        dtype=np.float64,
    )

    LAUE_D6 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
            [0.0, R3, 0.5, 0.0],
            [0.0, -R3, 0.5, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    LAUE_C6 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 0.0, 0.0, 1.0],
            [R3, 0.0, 0.0, 0.5],
            [R3, 0.0, 0.0, -0.5],
        ],
        dtype=np.float64,
    )

    LAUE_D3 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -0.5, R3, 0.0],
            [0.0, 0.5, R3, 0.0],
        ],
        dtype=np.float64,
    )

    LAUE_C3 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, R3],
            [0.5, 0.0, 0.0, -R3],
        ],
        dtype=np.float64,
    )

    LAUE_D4 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
            [0.0, R2, R2, 0.0],
            [0.0, -R2, R2, 0.0],
        ],
        dtype=np.float64,
    )

    LAUE_C4 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [R2, 0.0, 0.0, R2],
            [R2, 0.0, 0.0, -R2],
        ],
        dtype=np.float64,
    )

    LAUE_D2 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )

    LAUE_C2 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    LAUE_C1 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    LAUE_GROUPS = [
        LAUE_C1,  #  1 - Triclinic
        LAUE_C2,  #  2 - Monoclinic
        LAUE_D2,  #  3 - Orthorhombic
        LAUE_C4,  #  4 - Tetragonal low
        LAUE_D4,  #  5 - Tetragonal high
        LAUE_C3,  #  6 - Trigonal low
        LAUE_D3,  #  7 - Trigonal high
        LAUE_C6,  #  8 - Hexagonal low
        LAUE_D6,  #  9 - Hexagonal high
        LAUE_T,  #  10 - Cubic low
        LAUE_O,  #  11 - Cubic high
    ]

    return LAUE_GROUPS[laue_id - 1]
