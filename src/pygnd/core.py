import warnings
from pathlib import Path
from collections.abc import Iterable

import numpy as np
from tqdm import tqdm
from scipy import optimize
from joblib import Parallel, delayed

from pygnd import io, rotations, quaternions
from pygnd.utils import tqdm_joblib


warnings.filterwarnings("ignore", category=UserWarning)

PRECISION = np.float32


def get_linear_operator(
    cs: int, slip_systems: str = "all"
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-calculate the A matrix for the given crystal structure and desired slip systems.

    Args:
        cs (int): The crystal structure of the material. 1 for FCC, 2 for BCC, 3 for HCP.
        slip_systems (str, optional): The slip systems to be used. Defaults to 'all'.
                                      (FCC) - unused, always 'all'
                                      (BCC) - 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'screw+110+123', 'screw+112+123', 'all'
                                      (HCP) - 'basal', 'prismatic', 'pyramidal', 'basal+prismatic', 'basal+pyramidal', 'prismatic+pyramidal', 'all'

    Returns:
        A (np.ndarray): The A matrix for the given crystal structure and slip systems. Shape (9, n_slip_systems)
        B (np.ndarray): The B matrix (psuedo-inverse of A) for the given crystal structure. Shape (n_slip_systems, 9)
    """
    # Check the input values
    if type(cs) != int:
        raise ValueError("Crystal structure must be an integer value.")
    if cs not in [1, 2, 3]:
        raise ValueError("Crystal structure must be 1, 2, or 3.")
    if type(slip_systems) != str:
        raise ValueError("Slip systems must be a string.")
    slip_systems = slip_systems.lower().strip()
    if slip_systems not in [
        "all",
        "screw+110",
        "screw+112",
        "screw+123",
        "screw+110+112",
        "basal",
        "basal+prismatic",
    ]:
        raise ValueError(
            "Slip systems must be 'all', 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'basal', 'basal+prismatic', depending on the crystam structure."
        )

    # Create the A matrix for the given crystal structure
    if cs == 1:
        a = np.sqrt(3).astype(PRECISION) / 9
        c = np.sqrt(3).astype(PRECISION) / 84
        d = 1 / 18
        f = 3 / 14

        # See Arsenlis & Parks 1999
        B = np.array(
            [
                [a, 7 * c, -13 * c, -7 * c, -a, 13 * c, c, -c, 0],
                [-a, 13 * c, -7 * c, -c, 0, c, 7 * c, -13 * c, a],
                [0, c, -c, -13 * c, a, 7 * c, 13 * c, -7 * c, -a],
                [a, -7 * c, 13 * c, 7 * c, -a, 13 * c, -c, -c, 0],
                [-a, -13 * c, 7 * c, c, 0, c, -7 * c, -13 * c, a],
                [0, -c, c, 13 * c, a, 7 * c, -13 * c, -7 * c, -a],
                [a, -7 * c, -13 * c, 7 * c, -a, -13 * c, c, c, 0],
                [-a, -13 * c, -7 * c, c, 0, -c, 7 * c, 13 * c, a],
                [0, -c, -c, 13 * c, a, -7 * c, 13 * c, 7 * c, -a],
                [a, 7 * c, 13 * c, -7 * c, -a, -13 * c, -c, c, 0],
                [-a, 13 * c, 7 * c, -c, 0, -c, -7 * c, 13 * c, -a],
                [0, c, c, -13 * c, a, -7 * c, -13 * c, 7 * c, -a],
                [5 * d, f, 0, f, 5 * d, 0, 0, 0, -d],
                [5 * d, 0, f, 0, -d, 0, f, 0, 5 * d],
                [-d, 0, 0, 0, 5 * d, f, 0, f, 5 * d],
                [5 * d, -f, 0, -f, 5 * d, 0, 0, 0, -d],
                [5 * d, 0, -f, 0, -d, 0, -f, 0, 5 * d],
                [-d, 0, 0, 0, 5 * d, -f, 0, -f, 5 * d],
            ]
        ).astype(PRECISION)

        # FCC
        A = pseudo_inverse(B)

    elif cs == 2:
        # BCC
        A = generate_BCC_A_matrix()
        if slip_systems == "screw+110":
            A = A[:, :16]
        elif slip_systems == "screw+112":
            A = np.hstack((A[:, :4], A[:, 16:28]))
        elif slip_systems == "screw+123":
            A = np.hstack((A[:, :4], A[:, 28:]))
        elif slip_systems == "screw+110+112":
            A = A[:, :28]
        elif slip_systems == "screw+110+123":
            A = np.hstack((A[:, :16], A[:, 28:]))
        elif slip_systems == "screw+112+123":
            A = np.hstack((A[:, :4], A[:, 16:]))
        elif slip_systems == "all":
            pass
        B = pseudo_inverse(A)

    elif cs == 3:
        # HCP
        A = generate_HCP_A_matrix()
        if slip_systems == "basal":
            A = A[:, :6]  # 3 edge basal and 3 screw basal slip systems
        elif slip_systems == "prismatic":
            A = A[:, 6:9]  # 3 edge prismatic slip systems
        elif slip_systems == "pyramidal":
            A = A[:, 9:]  # 12 edge pyramidal and 12 screw pyramidal slip systems
        elif slip_systems == "basal+prismatic":
            A = A[:, :9]
        elif slip_systems == "basal+pyramidal":
            A = np.hstack((A[:, :6], A[:, 9:]))
        elif slip_systems == "prismatic+pyramidal":
            A = A[:, 6:]
        elif slip_systems == "all":
            pass
        B = pseudo_inverse(A)

    # Add this diagnostic
    # print(f"Condition number of A: {np.linalg.cond(A)}")
    # print(f"Rank of A: {np.linalg.matrix_rank(A)}")
    # print(f"||B @ A - I||_F: {np.linalg.norm(B @ A - np.eye(A.shape[1]), 'fro')}")
    # print(f"||A @ B - I||_F: {np.linalg.norm(A @ B - np.eye(A.shape[0]), 'fro')}")
    return (A, B)


def generate_BCC_A_matrix() -> np.ndarray:
    """Generate the A matrix for BCC crystal structure."""
    # Burgers vectors and slip plane normals for BCC
    b_n = np.array(
        [
            [[1, 1, -1], [1, 1, -1]],  # <111> screw
            [[1, -1, -1], [1, -1, -1]],
            [[1, -1, 1], [1, -1, 1]],
            [[1, 1, 1], [1, 1, 1]],
            [[1, 1, -1], [0, 1, 1]],  # {110}<111> edge
            [[1, 1, -1], [1, 0, 1]],
            [[1, 1, -1], [1, -1, 0]],
            [[1, -1, -1], [0, 1, -1]],
            [[1, -1, -1], [1, 0, 1]],
            [[1, -1, -1], [1, 1, 0]],
            [[1, -1, 1], [0, 1, 1]],
            [[1, -1, 1], [1, 0, -1]],
            [[1, -1, 1], [1, 1, 0]],
            [[1, 1, 1], [0, 1, -1]],
            [[1, 1, 1], [1, 0, -1]],
            [[1, 1, 1], [1, -1, 0]],
            [[-1, -1, 1], [-2, 1, -1]],  # {112}<111> edge
            [[-1, -1, 1], [1, -2, -1]],
            [[-1, -1, 1], [1, 1, 2]],
            [[-1, 1, 1], [-2, -1, -1]],
            [[-1, 1, 1], [1, 2, -1]],
            [[-1, 1, 1], [1, -1, 2]],
            [[1, -1, 1], [2, 1, -1]],
            [[1, -1, 1], [-1, -2, -1]],
            [[1, -1, 1], [-1, 1, 2]],
            [[1, 1, 1], [2, -1, -1]],
            [[1, 1, 1], [-1, 2, -1]],
            [[1, 1, 1], [-1, -1, 2]],
            [[1, 1, -1], [1, 2, 3]],  # {123}<111> edge
            [[1, 1, -1], [-1, 3, 2]],
            [[1, 1, -1], [2, 1, 3]],
            [[1, 1, -1], [-2, 3, 1]],
            [[1, 1, -1], [3, -1, 2]],
            [[1, 1, -1], [3, -2, 1]],
            [[1, -1, -1], [-1, 2, -3]],
            [[1, -1, -1], [1, 3, -2]],
            [[1, -1, -1], [2, -1, 3]],
            [[1, -1, -1], [2, 3, -1]],
            [[1, -1, -1], [3, 1, 2]],
            [[1, -1, -1], [3, 2, 1]],
            [[1, -1, 1], [1, -2, -3]],
            [[1, -1, 1], [1, 3, 2]],
            [[1, -1, 1], [2, -1, -3]],
            [[1, -1, 1], [2, 3, 1]],
            [[1, -1, 1], [3, 1, -2]],
            [[1, -1, 1], [3, 2, -1]],
            [[1, 1, 1], [1, 2, -3]],
            [[1, 1, 1], [1, -3, 2]],
            [[1, 1, 1], [2, 1, -3]],
            [[1, 1, 1], [2, -3, 1]],
            [[1, 1, 1], [-3, 1, 2]],
            [[1, 1, 1], [-3, 2, 1]],
        ]
    ).astype(float)
    burgers = b_n[:, 0] / np.sqrt(3)
    normals = b_n[:, 1] / np.linalg.norm(b_n[:, 1], axis=1)[:, None]

    # Get the sense vectors
    t = np.cross(normals, burgers)

    # Fix the screw dislocations (sense vectors are the burgers vectors)
    t[:4] = burgers[:4]

    # Calculate the outer product of the two vectors
    outer = np.einsum("...i,...j->...ij", burgers, t)

    # Convert to the (n_slip_systems, 9) matrix
    A_bcc = outer.reshape(-1, 9).T

    return A_bcc


def generate_HCP_A_matrix() -> np.ndarray:
    """Generate the A matrix for HCP crystal structure."""
    # Relevant Direcitons in [uvtw] notation
    b_n_uvtw = np.array(
        [
            [[1, 1, -2, 0], [0, 0, 0, 1]],  # Basal
            [[1, -2, 1, 0], [0, 0, 0, 1]],
            [[-2, 1, 1, 0], [0, 0, 0, 1]],
            [[2, -1, -1, 0], [0, 1, -1, 0]],  # Prismatic
            [[-1, 2, -1, 0], [1, 0, -1, 0]],
            [[1, 1, -2, 0], [1, -1, 0, 0]],
            [[-1, -1, 2, 3], [1, 0, -1, 1]],  # Pyramidal
            [[-2, 1, 1, 3], [1, 0, -1, 1]],
            [[1, 1, -2, 3], [0, -1, 1, 1]],
            [[-1, 2, -1, 3], [0, -1, 1, 1]],
            [[2, -1, -1, 3], [-1, 1, 0, 1]],
            [[1, -2, 1, 3], [-1, 1, 0, 1]],
            [[2, -1, -1, 3], [-1, 0, 1, 1]],
            [[1, 1, -2, 3], [-1, 0, 1, 1]],
            [[-1, -1, 2, 3], [0, 1, -1, 1]],
            [[1, -2, 1, 3], [0, 1, -1, 1]],
            [[-2, 1, 1, 3], [1, -1, 0, 1]],
            [[-1, 2, -1, 3], [1, -1, 0, 1]],
        ]
    ).astype(float)

    # Convert to uvw
    u = b_n_uvtw[:, 0, 0]
    v = b_n_uvtw[:, 0, 1]
    t = b_n_uvtw[:, 0, 2]
    w = b_n_uvtw[:, 0, 3]
    burgers = np.array([u - t, v - t, w]).T
    burgers /= np.linalg.norm(burgers, axis=1)[:, None]

    u = b_n_uvtw[:, 1, 0]
    v = b_n_uvtw[:, 1, 1]
    t = b_n_uvtw[:, 1, 2]
    w = b_n_uvtw[:, 1, 3]
    normals = np.array([u - t, v - t, w]).T
    normals /= np.linalg.norm(normals, axis=1)[:, None]

    # Get the sense vectors
    t = np.cross(normals, burgers)

    # Put in the screw dislocations
    burgers = np.vstack((burgers[:3], burgers))  # 3 screw basal dislocations
    t = np.vstack((burgers[:3], t))
    burgers = np.vstack((burgers, burgers[-12:]))  # 12 screw pyramidal dislocations
    t = np.vstack((t, burgers[-12:]))

    # Calculate the outer product of the two vectors
    outer = np.einsum("...i,...j->...ij", burgers, t)
    outer = outer.reshape(-1, 9).T
    outer = outer / np.linalg.norm(outer, axis=0)

    return outer


def pseudo_inverse(A: np.ndarray) -> np.ndarray:
    """Calculate the B matrix (psuedo-inverse of A) for the given A matrix.

    Args:
        A (np.ndarray): The A matrix. Shape (9, n_slip_systems)

    Returns:
        np.ndarray: The B matrix. Shape (n_slip_systems, 9)
    """
    # return A.T.dot(np.linalg.inv(A.dot(A.T))).astype(PRECISION)
    return np.linalg.pinv(A).astype(PRECISION)


def get_completeness(grain_ids: np.ndarray) -> np.ndarray:
    """
    Vectorized version of neighborhood analysis for 3D EBSD dataset.

    Args:
        grain_ids (np.ndarray): The grain ID map. Shape (n_x, n_y, n_z)

    Returns:
        np.ndarray: The completeness array. Shape (n_x, n_y, n_z, 3)
    """
    shape = grain_ids.shape

    # Initialize output array
    completeness = np.zeros((*shape, 3), dtype=np.int8)

    # Create masks for valid grain IDs
    valid_grains = grain_ids != 0

    # Compute transitions (now using broadcasting)
    x_trans = np.ones(shape, dtype=bool)
    y_trans = np.ones(shape, dtype=bool)
    z_trans = np.ones(shape, dtype=bool)

    if shape[0] > 1:
        x_trans = np.pad(
            grain_ids[:-1, ...] != grain_ids[1:, ...], ((0, 1), (0, 0), (0, 0))
        )
    if shape[1] > 1:
        y_trans = np.pad(
            grain_ids[:, :-1, :] != grain_ids[:, 1:, :], ((0, 0), (0, 1), (0, 0))
        )
    if shape[2] > 1:
        z_trans = np.pad(
            grain_ids[..., :-1] != grain_ids[..., 1:], ((0, 0), (0, 0), (0, 1))
        )

    # Interior points
    interior_mask = np.zeros_like(grain_ids, dtype=bool)
    interior_mask[1:-1, :, :] = True

    # X-direction vectorized analysis
    if shape[0] > 1:
        completeness[0, :, :, 0] = np.where(
            x_trans[0, :, :], 0, 1
        )  # Forward differences for first slice
        completeness[-1, :, :, 0] = np.where(
            x_trans[-2, :, :], 0, 2
        )  # Backward differences for last slice
        completeness[1:-1, :, :, 0] = np.select(  # Central differences
            [
                (x_trans[:-2, :, :] & x_trans[1:-1, :, :]),
                x_trans[:-2, :, :],
                x_trans[1:-1, :, :],
            ],
            [0, 1, 2],
            default=3,
        )

    # Y-direction vectorized analysis
    if shape[1] > 1:
        completeness[:, 0, :, 1] = np.where(y_trans[:, 0, :], 0, 1)
        completeness[:, -1, :, 1] = np.where(y_trans[:, -2, :], 0, 2)
        completeness[:, 1:-1, :, 1] = np.select(
            [
                (y_trans[:, :-2, :] & y_trans[:, 1:-1, :]),
                y_trans[:, :-2, :],
                y_trans[:, 1:-1, :],
            ],
            [0, 1, 2],
            default=3,
        )

    # Z-direction (similar logic)
    if shape[2] > 1:
        completeness[:, :, 0, 2] = np.where(z_trans[:, :, 0], 0, 1)
        completeness[:, :, -1, 2] = np.where(z_trans[:, :, -2], 0, 2)
        completeness[:, :, 1:-1, 2] = np.select(
            [
                (z_trans[:, :, :-2] & z_trans[:, :, 1:-1]),
                z_trans[:, :, :-2],
                z_trans[:, :, 1:-1],
            ],
            [0, 1, 2],
            default=3,
        )

    # Zero out values where grain_id is 0
    completeness[~valid_grains] = 0

    return completeness


def get_neighbors(completeness: np.ndarray) -> np.ndarray:
    """Convert the completeness array into a list of neighbor indices."""
    # Get the shape of the completeness array
    shape = completeness.shape[:-1]

    # Create coordinate shifts for the pairs and the scale for the finite difference calculation
    shifts0 = np.zeros((3,) + shape + (3,), dtype=np.int32)
    shifts1 = np.zeros((3,) + shape + (3,), dtype=np.int32)
    scale = np.zeros(shape + (3,), dtype=PRECISION)

    # Only central and backward differences will have a shift in the first point
    shifts0[0][(completeness[..., 0] == 2) | (completeness[..., 0] == 3)] = [-1, 0, 0]
    shifts0[1][(completeness[..., 1] == 2) | (completeness[..., 1] == 3)] = [0, -1, 0]
    shifts0[2][(completeness[..., 2] == 2) | (completeness[..., 2] == 3)] = [0, 0, -1]

    # Only central and forward differences will have a shift in the second point
    shifts1[0][(completeness[..., 0] == 1) | (completeness[..., 0] == 3)] = [1, 0, 0]
    shifts1[1][(completeness[..., 1] == 1) | (completeness[..., 1] == 3)] = [0, 1, 0]
    shifts1[2][(completeness[..., 2] == 1) | (completeness[..., 2] == 3)] = [0, 0, 1]

    # Create coordinate array
    coords = np.indices(shape).transpose(1, 2, 3, 0)  # (x, y, z, ndim)
    coords0 = np.stack([coords + shift for shift in shifts0], axis=-2)
    coords1 = np.stack([coords + shift for shift in shifts1], axis=-2)

    # Handle forward and backward differences (scale is 1)
    scale[..., 0][(completeness[..., 0] == 1) | (completeness[..., 0] == 2)] = 1
    scale[..., 1][(completeness[..., 1] == 1) | (completeness[..., 1] == 2)] = 1
    scale[..., 2][(completeness[..., 2] == 1) | (completeness[..., 2] == 2)] = 1

    # Handle central differences (scale is 2)
    scale[..., 0][(completeness[..., 0] == 3)] = 2
    scale[..., 1][(completeness[..., 1] == 3)] = 2
    scale[..., 2][(completeness[..., 2] == 3)] = 2

    # Make sure everywhere that has a 0 completeness has a 0 scale
    scale[completeness == 0] = 0

    # coords dimensions
    # 0: x index in volume
    # 1: y index in volume
    # 2: z index in volume
    # 3: the axis of the difference (0: x, 1: y, 2: z)
    # 4: the coordinate of the voxel (0: x, 1: y, 2: z)
    # so coords0[4, 5, 6, 0, 1] is the y-coordinate of the first voxel in the finite difference pair along the x-direction for the voxel at (4, 5, 6)
    # and coords1[4, 5, 6, 0, 1] is the y-coordinate of the second voxel in the pair

    return (coords0, coords1, scale)


def get_finite_difference_coordinates(
    grain_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the coordinates for finite difference pairs in a 3D EBSD dataset.

    Args:
        grain_ids: 3D numpy array containing grain IDs

    Returns:
        tuple of three 3D arrays containing the coordinates of the first voxel, the coordinates of the second voxel, and the scale factor
    """
    # Get the completeness array
    completeness = get_completeness(grain_ids)

    # Get the neighbors
    coords0, coords1, scale = get_neighbors(completeness)

    return (coords0, coords1, scale)


def get_orientation_gradients(
    quats: np.ndarray,
    pts0: np.ndarray,
    pts1: np.ndarray,
    distances: np.ndarray,
    cs: int,
    n_cpus: int = 1,
    chunk_size: int = None,
    progress_bar: bool = False,
) -> np.ndarray:
    """Calculate the orientation gradients for a 3D EBSD dataset.
    This is essentially the rotation vectors corresponding to the disorientation between neighboring voxels,
    divided by the spacing along each dimension. The result is a 3x3 matrix for each voxel.
    This function will call the private function _get_orientation_gradients to do the actual calculations.
    Supports parallel processing.

    Args:
        quats: 3D numpy array containing quaternions, (X, Y, Z, 4)
        pts0: 3D numpy array containing the coordinates of the first voxel in the finite difference pairs, (X, Y, Z, 3)
        pts1: 3D numpy array containing the coordinates of the second voxel in the finite difference pairs, (X, Y, Z, 3)
        distances: 3D numpy array containing the distances between the finite difference pairs, (X, Y, Z, 3)
        cs: The crystal structure of the material. 1 for FCC, 2 for BCC, 3 for HCP.
        n_cpus: The number of CPUs to use for parallel processing. If None, all available CPUs minus one will be used.

    Returns:
        3D numpy array containing the orientation gradients, (X, Y, Z, 3, 3)
          This is essectially 3 rotation vectors corresponding to the disorientation between neighboring voxels,
          each divided by the distance between the finite difference pair. The 3x3 matrix for each point is the rotation vector for each axis.
    """
    # Get the shape
    out_shape = quats.shape[:-1]

    # Reshape the data to be 1D
    quats = quats.reshape(-1, 4).astype(PRECISION)
    N = quats.shape[0]
    pts0 = pts0.reshape(-1, 3, 3)
    pts1 = pts1.reshape(-1, 3, 3)
    distances = distances.reshape(-1, 3)

    # Convert points to raveled indices
    pts0 = np.stack(
        [
            np.ravel_multi_index(pts0[:, 0].T, out_shape),
            np.ravel_multi_index(pts0[:, 1].T, out_shape),
            np.ravel_multi_index(pts0[:, 2].T, out_shape),
        ],
        axis=-1,
    )
    pts1 = np.stack(
        [
            np.ravel_multi_index(pts1[:, 0].T, out_shape),
            np.ravel_multi_index(pts1[:, 1].T, out_shape),
            np.ravel_multi_index(pts1[:, 2].T, out_shape),
        ],
        axis=-1,
    )

    # Get quaternion pairs
    q0 = np.stack(
        [quats[pts0[:, 0]], quats[pts0[:, 1]], quats[pts0[:, 2]]],
        axis=1,
        dtype=PRECISION,
    )  # (n_pairs, 3, 4)
    q1 = np.stack(
        [quats[pts1[:, 0]], quats[pts1[:, 1]], quats[pts1[:, 2]]],
        axis=1,
        dtype=PRECISION,
    )  # (n_pairs, 3, 4)
    del quats, pts0, pts1  # Free memory

    # Get laue_id
    laue_id = 11 if cs == 1 or cs == 2 else 9

    if n_cpus == 1:
        quats_disorientation = quaternions.qu_disorientation(
            q0, q1, laue_id, laue_id
        ).transpose(1, 0, 2)
    else:
        # Setup chunk size
        if chunk_size is None:
            chunk_size = min(quats.shape[0] // n_cpus, quats.shape[0] // 100)

        # Split the data into chunks
        q0 = np.array_split(q0, q0.shape[0] // chunk_size)
        q1 = np.array_split(q1, q1.shape[0] // chunk_size)
        n_chunks = len(q0)
        chunks = zip(q0, q1)

        # Run the calculations in parallel
        quats_disorientation = np.empty((N, 3, 4), dtype=PRECISION)
        if progress_bar:
            with tqdm_joblib(
                tqdm(total=n_chunks, desc="Calculating orientation gradients")
            ) as progress_bar:
                out = Parallel(n_jobs=n_cpus, timeout=9999999)(
                    delayed(quaternions.qu_disorientation)(q0, q1, laue_id, laue_id)
                    for q0, q1 in chunks
                )
        else:
            out = Parallel(n_jobs=n_cpus, timeout=9999999)(
                delayed(quaternions.qu_disorientation)(q0, q1, laue_id, laue_id)
                for q0, q1 in chunks
            )
        del q0, q1  # Free memory

        # Concatenate the results
        start_idx = 0
        for chunk in tqdm(out, desc="Unpacking orientation gradients"):
            end_idx = start_idx + chunk.shape[0]
            quats_disorientation[start_idx:end_idx] = chunk
            start_idx = end_idx
        del out, chunk
        quats_disorientation = quats_disorientation.transpose(1, 0, 2)

    # Convert quaternions to rotation vectors
    rot_vectors = quaternions.qu_log(quats_disorientation) * 2
    del quats_disorientation  # Free memory

    # Get the misorientations from the rotation vectors
    misorientation = np.linalg.norm(rot_vectors, axis=-1)

    # Get the orientation gradients
    with np.errstate(divide="ignore", invalid="ignore"):
        gradient_tensors = np.where(
            (misorientation == 0).reshape(3, -1, 1),
            0,
            rot_vectors / distances.T[..., None],
        )

    # Reshape the output
    gradient_tensors = gradient_tensors.transpose(1, 0, 2).reshape(out_shape + (3, 3))
    misorientation = misorientation.T.reshape(out_shape + (3,))
    return gradient_tensors, misorientation


def _minimize_l2(
    Lambda: np.ndarray, B: np.ndarray, chunk_size: int = None
) -> np.ndarray:
    """Perform the minimization using the L2 norm.
    Negative densities indicate left handed dislocations, positive densities indicate right handed dislocations.

    Args:
        Lambda: The Nye tensor components. Shape (n_voxels, 9)
        B: The B matrix. Shape (n_slip_systems, 9)
        chunk_size: The size of the chunks to process in parallel. If None, the entire array is processed at once.

    Returns:
        np.ndarray: The dislocation density. Shape (n_slip_systems, n_voxels)"""
    if chunk_size is None:
        out = B.dot(Lambda.T).reshape((-1,))
    else:
        # Split Lambda into chunks
        chunks = np.array_split(Lambda, Lambda.shape[0] // chunk_size)
        out = np.hstack([B.dot(chunk.T).reshape((-1,)) for chunk in chunks])

    dd = B.dot(Lambda.T).reshape((-1,))
    # print("L2 reconstruction error:", np.linalg.norm(B.dot(Lambda.T) - dd.reshape(B.shape[0], -1)))
    return dd


def _minimize_l1_right_only(
    Lambda: np.ndarray, A: np.ndarray, tol: float = 1e-4
) -> np.ndarray:
    """Perform the minimization using the L1 norm with proper equality constraints.
    This function usses a compact basis (same as L2 minimization).
    In the case of L1, this means that only right handed dislocations are allowed
    In L2, negative/positive values are treated as left/right handed dislocations.

    Solves: minimize ||rho||_1 subject to A @ rho = Lambda

    Args:
        Lambda: The Nye tensor components. Shape (n_voxels, 9)
        A: The A matrix. Shape (9, n_slip_systems)
        tol: Tolerance for constraint satisfaction (increased default)

    Returns:
        np.ndarray: The dislocation density. Shape (n_slip_systems, n_voxels)
    """
    Lambda = Lambda.astype(np.float64)
    A = A.astype(np.float64)

    n_constraints = A.shape[0]
    n_slip_systems = A.shape[1]
    N = Lambda.shape[0]

    dd = np.zeros((n_slip_systems, N), dtype=np.float64)
    bad_count = 0

    for i in range(N):
        # Formulation: minimize sum(t_i)
        # subject to:
        #   - A @ rho = Lambda[i]  (equality constraint)
        #   - rho_i <= t_i for all i (inequality)
        #   - -rho_i <= t_i for all i (inequality)
        #   - rho_i >= 0, t_i >= 0 (bounds)

        # Objective: minimize sum of t variables
        c = np.hstack(
            [
                np.zeros(n_slip_systems, dtype=np.float64),  # coefficients for rho
                np.ones(n_slip_systems, dtype=np.float64),  # coefficients for t
            ]
        )

        # Equality constraint: A @ rho = Lambda[i]
        A_eq = np.hstack(
            [
                A,  # (9, n_slip_systems)
                np.zeros(
                    (n_constraints, n_slip_systems), dtype=np.float64
                ),  # zeros for t variables
            ]
        )
        b_eq = Lambda[i].reshape(-1).astype(np.float64)

        # Inequality constraints: rho_i <= t_i and -rho_i <= t_i
        I = np.eye(n_slip_systems, dtype=np.float64)
        A_ub = np.vstack(
            [
                np.hstack([I, -I]),  # rho_i - t_i <= 0  (i.e., rho_i <= t_i)
                np.hstack(
                    [-I, -I]
                ),  # -rho_i - t_i <= 0 (i.e., -rho_i <= t_i, or rho_i >= -t_i)
            ]
        )
        b_ub = np.zeros(2 * n_slip_systems, dtype=np.float64)

        # Bounds: all variables non-negative
        bounds = [(0.0, None)] * (n_slip_systems * 2)

        # Solve
        result = optimize.linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",  # Changed from highs-ipm to highs (more robust)
            options={
                "presolve": True,
                "disp": False,
                "time_limit": 10.0,  # Add timeout per problem
                "dual_feasibility_tolerance": 1e-6,
                "primal_feasibility_tolerance": 1e-6,
            },
        )

        if not result.success:
            dd[:, i] = 0
            bad_count += 1
        else:
            dd[:, i] = result.x[:n_slip_systems]

    if bad_count > 0:
        print(f"L1 minimization: {bad_count} failed optimizations out of {N}")
    # print("L1 reconstruction error:", np.linalg.norm(A @ dd - Lambda.T))
    return dd


def _minimize_l1(Lambda: np.ndarray, A: np.ndarray, tol: float = 1e-4) -> np.ndarray:
    """Perform L1 minimization using expanded basis (separate left/right handed dislocations).

    Following Arsenlis & Parks 1999, the basis is expanded so that left-handed and
    right-handed dislocations each get their own index, all constrained to be non-negative.

    Args:
        Lambda: The Nye tensor components. Shape (n_voxels, 9)
        A: The A matrix for compact basis. Shape (9, n_slip_systems)
        tol: Tolerance for constraint satisfaction

    Returns:
        np.ndarray: The dislocation density in compact basis (allowing negative values).
                    Shape (n_slip_systems, n_voxels)
    """
    Lambda = Lambda.astype(np.float64)
    A = A.astype(np.float64)

    n_constraints = A.shape[0]
    n_slip_systems = A.shape[1]
    N = Lambda.shape[0]

    # Expand the basis: [A, -A] so that each column represents one sign
    # rho_expanded = [rho_plus, rho_minus] where both are ≥ 0
    # Then A @ (rho_plus - rho_minus) = Lambda
    A_expanded = np.hstack([A, -A])  # Shape (9, 2*n_slip_systems)
    n_variables = 2 * n_slip_systems

    dd_compact = np.zeros((n_slip_systems, N), dtype=np.float64)
    bad_count = 0

    for i in range(N):
        # Minimize sum of all densities (L1 norm of expanded representation)
        c = np.ones(n_variables, dtype=np.float64)

        # Equality constraint: A_expanded @ rho_expanded = Lambda[i]
        A_eq = A_expanded
        b_eq = Lambda[i].reshape(-1).astype(np.float64)

        # All variables non-negative (this is the key for L1)
        bounds = [(0.0, None)] * n_variables

        # Solve
        result = optimize.linprog(
            c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options={
                "presolve": True,
                "disp": False,
                "dual_feasibility_tolerance": tol,
                "primal_feasibility_tolerance": tol,
            },
        )

        if not result.success:
            dd_compact[:, i] = 0
            bad_count += 1
        else:
            # Convert back to compact basis: rho = rho_plus - rho_minus
            rho_plus = result.x[:n_slip_systems]
            rho_minus = result.x[n_slip_systems:]
            dd_compact[:, i] = rho_plus - rho_minus

    if bad_count > 0:
        print(f"L1 minimization: {bad_count} failed optimizations out of {N}")
    return dd_compact


def minimize(
    alpha: np.ndarray,
    cs: int,
    A: np.ndarray,
    B: np.ndarray,
    burgers: np.ndarray,
    minimization="l2",
    n_cpus: int = 1,
    chunk_size: int = None,
    progress_bar: bool = False,
) -> np.ndarray:
    """Minimize the dislocation density using the given minimization scheme.
    The equation to be solved is A*rho = Lambda, where A is the A matrix, rho is the dislocation density, and Lambda is the Nye tensor.
    This is solved directly using L2 minimization with the pseudo-inverse of A.
    This can also be solved using L1 minimization, which is done in parallel for each point in the Nye tensor.

    Args:
        alpha: The Nye tensor components. Shape (..., 3, 3)
        cs: The crystal structure of the material. 1 for FCC, 2 for BCC, 3 for HCP.
        A: The A matrix. Shape (9, n_slip_systems)
        B: The B matrix, psuedo-inverse of A. Shape (n_slip_systems, 9)
        burgers: The burgers vector for the material.
                 For HCP with pyramidal slip systems, this should be a tuple of the two burgers vectors.
        minimization: The minimization scheme to use. Either 'l2' or 'l1'.
        n_cpus: The number of CPUs to use for parallel processing during L1 minimization.
                If None, all available CPUs minus one will be used. Not used for L2 minimization.
        chunk_size: The size of the chunks to process in parallel during L1 minimization.
                    Default chunk size is max(1, n_voxels / (4 * n_cpus)).
        progress_bar: Whether to display a progress bar during L1 minimization."""
    # Equation to be solved -> A*rho[array form] = Lambda[Nye in array form]
    # Solve: A*rho = Lambd
    # Nye tensor must be converted into array form Lambda
    # Get shape
    shape = alpha.shape[:-2]
    Lambda = alpha.reshape(-1, 9)
    out_shape = (A.shape[1],) + shape

    # DEBUG
    # Lambda_i = Lambda[0]
    # print(f"Debug: Lambda[0] = {Lambda_i}")
    # print(f"Debug: A shape = {A.shape}, B shape = {B.shape}")
    # print(f"A matrix:\n{A}")
    # print(f"B matrix:\n{B}")
    # print(f"Debug: A @ B (should be close to identity):\n{A @ B}")
    # print(
    #     f"Debug: L1 minimization result for first point:\n{_minimize_l1(Lambda[0:1], A)}"
    # )
    # print(
    #     f"Debug: L2 minimization result for first point:\n{_minimize_l2(Lambda[0:1], B)}"
    # )
    # exit()

    if minimization == "l2":
        print("Performing L2 minimization...")
        dd = _minimize_l2(Lambda, B, chunk_size).reshape(out_shape)

    elif minimization == "l1":
        # Setup chunk size
        if chunk_size is None:
            chunk_size = max(1, Lambda.shape[0] // (n_cpus * 4))

        # Split into chunks
        chunks = np.array_split(Lambda, Lambda.shape[0] // chunk_size)

        # Add progress bar if desired
        if progress_bar:
            chunks = tqdm(chunks, desc="Minimizing (L1) chunks")

        # Process chunks in parallel
        with Parallel(n_jobs=n_cpus, timeout=9999999) as parallel:
            chunk_results = parallel(
                delayed(_minimize_l1)(chunk, A) for chunk in chunks
            )

        # Combine the results
        dd = np.hstack(chunk_results).reshape(out_shape)

    else:
        raise ValueError(
            "Minimization scheme not recognized. Please choose either 'l1' or 'l2'"
        )

    # Divide by Burgers vector correctly based on crystal structure and slip systems
    if cs == 1 or cs == 2:
        dd = dd / burgers
    else:
        if len(burgers) == 2:
            burgers_basal_prismatic = burgers[0]
            burgers_pyramidal = burgers[1]
        elif len(burgers) == 1 and dd.shape[0] <= 9:
            burgers_basal_prismatic = burgers
        elif len(burgers) == 1 and dd.shape[0] == 24:
            burgers_pyramidal = burgers
        else:
            raise ValueError(
                "For HCP, when mixing basal/prismatic and pyramidal slip systems, the Burgers vector must be a tuple of (basa/prismatic, pyramidal) Burgers vectors."
            )

        # Basal slip
        if dd.shape[0] == 6:
            dd = dd / burgers_basal_prismatic

        # Prismatic slip
        elif dd.shape[0] == 3:
            dd = dd / burgers_basal_prismatic

        # Pyramidal slip
        elif dd.shape[0] == 24:
            dd = dd / burgers_pyramidal

        # Basal + Prismatic slip
        elif dd.shape[0] == 9:
            dd = dd / burgers_basal_prismatic

        # Basal + Pyramidal slip
        elif dd.shape[0] == 30:
            dd[:6] = dd[:6] / burgers_basal_prismatic
            dd[6:] = dd[6:] / burgers_pyramidal

        # Prismatic + Pyramidal slip
        elif dd.shape[0] == 27:
            dd[:3] = dd[:3] / burgers_basal_prismatic
            dd[3:] = dd[3:] / burgers_pyramidal

        # All slip systems
        elif dd.shape[0] == 33:
            dd[:9] = dd[:9] / burgers_basal_prismatic
            dd[9:] = dd[9:] / burgers_pyramidal

    return dd


def calculate(
    euler: np.ndarray,
    ids: np.ndarray,
    cs: int,
    slip_systems: str,
    burgers: tuple[float, tuple],
    spacing: tuple,
    minimization: tuple[str, tuple] = "l2",
    n_cpus: int = -1,
    progress_bar: bool = True,
    chunk_size: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the GND density for a 2D or 3D EBSD dataset.

    Args:
        euler: The Euler angles for the dataset. Shape (..., 3)
        ids: The grain IDs for the dataset. Shape (...)
        cs: The crystal structure of the material. 1 for FCC, 2 for BCC, 3 for HCP.
        slip_systems (str, optional): The slip systems to be used. Defaults to 'all'.
                                      (FCC) - unused, always 'all'
                                      (BCC) - 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'screw+110+123', 'screw+112+123', 'all'
                                      (HCP) - 'basal', 'prismatic', 'pyramidal', 'basal+prismatic', 'basal+pyramidal', 'prismatic+pyramidal', 'all'
        burgers: The burgers vector for the material in meters.
                 For HCP with pyramidal slip systems, this should be a tuple of the two burgers vectors.
        spacing: The spacing between voxels in meters. Needs to be a tuple of the same length as the ids.
        minimization: The minimization scheme to use. Either 'l2' or 'l1' or a tuple containing both.
        n_cpus: The number of CPUs to use for parallel processing during L1 minimization.
                If None, all available CPUs minus one will be used. Not used for L2 minimization.
        progress_bar: Whether to display a progress bar during L1 minimization.

    Returns:
        dd: The dislocation density. Shape (n_slip_systems, ...)
        mis: The misorientation. Shape (3, ...)"""

    # Check inputs
    ndim = euler.ndim - 1
    if euler.shape[-1] != 3:
        raise ValueError("The Euler angles must have shape (..., 3)")
    if ids.shape != euler.shape[:-1]:
        raise ValueError("The grain IDs must have the same shape as the Euler angles")
    if cs not in (1, 2, 3):
        raise ValueError(
            "The crystal structure must be 1 for FCC, 2 for BCC, or 3 for HCP"
        )
    if len(spacing) != ndim:
        raise ValueError(
            "The spacing must have the same number of dimensions as the Euler angles"
        )

    euler = euler.astype(PRECISION)

    # Handle minimization
    if isinstance(minimization, tuple):
        if len(minimization) > 2:
            raise ValueError(
                "The minimization scheme must be either 'l1' or 'l2' or both, but cannot have more than two elements"
            )
        minimization = tuple(m.lower() for m in minimization)
    elif isinstance(minimization, str):
        minimization = (minimization.lower(),)
    minimization = sorted(minimization)
    for m in minimization:
        if m not in ("l1", "l2"):
            raise ValueError("The minimization scheme must be either 'l1' or 'l2'")

    # Get the linear operator
    A, B = get_linear_operator(cs, slip_systems)

    # Convert Euler angles to quaternions
    quats = rotations.eu2qu(euler)
    del euler  # Free memory

    # Get the finite difference coordinates
    print("Getting finite difference coordinates...")
    nbrs0, nbrs1, distances = get_finite_difference_coordinates(ids)
    distances *= spacing

    # Get the orientation gradients
    dphi, mis = get_orientation_gradients(
        quats,
        nbrs0,
        nbrs1,
        distances,
        cs,
        n_cpus,
        progress_bar=progress_bar,
        chunk_size=chunk_size,
    )
    del quats, nbrs0, nbrs1, distances  # Free memory
    mis = np.rad2deg(mis)
    mis = mis.transpose(3, 0, 1, 2)  # (..., 3) -> (3, ...)

    # Calculate the alpha tensor
    trace = np.trace(dphi, axis1=3, axis2=4)
    alpha = dphi.transpose(0, 1, 2, 4, 3) - trace[..., None, None] * np.eye(3).reshape(
        1, 1, 1, 3, 3
    )
    del dphi, trace  # Free memory

    # Minimize the dislocation density
    dd = {}
    for m in minimization:
        dd[m] = np.abs(
            minimize(alpha, cs, A, B, burgers, m, n_cpus, progress_bar=progress_bar)
        )

    return dd, mis


def calculate_and_save(
    cs: int,
    burgers: np.ndarray,
    dream3d_path: Path | str | None = None,
    ids_name: str | None = None,
    euler_name: str | None = None,
    ang_path: Path | str | None = None,
    grain_ids_path: Path | str | None = None,
    spacing_units: str = "um",
    minimization: str | Iterable[str] = "l2",
    n_cpus: int = -1,
    slip_systems: str = "all",
    progress_bar: bool = False,
    chunk_size: int = 1000,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Calculate the GND density for a 2D or 3D EBSD dataset and save the results to a DREAM3D file.

    Parameters:
        path (str): The path to the DREAM3D file.
        ids_name (str): The name of the grain ID array in the DREAM3D file.
        euler_name (str): The name of the Euler angles array in the DREAM3D file.
        spacing_units (str): Units of the spacing in the DREAM3D file. Options are
            'nm', 'nanometer', 'nanometers', 'um', 'µm', 'micron', 'microns', 'micrometer',
            'micrometers', 'mm', 'millimeter', 'millimeters', 'm', 'meter', 'meters'.
        cs (int): The crystal structure of the material. 1 for FCC, 2 for BCC, 3 for HCP.
        burgers (np.ndarray): The burgers vector for the material in meters.
                              For HCP with pyramidal slip systems, this should be a tuple of the two burgers vectors.
        minimization (str | Iterable[str]): The minimization scheme to use. Either 'l2' or 'l1' or both.
        n_cpus (int): The number of CPUs to use for parallel processing during L1 minimization.
                      If -1, all available CPUs minus one will be used. Not used for L2 minimization.
        slip_systems (str, optional): The slip systems to be used. Defaults to 'all'.
                                      (FCC) - unused, always 'all'
                                      (BCC) - 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'screw+110+123', 'screw+112+123', 'all'
                                      (HCP) - 'basal', 'prismatic', 'pyramidal', 'basal+prismatic', 'basal+pyramidal', 'prismatic+pyramidal', 'all'
        progress_bar (bool): Whether to display a progress bar during L1 minimization.
        chunk_size (int): The size of the chunks to process in parallel.
    """

    # Handle inputs
    if dream3d_path is not None and (ids_name is None or euler_name is None):
        raise ValueError(
            "If dream3d_path is provided, ids_name and euler_name must also be provided."
        )
    elif dream3d_path is not None and ang_path is not None:
        raise ValueError("Provide either dream3d_path or ang_path, not both.")
    if ang_path is not None and grain_ids_path is None:
        warnings.warn(
            "grain_ids_path not provided. Assuming the entire dataset is a single grain.",
            Warning,
        )

    # Load the data based on the file type
    if dream3d_path is not None:
        path = dream3d_path
        folder = Path(dream3d_path).parent
        euler, ids, spacing = io.read_dream3d(
            dream3d_path, ids_name, euler_name, spacing_units
        )
    elif ang_path is not None:
        folder = Path(ang_path).parent
        euler, ids, spacing = io.read_ang(ang_path, grain_ids_path)

    # Calculate the GND density
    dd, mis = calculate(
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

    print("\nResults summary:")
    print("----------------")
    for m in dd:
        print(f"- {m} GND max: {dd[m].max():.3e} m\u207b\u00b2")
        print(f"- {m} GND min (non-zero): {dd[m][dd[m] > 0].min():.3e} m\u207b\u00b2")
    print(f"- FDM_avg max: {mis.mean(axis=0).max():.3f}\u00b0")
    print(f"- FDM_max max: {mis.max(axis=0).max():.3f}\u00b0")
    print("----------------")

    if dream3d_path is not None:
        # Save the results to the DREAM3D file
        try:
            result = io.save_to_dream3d(path, ids_name, dd, mis)
        except Exception as e:
            print(f"Error saving data to DREAM3D file: {e}")
            result = False

        if not result:
            # Save temporary .npy files in case DREAM3D saving fails
            io.save_npz(dd, mis, folder)
            print(
                "Failed to save results to DREAM3D file. See .npy files for raw data."
            )
            return False
        else:
            print(f"Results saved to DREAM3D file: {Path(path).absolute()}")

    else:
        # Remove z dimension if 2D, order is (C, Z, Y, X), with C either being 3 (mis) or n_slip_systems (dd)
        dd = {k: v[:, 0] for k, v in dd.items()}
        mis = mis[:, 0]
        io.save_npz(dd, mis, folder)
        print("Calculation complete. Check .npy files for raw data.")
        io.generate_images(gnd_data=dd, fdm_data=mis, folder=folder)

    return True
