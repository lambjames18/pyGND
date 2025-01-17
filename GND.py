import os
from typing import Tuple
from math import ceil
import numpy as np
from scipy import optimize
from tqdm import tqdm
import multiprocessing as mp

import rotations
import quaternions

class GND:
    def __init__(self, cs: int, burgers: float, slip_systems: str, minimization="l2"):
        """Class to perform GND calculations on a microstructure.
        Inputs:
            cs: int, crystal structure (1: FCC, 2: BCC, 3: HCP)
            burgers: float, burgers vector magnitude in angstroms
            slip_systems: str, slip systems to use for BCC ('screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'all') and HCP ('basal', 'basal+prismatic', 'all')
            G: float, shear modulus in GPa
            nu: float, Poisson's ratio
        Returns:
            None"""
        self.cs = cs
        if self.cs == 1 or self.cs == 2:
            self.laue_id = 11
        elif self.cs == 3:
            self.laue_id = 9
        self.burgers = burgers * 1e-10
        self.minimization = minimization
        self.set_A_matrix(slip_systems.strip().replace(" ", "").lower())
        self.get_crystallography()
        self.get_symmetry_operators()

    def preflight(self):
        if self.coordinates is None or self.euler_angles is None or self.featIDs is None or self.spacing is None:
            raise ValueError("Data has not been set.")
        if self.coordinates.shape[0] != self.euler_angles.shape[0] or self.coordinates.shape[0] != self.featIDs.size:
            raise ValueError("Data is not the same size.")
        if self.A is None or self.B is None:
            raise ValueError("Crystallography has not been set.")
        if self.symOp is None:
            raise ValueError("Symmetry operators have not been set.")
        # Check 10 random points to see if the data is in the correct format
        for i in range(10):
            idx = np.random.randint(0, self.coordinates.shape[0])
            point_coords = self.coordinates[idx]
            density = self.compute(point_coords)
            print(f"Point: {point_coords} (ID {self.featIDs[idx]}), Density: {density:.2e}")
        print("Preflight checks complete.")

    def set_A_matrix(self, slip_systems):
        # For BCC: 'screw + 110', 'screw + 112', 'screw + 123', 'screw + 110 + 112', 'all'
        if self.cs == 2:
            if slip_systems == 'screw+110':
                self.A_matrix_choice = 1
            elif slip_systems == 'screw+112':
                self.A_matrix_choice = 2
            elif slip_systems == 'screw+123':
                self.A_matrix_choice = 3
            elif slip_systems == 'screw+110+112':
                self.A_matrix_choice = 4
            elif slip_systems == 'all':
                self.A_matrix_choice = 5
            else:
                raise ValueError("Slip systems provided are not valid: {} for BCC".format(slip_systems))
        # HCP: 'basal', 'basal + prismatic', 'all'
        elif self.cs == 3:
            if slip_systems == 'basal':
                self.A_matrix_choice = 1
            elif slip_systems == 'basal+prismatic':
                self.A_matrix_choice = 2
            elif slip_systems == 'all':
                self.A_matrix_choice = 3
            else:
                raise ValueError("Slip systems provided are not valid: {} for HCP".format(slip_systems))
        # For FCC: None (it is ignored)
        else:
            self.A_matrix_choice = None
            return

    def set_data(self, coordinates, euler_angles, feature_ids, spacing):
        """Set the data for the GND calculations.
        Inputs:
            coordinates: np.ndarray, shape (N, 3), coordinates of the points
            euler_angles: np.ndarray, shape (N, 3), Euler angles of the points in radians
            feature_ids: np.ndarray, shape (N,), feature IDs of the points
            spacing: float, spacing of the microstructure"""
        self.coordinates = coordinates
        self.euler_angles = euler_angles.astype(np.float64)
        self.GAO = self.eu2om_multi(self.euler_angles)
        # self.GAO = rotations.eu2om(self.euler_angles)
        # self.GAO  = self.GAO.reshape(3, 3, *self.euler_angles.shape[:-1])
        self.quats = rotations.eu2qu(self.euler_angles)
        self.featIDs = feature_ids
        self.spacing = spacing
    
    def enforce_mask_on_input(self, mask):
        self.GAO[:, :, mask] = 0.0
        self.featIDs[mask] = 0
        self.quats[mask] = 0.0
    
    def unpack_data(self, data):
        self.GND_SR = np.zeros(self.featIDs.size, dtype=float)
        self.GND_SS = np.zeros((self.featIDs.size, self.numSlip), dtype=float)
        self.misori = np.zeros(self.featIDs.size, dtype=float)
        for i, result in enumerate(data):
            self.GND_SR[i] = result[0]
            self.misori[i] = result[1]
            self.GND_SS[i] = result[2]
        self.GND_SR = self.GND_SR.reshape(self.featIDs.shape)
        self.GND_SS = self.GND_SS.reshape(self.featIDs.shape + (self.numSlip,))
        self.misori = self.misori.reshape(self.featIDs.shape)
    
    def run(self):
        for i in range(self.coordinates.shape[0]):
            point_coords = self.coordinates[i]
            self.GND_SR[i], self.misori[i], self.GND_SS[i] = self.compute(point_coords, self.featIDs, self.GAO, self.cs, self.symOp, self.spacing, self.A, self.B, self.burgers)
    
    def compute(self, coords, verbose=False):
        XenvCompleteness = None
        dthe = None
        kappaSR = None
        kappaSRprime = None
        alphaSR = None

        # Get coordinates of current point
        x1 = coords[0].astype(int)
        x2 = coords[1].astype(int)
        x3 = coords[2].astype(int)

        # no calculations if inside void or outside microstructure
        if verbose: print("Position:", (x1, x2, x3), "ID:", self.featIDs[x1, x2, x3], "GAO:", self.GAO[:, :, x1, x2, x3])
        if self.GAO[:, :, x1, x2, x3].sum() != 0 and self.featIDs[x1, x2, x3] != 0:
            # Determine what neighborhood the point has
            XenvCompleteness, YenvCompleteness, ZenvCompleteness = self._determine_neighborhood(self.featIDs, x1, x2, x3)
            # Determine Disorientation between material points and neighbors, influenced by neighborhood
            dthe, diffOperatorX, diffOperatorY, diffOperatorZ = self._determine_dthe(XenvCompleteness, YenvCompleteness, ZenvCompleteness, self.GAO, x1, x2, x3, self.symOp)
            # dthe = dthe[:, ::-1]
            # Calculate average misorientation from dthe
            angles = np.linalg.norm(dthe, axis=0)
            if np.all(angles == 0):
                avg_misori = 0
            else:
                avg_misori = np.mean(np.abs(angles[angles != 0]))
            # kappaSR = determine_kappaV5(dthe, diffOperatorX, diffOperatorY, diffOperatorZ, spacing)
            diffOperators = np.array([diffOperatorX, diffOperatorY, diffOperatorZ])
            kappaSR = self._determine_kappaV5(dthe, diffOperators, self.spacing)
            # Convert Kappa to crystal coordinates since dislocations are described in crystal coordinates
            kappaSRprime = self.GAO[:, :, x1, x2, x3].dot(kappaSR).dot(self.GAO[:, :, x1, x2, x3].T)
            # Calculate Nye Tensor (alpha) from curvature kappa
            alphaSR = kappaSRprime.T - np.trace(kappaSRprime)
            # determine dislocation densities (dd -> rho) from misorientations
            ddSR = self._minimize(alphaSR, self.cs, self.A, self.B, self.burgers)
            # determine total gnd density to be sum of dislocation density across all slip systems
            totalGNDdensitySR = np.abs(ddSR).sum()
            ddSR = np.abs(ddSR).T
            if verbose:
                print("Completeness: ", XenvCompleteness, YenvCompleteness, ZenvCompleteness)
                print("dthe: ", dthe)
                print("avg_misori: ", avg_misori)
                print("Diffoperators: ", diffOperatorX, diffOperatorY, diffOperatorZ)
                print("kappaSR: ", kappaSR)
                print("kappaSRprime: ", kappaSRprime)
                print("alphaSR: ", alphaSR)
                print("ddSR: ", ddSR)
                print("totalGNDdensitySR: {:.2e}".format(totalGNDdensitySR))
        else:
            # tame output for voxels where misorientation can't be calc
            avg_misori = 0
            totalGNDdensitySR = 0
            ddSR_dim = self.A.shape[1]
            ddSR = np.zeros((1, ddSR_dim))

        return (totalGNDdensitySR, avg_misori, ddSR)
    
    def get_symmetry_operators(self):
        # define symmetry operators for cubic or hexagonal symmetries
        if self.cs == 1 or self.cs == 2:
            # there are 24 symmetry operators for cubic symmetries
            symOp = quaternions.laue_elements(11)
            symOp = rotations.qu2om(symOp).astype(float)
            symOp = np.moveaxis(symOp, 0, -1)

        elif self.cs == 3:
            # there are 12 symmetry operators for hexagonal symmetries
            symOp = quaternions.laue_elements(9)
            symOp = rotations.qu2om(symOp).astype(float)
            symOp = np.moveaxis(symOp, 0, -1)

        else:
            print('\nWarning! Crystal structure is not known. No symmetry operators have been defined.\n\n')

        self.symOp = symOp

    def get_crystallography(self):
        # create linear operator B for FCC
        # constants used for linear operator
        a = np.sqrt(3)/9
        c = np.sqrt(3)/84
        d = 1/18
        f = 3/14

        # See Arsenlis & Parks 1999
        self.B = np.array([[  a,   7*c, -13*c,  -7*c,  -a,  13*c,     c,    -c,   0],
                           [ -a,  13*c,  -7*c,    -c,   0,     c,   7*c, -13*c,   a],
                           [  0,     c,    -c, -13*c,   a,   7*c,  13*c,  -7*c,  -a],
                           [  a,  -7*c,  13*c,   7*c,  -a,  13*c,    -c,    -c,   0],
                           [ -a, -13*c,   7*c,     c,   0,     c,  -7*c, -13*c,   a],
                           [  0,    -c,     c,  13*c,   a,   7*c, -13*c,  -7*c,  -a],
                           [  a,  -7*c, -13*c,   7*c,  -a, -13*c,     c,     c,   0],
                           [ -a, -13*c,  -7*c,     c,   0,    -c,   7*c,  13*c,   a],
                           [  0,    -c,    -c,  13*c,   a,  -7*c,  13*c,   7*c,  -a],
                           [  a,   7*c,  13*c,  -7*c,  -a, -13*c,    -c,     c,   0],
                           [ -a,  13*c,   7*c,    -c,   0,    -c,  -7*c,  13*c,  -a],
                           [  0,     c,     c, -13*c,   a,  -7*c, -13*c,   7*c,  -a],
                           [5*d,     f,     0,     f, 5*d,     0,     0,     0,  -d],
                           [5*d,     0,     f,     0,  -d,     0,     f,     0, 5*d],
                           [ -d,     0,     0,     0, 5*d,     f,     0,     f, 5*d],
                           [5*d,    -f,     0,    -f, 5*d,     0,     0,     0,  -d],
                           [5*d,     0,    -f,     0,  -d,     0,    -f,     0, 5*d],
                           [ -d,     0,     0,     0, 5*d,    -f,     0,    -f, 5*d]])
            
        # BCC
        if self.cs == 2:
            A_bcc = self._BCC_A_matrix_generationV2()
            if self.A_matrix_choice == 1:
                a_bcc = np.float64(A_bcc[:, :16])
                numModes = 2
            elif self.A_matrix_choice == 2:
                a_bcc = np.float64([A_bcc[:,:4], A_bcc[:,16:28]])
                numModes = 2
            elif self.A_matrix_choice == 3:
                a_bcc = np.float64([A_bcc[:,:4], A_bcc[:,28:]])
                numModes = 2
            elif self.A_matrix_choice == 4:
                a_bcc = np.float64(A_bcc[:,:28])
                numModes = 3
            elif self.A_matrix_choice == 5:
                numModes = 4
                a_bcc = np.float64(A_bcc)
            self.A = a_bcc
            self.numNye, self.numSlip = self.A.shape
            self.numModes = numModes
            
        # HCP
        elif self.cs == 3:
            d1, d2, d3, d4, d5 = self._HCP_A_matrix_mk3()
            if self.A_matrix_choice == 1:
                A_hcp = np.array([d1, d2])
                numModes = 2
            elif self.A_matrix_choice == 2:
                A_hcp = np.array([d1, d2, d3])
                numModes = 3
            elif self.A_matrix_choice == 3:
                A_hcp = np.array([d1, d2, d3, d4, d5])
                numModes = 5
            
            self.A = A_hcp
            self.numNye, self.numSlip = self.A.shape
            self.numModes = numModes
            
        # FCC
        else:
            # self.A = np.zeros((9,18)) # dummy variable
            BTB = self.B.T @ self.B
            BTB_inv = np.linalg.inv(BTB)
            self.A = BTB_inv @ self.B.T
            #defining number of slip systems and slip modes
            self.numSlip = 18
            self.numModes = 4 
    
    def eu2om_multi(self, eu: np.ndarray):
        thr = 1e-10
        shape = eu.shape
        vol_shape = shape[:-1]
        eu_flat = eu.reshape(-1, 3)
        phi1 = eu_flat[:, 0]
        Phi = eu_flat[:, 1]
        phi2 = eu_flat[:, 2]
        c1, c2, c3 = np.cos(phi1), np.cos(Phi), np.cos(phi2)
        s1, s2, s3 = np.sin(phi1), np.sin(Phi), np.sin(phi2)
        # ZXZ convention, passive rotation
        om = np.array([[ c1*c3-c2*s1*s3, c3*s1+c1*c2*s3, s3*s2],
                    [-c1*s3-c2*c3*s1, c1*c2*c3-s1*s3, c3*s2],
                    [          s2*s1,         -c1*s2,    c2]])

        om = np.where(np.abs(om) < thr, 0.0, om)
        # return np.moveaxis(om, -1, 0)
        return om.reshape((3, 3) + vol_shape)

    def _neighbors(self, x1, x2, x3):
        x1_min, x1_max = max(0, x1-1), min(self.featIDs.shape[0]-1, x1+1) + 1
        x2_min, x2_max = max(0, x2-1), min(self.featIDs.shape[1]-1, x2+1) + 1
        x3_min, x3_max = max(0, x3-1), min(self.featIDs.shape[2]-1, x3+1) + 1
        sub_volume = np.pad(self.featIDs[x1_min:x1_max, x2_min:x2_max, x3_min:x3_max], 1, 'constant')
        temp = np.copy(self.featIDs[x1_min:x1_max, x2_min:x2_max, x3_min:x3_max])
        temp[x1-x1_min, x2-x2_min, x3-x3_min] = -1
        center = np.array(np.where(temp == -1))[:, 0] + 1
        x1_neighbors = sub_volume[center[0]-1:center[0]+2, center[1], center[2]]
        x2_neighbors = sub_volume[center[0], center[1]-1:center[1]+2, center[2]]
        x3_neighbors = sub_volume[center[0], center[1], center[2]-1:center[2]+2]
        return (x1_neighbors, x2_neighbors, x3_neighbors)
    
    def _determine_neighborhood(self, featIDs, x1, x2, x3):
        # Create an empty neighborhood
        Xenv, Yenv, Zenv = "", "", ""
        # Get the local environment of the current voxel (1 connectivity)
        ref_id = featIDs[x1, x2, x3]
        x1_n, x2_n, x3_n = self._neighbors(x1, x2, x3)

        # Quick check for the most common case
        if np.allclose(x1_n, ref_id):
            Xenv = 'central'
        if np.allclose(x2_n, ref_id):
            Yenv = 'central'
        if np.allclose(x3_n, ref_id):
            Zenv = 'central'
        
        # Check which ones need to be completed, x1 then x2, then x3
        if Xenv == "":
            # Check if the next or previous voxel is the same as the reference
            # If neither are the same, set as central
            if x1_n[0] == ref_id:
                Xenv = "backward"
            elif x1_n[-1] == ref_id:
                Xenv = "forward"
            else:
                Xenv = 'constant'
        if Yenv == "":
            if x2_n[0] == ref_id:
                Yenv = "backward"
            elif x2_n[-1] == ref_id:
                Yenv = "forward"
            else:
                Yenv = 'constant'
        if Zenv == "":
            if x3_n[0] == ref_id:
                Zenv = "backward"
            elif x3_n[-1] == ref_id:
                Zenv = "forward"
            else:
                Zenv = 'constant'

        return (Xenv, Yenv, Zenv)
        
    def _deltathetakV5(self, gA, gB, symOp):
        gA = gA.astype(np.float64)
        gB = gB.astype(np.float64)
        if (gA == gB).all():
            return [0.0, 0.0, 0.0]
        else:
            # For Misorientation
            # delg = np.linalg.solve(gA, gB).conj().transpose(0, 2, 1)
            # l = np.around((np.diagonal(delgs).sum() - 1) / 2, 6)
            # deltheta = np.around(np.arccos(l), 6)
            # if deltheta == 0.0: return 0.0
            # misori_matrix[mask, 0] = -(delgs[0, 1, mask] - delgs[1, 0, mask]) * coeff
            # misori_matrix[mask, 1] = -(delgs[2, 0, mask] - delgs[0, 2, mask]) * coeff
            # misori_matrix[mask, 2] = -(delgs[1, 2, mask] - delgs[2, 1, mask]) * coeff

            #### For disorientation ####
            numSym = symOp.shape # 3x3x24
            misori_matrix = np.zeros((numSym[2]**2, 3), dtype=np.float64)
            # Calculate the combined rotations for gA and gB with symmetry operations applied
            gA_temps = np.einsum("ijl,jk->ikl", symOp, gA)  # 3x3x24
            gB_temps = np.einsum("ijl,jk->ikl", symOp, gB)  # 3x3x24
            # Move the axis of the symmetry operations to the first axis, conjugate and transpose
            indices = np.indices((numSym[2], numSym[2])).reshape(2, -1)
            gA_temps = gA_temps[:, :, indices[0]]  # 576x3x3
            gB_temps = gB_temps[:, :, indices[1]]  # 576x3x3
            # Calculate delta g
            delgs = np.einsum("ijl,kjl->ikl", gA_temps, gB_temps)  # 3x3x576
            delthetas = np.real(np.emath.arccos((np.trace(delgs) - 1.0) / 2.0))
            delthetas = np.where(delthetas < 1e-6, 0.0, delthetas)
            # Where deltheta is zero, misorientation matrix is zero
            mask = delthetas != 0
            misori_matrix[~mask] = 0.0
            coeff = (delthetas[mask] / (2 * np.sin(delthetas[mask])))
            # Calculate the misorientation matrix
            misori_matrix[mask, 0] = -(delgs[0, 1, mask] - delgs[1, 0, mask]) * coeff
            misori_matrix[mask, 1] = -(delgs[2, 0, mask] - delgs[0, 2, mask]) * coeff
            misori_matrix[mask, 2] = -(delgs[1, 2, mask] - delgs[2, 1, mask]) * coeff
            
            # Find the minimum misorientation matrix
            d_col = np.argmin(np.abs(misori_matrix), axis=0)
            disori = np.around(np.abs(misori_matrix[d_col].diagonal()), 6)
            return (disori[0], disori[1], disori[2])

    def test(self, coords):
        # Ge the environment of the current point
        XenvCompleteness, YenvCompleteness, ZenvCompleteness = self._determine_neighborhood(self.featIDs, *coords.astype(int))

        # Create the orientation gradient tensor for the current point
        completeness = [XenvCompleteness, YenvCompleteness, ZenvCompleteness]
        gradient_tensor = np.zeros((3, 3), dtype=np.float64)
        misorientation = np.zeros((3,), dtype=np.float64)
        for i, completeness in enumerate(completeness):
            if completeness == "constant":
                misorientation[i] = 0.0
                gradient_tensor[i] = np.zeros((3,), dtype=np.float64)
            else:
                coord0, coord1, scale = self._get_neighbor(completeness, i, coords)
                qA = self.quats[tuple(coord0)]
                qB = self.quats[tuple(coord1)]
                q_dis = quaternions.qu_disorientation(qA, qB, self.laue_id, self.laue_id)
                # print(qA, qB, q_dis)
                rot_vec_dis = quaternions.qu_log(q_dis) * 2
                # print(rot_vec_dis)
                misorientation[i] = np.linalg.norm(rot_vec_dis)
                gradient_tensor[:, i] = rot_vec_dis / (self.spacing[i] * scale)
        # print("Misorientation", misorientation)
        # print("Gradient tensor", gradient_tensor.reshape(-1))

        # Convert the gradient tensor to the crystal coordinates
        # 

        # Calculate the Nye tensor from the curvature tensor
        alpha = gradient_tensor.T - np.trace(gradient_tensor)
        # print("Alpha", alpha.reshape(-1))

        # Determine the dislocation density from the misorientation
        dd = self._minimize(alpha, self.cs, self.A, self.B, self.burgers)
        dd = np.abs(dd)

        return (dd.sum(), misorientation.mean(), dd.T)

    def _deltathetakV5_qu(self, gA, gB, symOp):
        gA = gA.astype(np.float64)
        gB = gB.astype(np.float64)
        if (gA == gB).all():
            return [0.0, 0.0, 0.0]
        else:
            # Convert to quaternions
            quA = rotations.om2qu(gA)
            quB = rotations.om2qu(gB)
            # Calculate the disorientation quaternion
            qu_dis = quaternions.qu_disorientation(quA, quB)



            # For Misorientation
            # delg = np.linalg.solve(gA, gB).conj().transpose(0, 2, 1)
            # l = np.around((np.diagonal(delgs).sum() - 1) / 2, 6)
            # deltheta = np.around(np.arccos(l), 6)
            # if deltheta == 0.0: return 0.0
            # misori_matrix[mask, 0] = -(delgs[0, 1, mask] - delgs[1, 0, mask]) * coeff
            # misori_matrix[mask, 1] = -(delgs[2, 0, mask] - delgs[0, 2, mask]) * coeff
            # misori_matrix[mask, 2] = -(delgs[1, 2, mask] - delgs[2, 1, mask]) * coeff

            #### For disorientation ####
            numSym = symOp.shape # 3x3x24
            misori_matrix = np.zeros((numSym[2]**2, 3), dtype=np.float64)
            # Calculate the combined rotations for gA and gB with symmetry operations applied
            gA_temps = np.einsum("ijl,jk->ikl", symOp, gA)  # 3x3x24
            gB_temps = np.einsum("ijl,jk->ikl", symOp, gB)  # 3x3x24
            # Move the axis of the symmetry operations to the first axis, conjugate and transpose
            indices = np.indices((numSym[2], numSym[2])).reshape(2, -1)
            gA_temps = gA_temps[:, :, indices[0]]  # 576x3x3
            gB_temps = gB_temps[:, :, indices[1]]  # 576x3x3
            # Calculate delta g
            delgs = np.einsum("ijl,kjl->ikl", gA_temps, gB_temps)  # 3x3x576
            delthetas = np.real(np.emath.arccos((np.trace(delgs) - 1.0) / 2.0))
            delthetas = np.where(delthetas < 1e-6, 0.0, delthetas)
            # Where deltheta is zero, misorientation matrix is zero
            mask = delthetas != 0
            misori_matrix[~mask] = 0.0
            coeff = (delthetas[mask] / (2 * np.sin(delthetas[mask])))
            # Calculate the misorientation matrix
            misori_matrix[mask, 0] = -(delgs[0, 1, mask] - delgs[1, 0, mask]) * coeff
            misori_matrix[mask, 1] = -(delgs[2, 0, mask] - delgs[0, 2, mask]) * coeff
            misori_matrix[mask, 2] = -(delgs[1, 2, mask] - delgs[2, 1, mask]) * coeff
            
            # Find the minimum misorientation matrix
            d_col = np.argmin(np.abs(misori_matrix), axis=0)
            disori = np.around(np.abs(misori_matrix[d_col].diagonal()), 6)
            return (disori[0], disori[1], disori[2])

    def _get_neighbor(self, completeness, axis, coord):
        d1 = np.array([0, 0, 0])
        d2 = np.array([0, 0, 0])
        if completeness == 'backward':
            d1[axis] = -1
            d2[axis] = 0
            scale = 1
        elif completeness == 'forward':
            d1[axis] = 0
            d2[axis] = 1
            scale = 1
        elif completeness == 'central':
            d1[axis] = -1
            d2[axis] = 1
            scale = 2
        else:
            scale = 1
        return (coord + d1, coord + d2, scale)

    def _determine_dthe(self, XenvCompleteness, YenvCompleteness, ZenvCompleteness, GAO, x1, x2, x3, symOp):
        environment_case = {'forward': self._deltathetakV5, 'backward': self._deltathetakV5, 'central': self._deltathetakV5, 'constant': lambda *args: [0.0, 0.0, 0.0]}

        dthe = np.zeros((3, 3), dtype=np.float64)
        gx1, gx2, gy1, gy2, gz1, gz2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        diffOperatorX, diffOperatorY, diffOperatorZ = 1, 1, 1

        # switch statement evaluating expression for x environment
        if XenvCompleteness == 'backward':
            gx1, gx2 = GAO[:, :, x1-1, x2, x3], GAO[:, :, x1, x2, x3]
        elif XenvCompleteness == 'forward':
            gx1, gx2 = GAO[:, :, x1, x2, x3], GAO[:, :, x1+1, x2, x3]
        elif XenvCompleteness == 'central':
            gx1, gx2 = GAO[:, :, x1-1, x2, x3], GAO[:, :, x1+1, x2, x3]
            diffOperatorX = 2

        if YenvCompleteness == 'backward':
            gy1, gy2 = GAO[:, :, x1, x2-1, x3], GAO[:, :, x1, x2, x3]
        elif YenvCompleteness == 'forward':
            gy1, gy2 = GAO[:, :, x1, x2, x3], GAO[:, :, x1, x2+1, x3]
        elif YenvCompleteness == 'central':
            gy1, gy2 = GAO[:, :, x1, x2-1, x3], GAO[:, :, x1, x2+1, x3]
            diffOperatorY = 2

        if ZenvCompleteness == 'backward':
            gz1, gz2 = GAO[:, :, x1, x2, x3-1], GAO[:, :, x1, x2, x3]
        elif ZenvCompleteness == 'forward':
            gz1, gz2 = GAO[:, :, x1, x2, x3], GAO[:, :, x1, x2, x3+1]
        elif ZenvCompleteness == 'central':
            gz1, gz2 = GAO[:, :, x1, x2, x3-1], GAO[:, :, x1, x2, x3+1]
            diffOperatorZ = 2
        
        dthe[:, 0] = environment_case[XenvCompleteness](gx1, gx2, symOp)
        dthe[:, 1] = environment_case[YenvCompleteness](gy1, gy2, symOp)
        dthe[:, 2] = environment_case[ZenvCompleteness](gz1, gz2, symOp)
        return dthe, diffOperatorX, diffOperatorY, diffOperatorZ

    def _determine_kappaV5(self, dthe, diffOperators, spacing):
        # kappa must be calculated for material point
        kappa = dthe / (diffOperators * spacing)
        return kappa

    def _minimize(self, alpha, cs, A, B, burgers):
        # Equation to be solved -> A*rho[array form] = Lambda[Nye in array form] 
        # Solve: A*rho = Lambd
        # Nye tensor must be converted into array form Lambda
        Lambda = alpha.reshape(-1, 1)  # Shape (9x1)
        if self.minimization == 'l2':
            if cs == 2 or cs == 3:
                # two steps to solve via minimize‖Ax−b‖2
                B = A.T.dot(np.linalg.inv(A.dot(A.T)))
                dd = B.dot(Lambda)
                if self.numSlip > 9 & cs == 3:
                    burgers_ca = 4.68                                          ##### WHAT IS  THIS
                    burgers_ca = burgers_ca*1E-10
                    dd[:9] = dd[:9] / burgers
                    dd[9:33] = dd[9:33] / burgers_ca
                else:
                    dd = dd/burgers
            else:
                dd = B.dot(Lambda)/burgers  # same as matmul
        elif self.minimization == 'l1':
            n_constraints, n_slip_systems = A.shape
            c = np.hstack((np.zeros(n_slip_systems), np.ones(n_slip_systems)))
            A_eq = np.hstack([A, np.zeros((n_constraints, n_slip_systems))])
            b_eq = Lambda.reshape(-1)
            I = np.eye(n_slip_systems)
            A_ub = np.vstack([np.hstack([I, -I]), np.hstack([-I, -I])])
            b_ub = np.zeros(2*n_slip_systems)
            bounds = [(0, np.inf)]*n_slip_systems*2
            bounds = np.array(bounds)
            result = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            dd = result.x[:n_slip_systems] / burgers
        else:
            raise ValueError("Minimization scheme not recognized. Please choose either 'l1' or 'l2'")
        return dd

    def _BCC_A_matrix_generationV2(self):
        # BCC A matrix formulation
        # b vectors for systems {110}{112}{321} 1->4 as screw
        # b vectors for systems {110} 5->16 as edge
        # b vectors for systems {112} 17->28 as edge
        # b vectors for systems {123} 29->52 as edge
        bedge = np.float32((1/np.sqrt(3))* np.array([[ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [ 1,  1,  1], # {110}<111> SLIP
                                                    [ 1,  1,  1], [-1, -1,  1], [-1, -1,  1], [-1, -1,  1], [-1,  1,  1], [-1,  1,  1], [-1,  1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [ 1,  1,  1], # {112}<111> SLIP
                                                    [ 1,  1,  1], [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1], [ 1,  1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1, -1], [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1, -1,  1], [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1], [ 1,  1,  1]])) # {123}<111> SLIP

        nedge = np.zeros((48,3))
        # {110}<111> SLIP
        nedge[:12] = np.float32(1/np.sqrt(2) * np.array([[0,  1,  1], [1,  0,  1], [1, -1,  0], [0,  1, -1], [1,  0,  1], [1,  1,  0], [0,  1,  1], [1,  0, -1], [1,  1,  0], [0,  1, -1], [1,  0, -1], [1, -1,  0]]))
        # {112}<111> SLIP
        nedge[12:24] = np.float32(1/np.sqrt(6) * np.array([[-2,  1, -1], [ 1, -2, -1], [ 1,  1,  2], [-2, -1, -1], [ 1,  2, -1], [ 1, -1,  2], [ 2,  1, -1], [-1, -2, -1], [-1,  1,  2], [ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]))
        # {123}<111> SLIP
        nedge[24:] = np.float32(1/np.sqrt(14) * np.array([[ 1,  2,  3], [-1,  3,  2], [ 2,  1,  3], [-2,  3,  1], [ 3, -1,  2], [ 3, -2,  1], [-1,  2, -3], [ 1,  3, -2], [ 2, -1,  3], [ 2,  3, -1], [ 3,  1,  2], [ 3,  2,  1], [ 1, -2, -3], [ 1,  3,  2], [ 2, -1, -3], [ 2,  3,  1], [ 3,  1, -2], [ 3,  2, -1], [ 1,  2, -3], [ 1, -3,  2], [ 2,  1, -3], [ 2, -3,  1], [-3,  1,  2], [-3,  2,  1]]))

        b = np.around(1/np.sqrt(3) * np.array([[1,  1, -1], [1, -1, -1], [1, -1,  1], [1,  1,  1]]), 4)
        tscrew = np.copy(b)
        t = np.zeros((48,3))

        # prepping dislocation dyads matrix
        d = np.zeros((9,52))

        # Calc Screw Dislocation Density
        for index in range(4):
            d[0,index] = np.around(b[index, 0] * tscrew[index, 0], 8)
            d[1,index] = np.around(b[index, 0] * tscrew[index, 1], 8)
            d[2,index] = np.around(b[index, 0] * tscrew[index, 2], 8)
            d[3,index] = np.around(b[index, 1] * tscrew[index, 0], 8)
            d[4,index] = np.around(b[index, 1] * tscrew[index, 1], 8)
            d[5,index] = np.around(b[index, 1] * tscrew[index, 2], 8)
            d[6,index] = np.around(b[index, 2] * tscrew[index, 0], 8)
            d[7,index] = np.around(b[index, 2] * tscrew[index, 1], 8)
            d[8,index] = np.around(b[index, 2] * tscrew[index, 2], 8)

        # Calc Edge Dislocation Density
        t[:12] = np.array([[-0.8165,  0.4082, -0.4082], [-0.4082,  0.8165,  0.4082], [ 0.4082,  0.4082,  0.8165], [-0.8165, -0.4082, -0.4082], [ 0.4082,  0.8165, -0.4082], [-0.4082,  0.4082, -0.8165], [ 0.8165,  0.4082, -0.4082], [-0.4082, -0.8165, -0.4082], [ 0.4082, -0.4082, -0.8165], [ 0.8165, -0.4082, -0.4082], [ 0.4082, -0.8165,  0.4082], [-0.4082, -0.4082,  0.8165]])

        for index1 in range(12, t.shape[0]):
            t[index1] = np.float32(np.cross(nedge[index1],bedge[index1]))

        # Calculate Edge dislocation density
        for index in range(4, d.shape[1]):
            d[0, index] = bedge[index-4, 0] * t[index-4, 0]
            d[1, index] = bedge[index-4, 0] * t[index-4, 1]
            d[2, index] = bedge[index-4, 0] * t[index-4, 2]
            d[3, index] = bedge[index-4, 1] * t[index-4, 0]
            d[4, index] = bedge[index-4, 1] * t[index-4, 1]
            d[5, index] = bedge[index-4, 1] * t[index-4, 2]
            d[6, index] = bedge[index-4, 2] * t[index-4, 0]
            d[7, index] = bedge[index-4, 2] * t[index-4, 1]
            d[8, index] = bedge[index-4, 2] * t[index-4, 2]
        
        screws = d[:, :4]
        edges = d[:, 4:]
        A_bcc = np.hstack((screws.astype(float), edges.astype(float)))
        return A_bcc

    def _HCP_A_matrix_mk3(self):
        # Relevant Direcitons in [uvtw] notation

        # Basal Slip
        bMBbasal = np.array([[1, 1, -2, 0], [1, -2, 1, 0], [-2, 1, 1, 0]])
        nMBbasal = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])

        # Prismatic Slip
        bMBprismatic = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [1, 1, -2, 0]])
        nMBprismatic = np.array([[0, 1, -1, 0], [1, 0, -1, 0], [1, -1, 0, 0]])

        # Pyramidal <c+a> Slip
        bMBpyramidalCplusA = np.array([[-1, -1,  2, 3], [-2,  1,  1, 3], [ 1,  1, -2, 3], [-1,  2, -1, 3], [ 2, -1, -1, 3], [ 1, -2,  1, 3], [ 2, -1, -1, 3], [ 1,  1, -2, 3], [-1, -1,  2, 3], [ 1, -2,  1, 3], [-2,  1,  1, 3], [-1,  2, -1, 3]])
                        
        nMBpyramidalCplusA = np.array([[ 1,  0, -1, 1], [ 1,  0, -1, 1], [ 0, -1,  1, 1], [ 0, -1,  1, 1], [-1,  1,  0, 1], [-1,  1,  0, 1], [-1,  0,  1, 1], [-1,  0,  1, 1], [ 0,  1, -1, 1], [ 0,  1, -1, 1], [ 1, -1,  0, 1], [ 1, -1,  0, 1]]) 
                        
        # Conversion to [UVW] Miller Indices
        bMillerBASAL = np.zeros((3,3))
        nMillerBASAL = np.zeros((3,3))
        tMillerBASAL = np.zeros((3,3))

        for index in range(3):
            u = bMBbasal[index, 0]
            v = bMBbasal[index, 1]
            t = bMBbasal[index, 2]
            w = bMBbasal[index, 3]

            bMillerBASAL[index, 0] = 1/3 * (u - t)
            bMillerBASAL[index, 1] = 1/3 * (v - t)
            bMillerBASAL[index, 2] = 1/3 * w
            
            u = nMBbasal[index, 0]
            v = nMBbasal[index, 1]
            t = nMBbasal[index, 2]
            w = nMBbasal[index, 3]

            nMillerBASAL[index, 0] = u - t
            nMillerBASAL[index, 1] = v - t
            nMillerBASAL[index, 2] = w

        bMillerPRISMATIC = np.zeros((3,3))
        nMillerPRISMATIC = np.zeros((3,3))
        tMillerPRISMATIC = np.zeros((3,3))

        for index in range(3):
            u = bMBprismatic[index, 0]
            v = bMBprismatic[index, 1]
            t = bMBprismatic[index, 2]
            w = bMBprismatic[index, 3]

            bMillerPRISMATIC[index, 0] = 1/3 * (u - t)
            bMillerPRISMATIC[index, 1] = 1/3 * (v - t)
            bMillerPRISMATIC[index, 2] = 1/3 * w
            
            u = nMBprismatic[index, 0]
            v = nMBprismatic[index, 1]
            t = nMBprismatic[index, 2]
            w = nMBprismatic[index, 3]

            nMillerPRISMATIC[index, 0] = u - t
            nMillerPRISMATIC[index, 1] = v - t
            nMillerPRISMATIC[index, 2] = w

        bMillerPYRAMIDALcplusa = np.zeros((12,3))
        nMillerPYRAMIDALcplusa = np.zeros((12,3))
        tMillerPYRAMIDALcplusa = np.zeros((12,3))

        for index in range(12):
            u = bMBpyramidalCplusA[index, 0]
            v = bMBpyramidalCplusA[index, 1]
            t = bMBpyramidalCplusA[index, 2]
            w = bMBpyramidalCplusA[index, 3]

            bMillerPYRAMIDALcplusa[index, 0] = 1/3 * (u - t)
            bMillerPYRAMIDALcplusa[index, 1] = 1/3 * (v - t)
            bMillerPYRAMIDALcplusa[index, 2] = 1/3 * w
            
            u = nMBpyramidalCplusA[index, 0]
            v = nMBpyramidalCplusA[index, 1]
            t = nMBpyramidalCplusA[index, 2]
            w = nMBpyramidalCplusA[index, 3]

            nMillerPYRAMIDALcplusa[index, 0] = u - t
            nMillerPYRAMIDALcplusa[index, 1] = v - t
            nMillerPYRAMIDALcplusa[index, 2] = w

        # determining tangent vectors for edge dislocations
        for index3 in range(3):
            tMillerBASAL[index3] = np.cross(nMillerBASAL[index3], bMillerBASAL[index3])
        for index1 in range(3):
            tMillerPRISMATIC[index1] = np.cross(nMillerPRISMATIC[index1], bMillerPRISMATIC[index1])
        for index2 in range(12):
            tMillerPYRAMIDALcplusa[index2] = np.cross(nMillerPYRAMIDALcplusa[index2], bMillerPYRAMIDALcplusa[index2])

        # --- normalize w/r to Miller coordinates
        bMillerBASAL[0] = 1/np.sqrt(2) * bMillerBASAL[0]
        tMillerBASAL[0] = 1/np.sqrt(2) * tMillerBASAL[0]

        bMillerPRISMATIC[2] = 1/np.sqrt(2) * bMillerPRISMATIC[2]
        tMillerPRISMATIC = 1/2 *tMillerPRISMATIC
        nMillerPRISMATIC[:2] = 1/np.sqrt(5) * nMillerPRISMATIC[:2]
        nMillerPRISMATIC[2] = 1/np.sqrt(2) * nMillerPRISMATIC[2]

        bMillerPYRAMIDALcplusa[:2] = 1/np.sqrt(2) * bMillerPYRAMIDALcplusa[:2]
        bMillerPYRAMIDALcplusa[2] = 1/np.sqrt(3) * bMillerPYRAMIDALcplusa[2]
        bMillerPYRAMIDALcplusa[3:7] = 1/np.sqrt(2) * bMillerPYRAMIDALcplusa[3:7]
        bMillerPYRAMIDALcplusa[7:9] = 1/np.sqrt(3) * bMillerPYRAMIDALcplusa[7:9]
        bMillerPYRAMIDALcplusa[9:] = 1/np.sqrt(2) * bMillerPYRAMIDALcplusa[9:]

        tMillerPYRAMIDALcplusa[0] = 1/np.sqrt(14) * tMillerPYRAMIDALcplusa[0]
        tMillerPYRAMIDALcplusa[1] = 1/np.sqrt(11) * tMillerPYRAMIDALcplusa[1]
        tMillerPYRAMIDALcplusa[2] = 1/np.sqrt(14) * tMillerPYRAMIDALcplusa[2]
        tMillerPYRAMIDALcplusa[3] = 1/np.sqrt(11) * tMillerPYRAMIDALcplusa[3]
        tMillerPYRAMIDALcplusa[4:6] = 1/np.sqrt(6) * tMillerPYRAMIDALcplusa[4:6]
        tMillerPYRAMIDALcplusa[6] = 1/np.sqrt(11) * tMillerPYRAMIDALcplusa[6]
        tMillerPYRAMIDALcplusa[7:9] = 1/np.sqrt(14) * tMillerPYRAMIDALcplusa[7:9]
        tMillerPYRAMIDALcplusa[9] = 1/np.sqrt(11) * tMillerPYRAMIDALcplusa[9]
        tMillerPYRAMIDALcplusa[10:] = 1/np.sqrt(6) * tMillerPYRAMIDALcplusa[10:]

        nMillerPYRAMIDALcplusa[:4] = 1/np.sqrt(6) * nMillerPYRAMIDALcplusa[:4]
        nMillerPYRAMIDALcplusa[4:6] = 1/np.sqrt(3) * nMillerPYRAMIDALcplusa[4:6]
        nMillerPYRAMIDALcplusa[6:10] = 1/np.sqrt(6) * nMillerPYRAMIDALcplusa[6:10]
        nMillerPYRAMIDALcplusa[10:] = 1/np.sqrt(3) * nMillerPYRAMIDALcplusa[10:]

        # determining tangent vectors for screw dislocations
        tMillerBASALscrew = bMillerBASAL
        tMillerPYRAMIDALcplusascrew = bMillerPYRAMIDALcplusa

        # prepping dislocation dyads matrix
        d1 = np.zeros((9,3))
        d2 = np.zeros((9,3))
        d3 = np.zeros((9,3))
        d4 = np.zeros((9,12))
        d5 = np.zeros((9,12))

        for index in range(3):
            d1[0, index] = bMillerBASAL[index, 0] * tMillerBASALscrew[index, 0]
            d1[1, index] = bMillerBASAL[index, 0] * tMillerBASALscrew[index, 1]
            d1[2, index] = bMillerBASAL[index, 0] * tMillerBASALscrew[index, 2]
            d1[3, index] = bMillerBASAL[index, 1] * tMillerBASALscrew[index, 0]
            d1[4, index] = bMillerBASAL[index, 1] * tMillerBASALscrew[index, 1]
            d1[5, index] = bMillerBASAL[index, 1] * tMillerBASALscrew[index, 2]
            d1[6, index] = bMillerBASAL[index, 2] * tMillerBASALscrew[index, 0]
            d1[7, index] = bMillerBASAL[index, 2] * tMillerBASALscrew[index, 1]
            d1[8, index] = bMillerBASAL[index, 2] * tMillerBASALscrew[index, 2]
            
            d2[0, index] = bMillerBASAL[index, 0] * tMillerBASAL[index, 0]
            d2[1, index] = bMillerBASAL[index, 0] * tMillerBASAL[index, 1]
            d2[2, index] = bMillerBASAL[index, 0] * tMillerBASAL[index, 2]
            d2[3, index] = bMillerBASAL[index, 1] * tMillerBASAL[index, 0]
            d2[4, index] = bMillerBASAL[index, 1] * tMillerBASAL[index, 1]
            d2[5, index] = bMillerBASAL[index, 1] * tMillerBASAL[index, 2]
            d2[6, index] = bMillerBASAL[index, 2] * tMillerBASAL[index, 0]
            d2[7, index] = bMillerBASAL[index, 2] * tMillerBASAL[index, 1]
            d2[8, index] = bMillerBASAL[index, 2] * tMillerBASAL[index, 2]
            
            d3[0, index] = bMillerPRISMATIC[index, 0] * tMillerPRISMATIC[index, 0]
            d3[1, index] = bMillerPRISMATIC[index, 0] * tMillerPRISMATIC[index, 1]
            d3[2, index] = bMillerPRISMATIC[index, 0] * tMillerPRISMATIC[index, 2]
            d3[3, index] = bMillerPRISMATIC[index, 1] * tMillerPRISMATIC[index, 0]
            d3[4, index] = bMillerPRISMATIC[index, 1] * tMillerPRISMATIC[index, 1]
            d3[5, index] = bMillerPRISMATIC[index, 1] * tMillerPRISMATIC[index, 2]
            d3[6, index] = bMillerPRISMATIC[index, 2] * tMillerPRISMATIC[index, 0]
            d3[7, index] = bMillerPRISMATIC[index, 2] * tMillerPRISMATIC[index, 1]
            d3[8, index] = bMillerPRISMATIC[index, 2] * tMillerPRISMATIC[index, 2]

        for index in range(12):
            d4[0, index] = bMillerPYRAMIDALcplusa[index,0] * tMillerPYRAMIDALcplusa[index,0]
            d4[1, index] = bMillerPYRAMIDALcplusa[index,0] * tMillerPYRAMIDALcplusa[index,1]
            d4[2, index] = bMillerPYRAMIDALcplusa[index,0] * tMillerPYRAMIDALcplusa[index,2]
            d4[3, index] = bMillerPYRAMIDALcplusa[index,1] * tMillerPYRAMIDALcplusa[index,0]
            d4[4, index] = bMillerPYRAMIDALcplusa[index,1] * tMillerPYRAMIDALcplusa[index,1]
            d4[5, index] = bMillerPYRAMIDALcplusa[index,1] * tMillerPYRAMIDALcplusa[index,2]
            d4[6, index] = bMillerPYRAMIDALcplusa[index,2] * tMillerPYRAMIDALcplusa[index,0]
            d4[7, index] = bMillerPYRAMIDALcplusa[index,2] * tMillerPYRAMIDALcplusa[index,1]
            d4[8, index] = bMillerPYRAMIDALcplusa[index,2] * tMillerPYRAMIDALcplusa[index,2]
            
            d5[0,index] = bMillerPYRAMIDALcplusa[index,0] * tMillerPYRAMIDALcplusascrew[index, 0]
            d5[1,index] = bMillerPYRAMIDALcplusa[index,0] * tMillerPYRAMIDALcplusascrew[index, 1]
            d5[2,index] = bMillerPYRAMIDALcplusa[index,0] * tMillerPYRAMIDALcplusascrew[index, 2]
            d5[3,index] = bMillerPYRAMIDALcplusa[index,1] * tMillerPYRAMIDALcplusascrew[index, 0]
            d5[4,index] = bMillerPYRAMIDALcplusa[index,1] * tMillerPYRAMIDALcplusascrew[index, 1]
            d5[5,index] = bMillerPYRAMIDALcplusa[index,1] * tMillerPYRAMIDALcplusascrew[index, 2]
            d5[6,index] = bMillerPYRAMIDALcplusa[index,2] * tMillerPYRAMIDALcplusascrew[index, 0]
            d5[7,index] = bMillerPYRAMIDALcplusa[index,2] * tMillerPYRAMIDALcplusascrew[index, 1]
            d5[8,index] = bMillerPYRAMIDALcplusa[index,2] * tMillerPYRAMIDALcplusascrew[index, 2]

        # -- from here, dislocation dyads are established and user will define slip
        # systems to contribute to A matrix
        return (d1, d2, d3, d4, d5)



def get_linear_operator(cs:int, slip_systems:str="all") -> Tuple[np.ndarray, np.ndarray]:
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
    if slip_systems not in ['all', 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'basal', 'basal+prismatic']:
        raise ValueError("Slip systems must be 'all', 'screw+110', 'screw+112', 'screw+123', 'screw+110+112', 'basal', 'basal+prismatic', depending on the crystam structure.")

    # Create the A matrix for the given crystal structure
    if cs == 1:
        a = np.sqrt(3)/9
        c = np.sqrt(3)/84
        d = 1/18
        f = 3/14

        # See Arsenlis & Parks 1999
        B = np.array([[  a,   7*c, -13*c,  -7*c,  -a,  13*c,     c,    -c,   0],
                      [ -a,  13*c,  -7*c,    -c,   0,     c,   7*c, -13*c,   a],
                      [  0,     c,    -c, -13*c,   a,   7*c,  13*c,  -7*c,  -a],
                      [  a,  -7*c,  13*c,   7*c,  -a,  13*c,    -c,    -c,   0],
                      [ -a, -13*c,   7*c,     c,   0,     c,  -7*c, -13*c,   a],
                      [  0,    -c,     c,  13*c,   a,   7*c, -13*c,  -7*c,  -a],
                      [  a,  -7*c, -13*c,   7*c,  -a, -13*c,     c,     c,   0],
                      [ -a, -13*c,  -7*c,     c,   0,    -c,   7*c,  13*c,   a],
                      [  0,    -c,    -c,  13*c,   a,  -7*c,  13*c,   7*c,  -a],
                      [  a,   7*c,  13*c,  -7*c,  -a, -13*c,    -c,     c,   0],
                      [ -a,  13*c,   7*c,    -c,   0,    -c,  -7*c,  13*c,  -a],
                      [  0,     c,     c, -13*c,   a,  -7*c, -13*c,   7*c,  -a],
                      [5*d,     f,     0,     f, 5*d,     0,     0,     0,  -d],
                      [5*d,     0,     f,     0,  -d,     0,     f,     0, 5*d],
                      [ -d,     0,     0,     0, 5*d,     f,     0,     f, 5*d],
                      [5*d,    -f,     0,    -f, 5*d,     0,     0,     0,  -d],
                      [5*d,     0,    -f,     0,  -d,     0,    -f,     0, 5*d],
                      [ -d,     0,     0,     0, 5*d,    -f,     0,    -f, 5*d]])

        # FCC
        A = pseudo_inverse(B)

    elif cs == 2:
        # BCC
        A = generate_BCC_A_matrix()
        if slip_systems == 'screw+110':
            A = A[:,:16]
        elif slip_systems == 'screw+112':
            A = np.hstack((A[:,:4], A[:,16:28]))
        elif slip_systems == 'screw+123':
            A = np.hstack((A[:,:4], A[:,28:]))
        elif slip_systems == 'screw+110+112':
            A = A[:,:28]
        elif slip_systems == 'screw+110+123':
            A = np.hstack((A[:,:16], A[:,28:]))
        elif slip_systems == 'screw+112+123':
            A = np.hstack((A[:,:4], A[:,16:]))
        elif slip_systems == 'all':
            pass
        B = pseudo_inverse(A)

    elif cs == 3:
        # HCP
        A = generate_HCP_A_matrix()
        if slip_systems == 'basal':
            A = A[:,:6]  # 3 edge basal and 3 screw basal slip systems
        elif slip_systems == 'prismatic':
            A = A[:,6:9]  # 3 edge prismatic slip systems
        elif slip_systems == 'pyramidal':
            A = A[:,9:]  # 12 edge pyramidal and 12 screw pyramidal slip systems
        elif slip_systems == 'basal+prismatic':
            A = A[:,:9]
        elif slip_systems == 'basal+pyramidal':
            A = np.hstack((A[:,:6], A[:,9:]))
        elif slip_systems == 'prismatic+pyramidal':
            A = A[:,6:]
        elif slip_systems == 'all':
            pass
        B = pseudo_inverse(A)

    return (A, B)


def generate_BCC_A_matrix() -> np.ndarray:
    """Generate the A matrix for BCC crystal structure."""
    # Burgers vectors and slip plane normals for BCC
    b_n = np.array([
        [[ 1,  1, -1], [ 1,  1, -1]],  # <111> screw
        [[ 1, -1, -1], [ 1, -1, -1]],
        [[ 1, -1,  1], [ 1, -1,  1]],
        [[ 1,  1,  1], [ 1,  1,  1]],
        [[ 1,  1, -1], [ 0,  1,  1]],  # {110}<111> edge
        [[ 1,  1, -1], [ 1,  0,  1]],
        [[ 1,  1, -1], [ 1, -1,  0]],
        [[ 1, -1, -1], [ 0,  1, -1]],
        [[ 1, -1, -1], [ 1,  0,  1]],
        [[ 1, -1, -1], [ 1,  1,  0]],
        [[ 1, -1,  1], [ 0,  1,  1]],
        [[ 1, -1,  1], [ 1,  0, -1]],
        [[ 1, -1,  1], [ 1,  1,  0]],
        [[ 1,  1,  1], [ 0,  1, -1]],
        [[ 1,  1,  1], [ 1,  0, -1]],
        [[ 1,  1,  1], [ 1, -1,  0]],
        [[-1, -1,  1], [-2,  1, -1]],  # {112}<111> edge
        [[-1, -1,  1], [ 1, -2, -1]],
        [[-1, -1,  1], [ 1,  1,  2]],
        [[-1,  1,  1], [-2, -1, -1]],
        [[-1,  1,  1], [ 1,  2, -1]],
        [[-1,  1,  1], [ 1, -1,  2]],
        [[ 1, -1,  1], [ 2,  1, -1]],
        [[ 1, -1,  1], [-1, -2, -1]],
        [[ 1, -1,  1], [-1,  1,  2]],
        [[ 1,  1,  1], [ 2, -1, -1]],
        [[ 1,  1,  1], [-1,  2, -1]],
        [[ 1,  1,  1], [-1, -1,  2]],
        [[ 1,  1, -1], [ 1,  2,  3]],  # {123}<111> edge
        [[ 1,  1, -1], [-1,  3,  2]],
        [[ 1,  1, -1], [ 2,  1,  3]],
        [[ 1,  1, -1], [-2,  3,  1]],
        [[ 1,  1, -1], [ 3, -1,  2]],
        [[ 1,  1, -1], [ 3, -2,  1]],
        [[ 1, -1, -1], [-1,  2, -3]],
        [[ 1, -1, -1], [ 1,  3, -2]],
        [[ 1, -1, -1], [ 2, -1,  3]],
        [[ 1, -1, -1], [ 2,  3, -1]],
        [[ 1, -1, -1], [ 3,  1,  2]],
        [[ 1, -1, -1], [ 3,  2,  1]],
        [[ 1, -1,  1], [ 1, -2, -3]],
        [[ 1, -1,  1], [ 1,  3,  2]],
        [[ 1, -1,  1], [ 2, -1, -3]],
        [[ 1, -1,  1], [ 2,  3,  1]],
        [[ 1, -1,  1], [ 3,  1, -2]],
        [[ 1, -1,  1], [ 3,  2, -1]],
        [[ 1,  1,  1], [ 1,  2, -3]],
        [[ 1,  1,  1], [ 1, -3,  2]],
        [[ 1,  1,  1], [ 2,  1, -3]],
        [[ 1,  1,  1], [ 2, -3,  1]],
        [[ 1,  1,  1], [-3,  1,  2]],
        [[ 1,  1,  1], [-3,  2,  1]],
    ]).astype(float)
    burgers = b_n[:, 0] / np.sqrt(3)
    normals = b_n[:, 1] / np.linalg.norm(b_n[:, 1], axis=1)[:, None]

    # Get the sense vectors
    t = np.cross(normals, burgers)

    # Fix the screw dislocations (sense vectors are the burgers vectors)
    t[:4] = burgers[:4]

    # Calculate the outer product of the two vectors
    outer = np.einsum('...i,...j->...ij', burgers, t)

    # Convert to the (n_slip_systems, 9) matrix
    A_bcc = outer.reshape(-1, 9).T

    return A_bcc


def generate_HCP_A_matrix() -> np.ndarray:
    """Generate the A matrix for HCP crystal structure."""
    # Relevant Direcitons in [uvtw] notation
    b_n_uvtw = np.array([
        [[ 1,  1, -2, 0], [ 0,  0,  0, 1]],  # Basal
        [[ 1, -2,  1, 0], [ 0,  0,  0, 1]],
        [[-2,  1,  1, 0], [ 0,  0,  0, 1]],
        [[2, - 1, -1, 0], [ 0,  1, -1, 0]],  # Prismatic
        [[-1,  2, -1, 0], [ 1,  0, -1, 0]],
        [[ 1,  1, -2, 0], [ 1, -1,  0, 0]],
        [[-1, -1,  2, 3], [ 1,  0, -1, 1]],  # Pyramidal
        [[-2,  1,  1, 3], [ 1,  0, -1, 1]],
        [[ 1,  1, -2, 3], [ 0, -1,  1, 1]],
        [[-1,  2, -1, 3], [ 0, -1,  1, 1]],
        [[ 2, -1, -1, 3], [-1,  1,  0, 1]],
        [[ 1, -2,  1, 3], [-1,  1,  0, 1]],
        [[ 2, -1, -1, 3], [-1,  0,  1, 1]],
        [[ 1,  1, -2, 3], [-1,  0,  1, 1]],
        [[-1, -1,  2, 3], [ 0,  1, -1, 1]],
        [[ 1, -2,  1, 3], [ 0,  1, -1, 1]],
        [[-2,  1,  1, 3], [ 1, -1,  0, 1]],
        [[-1,  2, -1, 3], [ 1, -1,  0, 1]],
    ]).astype(float)

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
    outer = np.einsum('...i,...j->...ij', burgers, t)
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
    return A.T.dot(np.linalg.inv(A.dot(A.T)))


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
        x_trans = np.pad(grain_ids[:-1,...] != grain_ids[1:,...], ((0,1), (0,0), (0,0)))
    if shape[1] > 1:
        y_trans = np.pad(grain_ids[:,:-1,:] != grain_ids[:,1:,:], ((0,0), (0,1), (0,0)))
    if shape[2] > 1:
        z_trans = np.pad(grain_ids[...,:-1] != grain_ids[...,1:], ((0,0), (0,0), (0,1)))

    # Interior points
    interior_mask = np.zeros_like(grain_ids, dtype=bool)
    interior_mask[1:-1,:,:] = True
    
    # X-direction vectorized analysis
    if shape[0] > 1:
        completeness[0,:,:,0] = np.where(x_trans[0,:,:], 0, 1)   # Forward differences for first slice
        completeness[-1,:,:,0] = np.where(x_trans[-2,:,:], 0, 2) # Backward differences for last slice
        completeness[1:-1,:,:,0] = np.select(                    # Central differences
            [                                  
                (x_trans[:-2,:,:] & x_trans[1:-1,:,:]),
                x_trans[:-2,:,:],
                x_trans[1:-1,:,:],
            ],
            [0, 1, 2],
            default=3
        )
    
    # Y-direction vectorized analysis
    if shape[1] > 1:
        completeness[:,0,:,1] = np.where(y_trans[:,0,:], 0, 1)
        completeness[:,-1,:,1] = np.where(y_trans[:,-2,:], 0, 2)
        completeness[:,1:-1,:,1] = np.select(
            [
                (y_trans[:,:-2,:] & y_trans[:,1:-1,:]),
                y_trans[:,:-2,:],
                y_trans[:,1:-1,:]
            ],
            [0, 1, 2],
            default=3
        )
    
    # Z-direction (similar logic)
    if shape[2] > 1:
        completeness[:,:,0,2] = np.where(z_trans[:,:,0], 0, 1)
        completeness[:,:,-1,2] = np.where(z_trans[:,:,-2], 0, 2)
        completeness[:,:,1:-1,2] = np.select(
            [
                (z_trans[:,:,:-2] & z_trans[:,:,1:-1]),
                z_trans[:,:,:-2],
                z_trans[:,:,1:-1]
            ],
            [0, 1, 2],
            default=3
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
    scale = np.zeros(shape + (3,), dtype=np.float32)

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


def get_finite_difference_coordinates(grain_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the coordinates for finite difference pairs in a 3D EBSD dataset.
    
    Args:
        grain_ids: 3D numpy array containing grain IDs
        
    Returns:
        Tuple of three 3D arrays containing the coordinates of the first voxel, the coordinates of the second voxel, and the scale factor
    """
    # Get the completeness array
    completeness = get_completeness(grain_ids)
    
    # Get the neighbors
    coords0, coords1, scale = get_neighbors(completeness)
    
    return (coords0, coords1, scale)


def get_orientation_gradients(quats: np.ndarray, pts0: np.ndarray, pts1: np.ndarray, distances: np.ndarray, cs: int, n_cpus: int = None) -> np.ndarray:
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
    # Get the cpus
    if n_cpus is None:
        n_cpus = os.cpu_count() - 1

    # Get the shape
    shape = quats.shape[:-1]
    
    # Run calculations in parallel, split up by the z-axis
    # with mp.Pool(processes=n_cpus) as pool:
    #     results = list(
    #         tqdm(
    #             pool.imap(
    #                 get_rotation_vectors,
    #                 [(quats[i], pts0[i], pts1[i], cs) for i in range(shape[0])],
    #             ),
    #             total=shape[0],
    #             desc="Calculating Orientation Gradients",
    #         )
    #     )
    # rot_vectors = np.stack(results, axis=0).reshape(shape + (3, 3))
    rot_vectors = get_rotation_vectors(quats, pts0, pts1, cs)

    # Get the misorientations from the rotation vectors
    misorientation = np.linalg.norm(rot_vectors, axis=-1)

    # Get the orientation gradients
    gradient_tensors = np.zeros_like(rot_vectors)
    m = misorientation != 0
    print(gradient_tensors.shape, rot_vectors.shape, distances.shape)
    print(gradient_tensors[m].shape, rot_vectors[m].shape, distances[m].shape)
    print(distances[m][..., None].shape)
    gradient_tensors[m] = rot_vectors[m] / distances[m][..., None]

    return gradient_tensors, misorientation
    

def get_rotation_vectors(*args) -> np.ndarray:
    """Calculate the orientation gradients for a 3D EBSD dataset.
    This is essentially the rotation vectors corresponding to the disorientation between neighboring voxels,
    divided by the spacing along each dimension. The result is a 3x3 matrix for each voxel.
    Note that the arguments are passed as a single tuple to allow for parallel processing,
    but the function can also be called with the arguments passed individually.

    Args:
        quats: 3D numpy array containing quaternions, (X, Y, Z, 4)
        pts0: 3D numpy array containing the coordinates of the first voxel in the finite difference pairs, (X, Y, Z, 3)
        pts1: 3D numpy array containing the coordinates of the second voxel in the finite difference pairs, (X, Y, Z, 3)
        distances: 3D numpy array containing the distances between the finite difference pairs, (X, Y, Z, 3)
        cs: The crystal structure of the material. 1 for FCC, 2 for BCC, 3 for HCP.

    Returns:
        3D numpy array containing the orientation gradients, (X, Y, Z, 3, 3)
          This is essectially 3 rotation vectors corresponding to the disorientation between neighboring voxels,
          each divided by the distance between the finite difference pair. The 3x3 matrix for each point is the rotation vector for each axis.
    """
    # Handle the arguments
    if len(args) == 1:
        quats, pts0, pts1, cs = args[0]
    else:
        quats, pts0, pts1, cs = args
    if quats.shape[:-1] != pts0.shape[:-2] or quats.shape[:-1] != pts1.shape[:-2]:
        raise ValueError("Quaternions and points must have the same shape.")

    # Make sure the data is 3D
    if quats.ndim != 4:
        quats = quats[np.newaxis, ...]
        pts0 = pts0[np.newaxis, ...]
        pts1 = pts1[np.newaxis, ...]

    # Handle the crystal structure
    if cs not in [1, 2, 3]:
        raise ValueError("Crystal structure must be 1, 2, or 3.")
    elif cs == 1 or cs == 2:
        laue_id = 11
    else:
        laue_id = 9

    # Get shape
    shape = quats.shape[:-1]

    # Create global mask
    valid = np.any(pts0 != pts1, axis=-1)

    # Create the output arrays
    dis_quats = np.stack([np.zeros_like(quats), np.zeros_like(quats), np.zeros_like(quats)], axis=0)
    for i in range(3):
        # Create a mask of where the two points are not the same
        m = valid[..., i]
        if not np.any(m):
            continue
        q0 = quats[tuple(pts0[m][:, i].reshape(-1, 3).T)]  # (X*Y*Z, 4)
        q1 = quats[tuple(pts1[m][:, i].reshape(-1, 3).T)]  # (X*Y*Z, 4)
        dis_quats[i, m] = quaternions.qu_disorientation(q0, q1, laue_id, laue_id)  # (X*Y*Z, 4)
    rot_vectors = quaternions.qu_log(dis_quats) * 2  # (3, X, Y, Z, 3)
    return rot_vectors.transpose(1, 2, 3, 0, 4)


def minimize(alpha, cs, A, B, burgers, minimization='l2') -> np.ndarray:
    # Equation to be solved -> A*rho[array form] = Lambda[Nye in array form] 
    # Solve: A*rho = Lambd
    # Nye tensor must be converted into array form Lambda
    # Get shape
    shape = alpha.shape[:-2]
    numSlip = A.shape[1]
    Lambda = alpha.reshape(shape + (-1, 1))  # Shape (9x1)
    if minimization == 'l2':
        if cs == 2 or cs == 3:
            # two steps to solve via minimize‖Ax−b‖2
            dd = B.dot(Lambda)
            if numSlip > 9 & cs == 3:
                # TODO: Handle c axis slip systems which have a different burgers vector
                burgers_ca = 4.68
                burgers_ca = burgers_ca*1E-10
                dd[:9] = dd[:9] / burgers
                dd[9:33] = dd[9:33] / burgers_ca
            else:
                dd = dd/burgers
        else:
            dd = B.dot(Lambda)/burgers  # same as matmul
                
    elif minimization == 'l1':
        n_constraints, n_slip_systems = A.shape
        dd = np.zeros((n_slip_systems,) + shape)
        print("A", A.shape)
        print("Lambda", Lambda.shape)
        for idx in tqdm(np.ndindex(shape), total=np.prod(shape), desc="Minimizing"):
            i, j, k = idx
            c = np.hstack((np.zeros(n_slip_systems), np.ones(n_slip_systems)))
            A_eq = np.hstack([A, np.zeros((n_constraints, n_slip_systems))])
            b_eq = Lambda[i, j, k].reshape(-1)
            I = np.eye(n_slip_systems)
            # A_ub = np.vstack([np.hstack([I, -I]), np.hstack([-I, -I])])
            # b_ub = np.zeros(2*n_slip_systems)
            bounds = [(0, np.inf)]*n_slip_systems*2
            bounds = np.array(bounds)
            result = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            # result = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            dd[:, i, j, k] = result.x[:n_slip_systems] / burgers
    else:
        raise ValueError("Minimization scheme not recognized. Please choose either 'l1' or 'l2'")
    dd = dd.reshape((B.shape[0],) + shape).transpose(1, 2, 3, 0)
    return dd




if __name__ == "__main__":
    import utillities as utils
    import matplotlib.pyplot as plt

    which = "2D"

    cs = 1
    minimization = 'l2'
    n_cpus = 1

    if which == "2D":
        path = "E:/rolled_Al/merged_1x1.ang"
        burgers = 2.86e-10
        euler, ids, spacing = utils.read_ang(path)
        euler = euler[:, :300, :300]
        ids = ids[:, :300, :300]

    elif which == "3D":
        path = "D:/Research/CoNi_90/Data/3D/CoNi90.dream3d"
        burgers = 2.48e-10
        euler, ids, spacing = utils.read_dream3d(path)
        # euler = euler[200:300, 200:300, 200:300]
        # ids = ids[200:300, 200:300, 200:300]

    spacing *= 1e-6
    A, B = get_linear_operator(cs)

    quats = rotations.eu2qu(euler)
    print(" ")
    np.set_printoptions(linewidth=200)

    import time
    t0 = time.time()
    nbrs0, nbrs1, distances = get_finite_difference_coordinates(ids)
    distances *= spacing
    print("Neighbors time:", time.time() - t0)
    
    t0 = time.time()
    dphi, mis = get_orientation_gradients(quats, nbrs0, nbrs1, distances, cs, n_cpus)
    print("Orientation gradients time:", time.time() - t0)
    
    t0 = time.time()
    trace = np.trace(dphi, axis1=3, axis2=4)
    alpha = dphi.transpose(0, 1, 2, 4, 3) - trace[..., None, None]
    print("Alpha time:", time.time() - t0)

    t0 = time.time()
    dd = minimize(alpha, cs, A, B, burgers, minimization)
    dd = np.abs(dd)
    print("Minimization time:", time.time() - t0)

    print("Misorientation", mis.min(), mis.mean(), mis.max())
    print("Dislocation Density", dd.min(), dd.mean(), dd.max())

    avg_mis = np.rad2deg(np.mean(mis, axis=-1))

    dd_total = np.sum(dd, axis=-1)
    dd_total = np.log10(dd_total + 1e-6)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    im_mis = ax[0].imshow(avg_mis[0], cmap='viridis')
    im_gnd = ax[1].imshow(dd_total[0], cmap='RdBu_r')

    plt.tight_layout()
    plt.subplots_adjust(right=0.89, wspace=0.5)
    l = ax[0].get_position()
    cbar_ax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    fig.colorbar(im_mis, cax=cbar_ax, label="Misorientation (degrees)")
    l = ax[1].get_position()
    cbar_ax = fig.add_axes([l.x1 + 0.01, l.y0, 0.02, l.height])
    fig.colorbar(im_gnd, cax=cbar_ax, label="Log10(Dislocation Density)")
    plt.show()
    
    