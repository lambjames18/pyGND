import numpy as np
from scipy.optimize import linprog

class GND:
    def __init__(self, cs: int, burgers: float, slip_systems: str, G: float, nu: float, scheme="l2"):
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
        self.burgers = burgers * 1e-10
        self. G = G
        if self.G is not None:
            self.G *= 1e9
        self.nu = nu
        self.scheme = scheme
        self.set_A_matrix(slip_systems.strip().replace(" ", "").lower())
        self.get_crystallography()
        self.get_symmetry_operators()

    def preflight(self):
        if self.scheme == "l1" and (self.G is None or self.nu is None):
            raise ValueError("Shear modulus and Poisson's ratio must be provided for the L1 scheme.")
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
            return

    def set_data(self, coordinates, euler_angles, feature_ids, spacing):
        self.coordinates = coordinates
        self.euler_angles = euler_angles.astype(np.float64)
        self.GAO = self.eu2om_multi(self.euler_angles)
        self.featIDs = feature_ids
        self.spacing = spacing
    
    def enforce_mask_on_input(self, mask):
        self.GAO[:, :, mask] = 0.0
    
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
            avg_misori = np.mean(np.abs(dthe))
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
            # 576 (24 x 24) axis/angle pairs exist for any two cubic crystal lattices
            sym1 =  np.array([[ 1,  0,  0], [ 0,  1,  0], [ 0,  0,  1]])
            sym2 =  np.array([[ 0,  0,  1], [ 1,  0,  0], [ 0,  1,  0]])
            sym3 =  np.array([[ 0,  1,  0], [ 0,  0,  1], [ 1,  0,  0]])
            sym4 =  np.array([[ 0, -1,  0], [ 0,  0,  1], [-1,  0,  0]])
            sym5 =  np.array([[ 0, -1,  0], [ 0,  0, -1], [ 1,  0,  0]])
            sym6 =  np.array([[ 0,  1,  0], [ 0,  0, -1], [-1,  0,  0]])
            sym7 =  np.array([[ 0,  0, -1], [ 1,  0,  0], [ 0, -1,  0]])
            sym8 =  np.array([[ 0,  0, -1], [-1,  0,  0], [ 0,  1,  0]])
            sym9 =  np.array([[ 0,  0,  1], [-1,  0,  0], [ 0, -1,  0]])
            sym10 = np.array([[-1,  0,  0], [ 0,  1,  0], [ 0,  0, -1]])
            sym11 = np.array([[-1,  0,  0], [ 0, -1,  0], [ 0,  0,  1]])
            sym12 = np.array([[ 1,  0,  0], [ 0, -1,  0], [ 0,  0, -1]])
            sym13 = np.array([[ 0,  0, -1], [ 0, -1,  0], [-1,  0,  0]])
            sym14 = np.array([[ 0,  0,  1], [ 0, -1,  0], [ 1,  0,  0]])
            sym15 = np.array([[ 0,  0,  1], [ 0,  1,  0], [-1,  0,  0]])
            sym16 = np.array([[ 0,  0, -1], [ 0,  1,  0], [ 1,  0,  0]])
            sym17 = np.array([[-1,  0,  0], [ 0,  0, -1], [ 0, -1,  0]])
            sym18 = np.array([[ 1,  0,  0], [ 0,  0, -1], [ 0,  1,  0]])
            sym19 = np.array([[ 1,  0,  0], [ 0,  0,  1], [ 0, -1,  0]])
            sym20 = np.array([[-1,  0,  0], [ 0,  0,  1], [ 0,  1,  0]])
            sym21 = np.array([[ 0, -1,  0], [-1,  0,  0], [ 0,  0, -1]])
            sym22 = np.array([[ 0,  1,  0], [-1,  0,  0], [ 0,  0, -1]])
            sym23 = np.array([[ 0,  1,  0], [ 1,  0,  0], [ 0,  0, -1]])
            sym24 = np.array([[ 0, -1,  0], [ 1,  0,  0], [ 0,  0, -1]])
            symOp = np.dstack((sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18, sym19, sym20, sym21, sym22, sym23, sym24))
            
        elif self.cs == 3:
            # there are 12 symmetry operators for hexagonal symmetries
            # like A matrix for HCP, ortho-hexagonal coordinates
            # 144 (12 x 12) axis/angle pairs exist for any two hexagonal lattices
            a = np.sqrt(3)/2
            sym1 =  np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            sym2 =  np.array([[-0.5, a, 0], [-a, -0.5, 0], [0, 0, 1]])
            sym3 =  np.array([[-0.5, -a, 0], [a, -0.5, 0], [0, 0, 1]])
            sym4 =  np.array([[0.5, a, 0], [-a, 0.5, 0], [0, 0, 1]])
            sym5 =  np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
            sym6 =  np.array([[0.5, -a, 0], [a, 0.5, 0], [0, 0, 1]])
            sym7 =  np.array([[-0.5, -a, 0], [-a, 0.5, 0], [0, 0, -1]])
            sym8 =  np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            sym9 =  np.array([[-0.5, a, 0], [a, 0.5, 0], [0, 0, -1]])
            sym10 = np.array([[0.5, a, 0], [a, -0.5, 0], [0, 0, -1]])
            sym11 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            sym12 = np.array([[0.5, -a, 0], [-a, -0.5, 0], [0, 0, -1]])
            symOp = np.dstack((sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8, sym9, sym10, sym11, sym12))
        else:
            print('\nWarning! Crystal structure is not known. No symmetry operators have been defined.\n\n')
        self.symOp = symOp
        print("\t{} symmetry operators".format(self.symOp.shape[-1]))

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
            self.A = np.zeros((9,18)) # dummy variable
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
        self._latestgA = gA
        self._latestgB = gB
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
        if self.scheme == 'l2':
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
        elif self.scheme == 'l1':
            c = (np.ones(A.shape[1]) * self.G * burgers**2 / (4 * np.pi))
            c[:12] *= (1 - self.nu)**(-1)
            optimum = linprog(c, A_eq=A, b_eq=Lambda, bounds=(0, None))
            dd = optimum.x / burgers
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

    def _determine_dthe_slow(self, XenvCompleteness, YenvCompleteness, ZenvCompleteness, GAO, x1, x2, x3, symOp):

        # determine misorientation and kappa based on material point neighborhood
        dthe = np.zeros((3, 3))

        # orientation matrix of material point
        gA = GAO[:, :, x1, x2, x3]

        # switch statement evaluating expression for x environment
        if XenvCompleteness == 'backward':
            gE = GAO[:, :, x1-1, x2, x3]  #setting Euler Angle at x - 1
            # First Nearest Neighbors 1st order backward difference----
            diffOperatorX = 1
            # calc specific miorientation angles for kappa calc
            dthe[0, 0] = self._deltathetak_original(gE, gA, 1, symOp)
            dthe[1, 0] = self._deltathetak_original(gE, gA, 2, symOp)
            dthe[2, 0] = self._deltathetak_original(gE, gA, 3, symOp)
            
        elif XenvCompleteness ==  'forward':
            gB = GAO[:, :, x1+1, x2, x3]  #setting Euler Angle at x + 1
            # First Nearest Neighbors 1st order forward difference-----
            diffOperatorX = 1
            # calc specific miorientation angles for kappa calc
            dthe[0, 0] = self._deltathetak_original(gA, gB, 1, symOp)
            dthe[1, 0] = self._deltathetak_original(gA, gB, 2, symOp)
            dthe[2, 0] = self._deltathetak_original(gA, gB, 3, symOp)
            
        elif XenvCompleteness == 'central':
            gB = GAO[:, :, x1+1, x2, x3]  #setting Euler Angle at x + 1
            gE = GAO[:, :, x1-1, x2, x3]  #setting Euler Angle at x - 1
            # central finite difference
            diffOperatorX = 2
            # calc specific miorientation angles for kappa calc
            dthe[0, 0] = self._deltathetak_original(gE, gB, 1, symOp)
            dthe[1, 0] = self._deltathetak_original(gE, gB, 2, symOp)
            dthe[2, 0] = self._deltathetak_original(gE, gB, 3, symOp)
            
        elif XenvCompleteness ==  'constant':
            # in case no misorientation present
            diffOperatorX = 1
            dthe[0, 0] = 0
            dthe[1, 0] = 0
            dthe[2, 0] = 0

        # switch statement evaluating expression for y environment
        if YenvCompleteness == 'backward':
            gF = GAO[:, :, x1, x2-1, x3]  #setting Euler Angle at y - 1
            # First Nearest Neighbors 1st order backward difference----
            diffOperatorY = 1
            # calc specific miorientation angles for kappa calc
            dthe[0, 1] = self._deltathetak_original(gF, gA, 1, symOp)
            dthe[1, 1] = self._deltathetak_original(gF, gA, 2, symOp)
            dthe[2, 1] = self._deltathetak_original(gF, gA, 3, symOp)
            
        elif YenvCompleteness == 'forward':
            gC = GAO[:, :, x1, x2+1, x3]  #setting Euler Angle at y + 1
            # First Nearest Neighbors 1st order forward difference----
            diffOperatorY = 1
            # calc specific miorientation angles for kappa calc
            dthe[0, 1] = self._deltathetak_original(gA, gC, 1, symOp)
            dthe[1, 1] = self._deltathetak_original(gA, gC, 2, symOp)
            dthe[2, 1] = self._deltathetak_original(gA, gC, 3, symOp)
            
        elif YenvCompleteness == 'central':
            gC = GAO[:, :, x1, x2+1, x3]  #setting Euler Angle at y + 1
            gF = GAO[:, :, x1, x2-1, x3]  #setting Euler Angle at y - 1
            # central finite difference
            diffOperatorY = 2
            # calc specific miorientation angles for kappa calc
            dthe[0, 1] = self._deltathetak_original(gF, gC, 1, symOp)
            dthe[1, 1] = self._deltathetak_original(gF, gC, 2, symOp)
            dthe[2, 1] = self._deltathetak_original(gF, gC, 3, symOp)
            
        elif YenvCompleteness ==  'constant':
            # in case no misorientation, or no neighbor
            diffOperatorY = 1
            dthe[0, 1] = 0
            dthe[1, 1] = 0
            dthe[2, 1] = 0

        # switch statement evaluating expression for environment
        if ZenvCompleteness == 'backward':
            gG = GAO[:, :, x1, x2, x3-1]  #setting Euler Angle at z - 1
            # First Nearest Neighbors 1st order backward difference----
            diffOperatorZ = 1
            # calc specific miorientation angles for kappa calc
            dthe[0,2] = self._deltathetak_original(gG, gA, 1, symOp)
            dthe[1,2] = self._deltathetak_original(gG, gA, 2, symOp)
            dthe[2,2] = self._deltathetak_original(gG, gA, 3, symOp)
            
        elif ZenvCompleteness == 'forward':
            gD = GAO[:, :, x1, x2, x3+1]  #setting Euler Angle at z + 1
            # First Nearest Neighbors 1st order forward difference-----
            diffOperatorZ = 1
            # calc specific miorientation angles for kappa calc
            dthe[0, 2] = self._deltathetak_original(gA, gD, 1, symOp)
            dthe[1, 2] = self._deltathetak_original(gA, gD, 2, symOp)
            dthe[2, 2] = self._deltathetak_original(gA, gD, 3, symOp)
            
        elif ZenvCompleteness == 'central':
            gD = GAO[:, :, x1, x2, x3+1]  #setting Euler Angle at z + 1
            gG = GAO[:, :, x1, x2, x3-1]  #setting Euler Angle at z - 1
            diffOperatorZ = 2
            # calc specific miorientation angles for kappa calc
            dthe[0, 2] = self._deltathetak_original(gG, gD, 1, symOp)
            dthe[1, 2] = self._deltathetak_original(gG, gD, 2, symOp)
            dthe[2, 2] = self._deltathetak_original(gG, gD, 3, symOp)
            
        elif ZenvCompleteness == 'constant':
            # zero misorientation along axis if no neighbor
            diffOperatorZ = 1
            dthe[0, 2] = 0
            dthe[1, 2] = 0
            dthe[2, 2] = 0 
        return dthe, diffOperatorX, diffOperatorY, diffOperatorZ

    def _deltathetak_original(self, gA, gB, k, symOp):
        self._latestgA = gA
        self._latestgB = gB
        if (gA == gB).all():
            return 0
        else:
            # For Misorientation
            # delg = np.linalg.solve(gA, gB).conj().transpose(0, 2, 1)
            # l = np.around((np.diagonal(delgs).sum() - 1) / 2, 6)
            # deltheta = np.around(np.arccos(l), 6)
            # if deltheta == 0.0: return 0.0
            # elif k == 1: return -(delg[1, 2] - delg[2, 1]) * (deltheta / (2 * np.sin(deltheta)))
            # elif k == 2: return -(delg[2, 0] - delg[0, 2]) * (deltheta / (2 * np.sin(deltheta)))
            # elif k == 3: return -(delg[0, 1] - delg[1, 0]) * (deltheta / (2 * np.sin(deltheta)))
            # else: return 0.0

            # For disorientation
            numSym = symOp.shape # 3x3x24
            # misori_matrix = np.zeros((numSym[2]**2, 3), dtype=np.float32)
            misori_matrix = np.zeros(numSym[2]**2, dtype=np.float32)
            gA_temps = np.einsum("ijl,jkl->ikl", symOp, np.einsum('ij,kjl->ikl', gA, symOp))
            gB_temps = np.einsum("ijl,jkl->ikl", symOp, np.einsum('ij,kjl->ikl', gB, symOp))
            # gA_temps = np.einsum("ijl,jk->ikl", symOp, gA)
            # gB_temps = np.einsum("ijl,jk->ikl", symOp, gB)
            gA_temps = np.moveaxis(gA_temps, 2, 0).conj().transpose(0, 2, 1)
            gB_temps = np.moveaxis(gB_temps, 2, 0).conj().transpose(0, 2, 1)
            indices = np.indices((numSym[2], numSym[2])).reshape(2, -1)
            gA_temps = gA_temps[indices[0]] # 24x24
            gB_temps = gB_temps[indices[1]] # 24x24
            delgs = np.linalg.solve(gA_temps, gB_temps).conj().transpose(0, 2, 1) # 576x3x3
            # delgs = np.linalg.solve(gA_temps, gB_temps).transpose(0, 2, 1) # 576x3x3
            l = np.around((np.diagonal(delgs, axis1=1, axis2=2).sum(axis=1) - 1) / 2, 6)
            delthetas = np.around(np.arccos(l), 6)
            
            # Where deltheta is zero, misorientation matrix is zero
            misori_matrix[delthetas == 0.0] = 0.0
            mask = delthetas != 0.0
            coeff = (delthetas[mask] / (2 * np.sin(delthetas[mask])))
            
            if k == 1:   misori_matrix[mask] = -(delgs[mask, 0, 1] - delgs[mask, 1, 0]) * coeff
            elif k == 2: misori_matrix[mask] = -(delgs[mask, 2, 0] - delgs[mask, 0, 2]) * coeff
            elif k == 3: misori_matrix[mask] = -(delgs[mask, 1, 2] - delgs[mask, 2, 1]) * coeff
            else:        misori_matrix[mask] = 0
            
            d_col = np.argmin(np.abs(misori_matrix))
            disori = np.abs(misori_matrix[d_col])
            return disori

    def _determine_neighborhood_old(self, featIDs, x1, x2, x3):
        # checking completeness of the voxel neighborhood in x dimension
        # if the voxel is on the edge of the microstructure
        if x1 == 0:
            XenvCompleteness = 'forward'
        # if the voxel is on the other edge of the microstructure
        elif x1 == featIDs.shape[0] - 1:
            XenvCompleteness = 'backward'
        # if the voxel is on the edge of a grain
        elif featIDs[x1, x2, x3] == featIDs[x1+1, x2, x3] & featIDs[x1, x2, x3] != featIDs[x1-1, x2, x3]:
            XenvCompleteness = 'forward'
        # if the voxel is on the other edge of a grain
        elif featIDs[x1, x2, x3] != featIDs[x1+1, x2, x3] & featIDs[x1, x2, x3] == featIDs[x1-1, x2, x3]:
            XenvCompleteness = 'backward'
        # if the voxel is somewhere in the middle of a grain
        elif featIDs[x1, x2, x3] == featIDs[x1+1, x2, x3] & featIDs[x1, x2, x3] == featIDs[x1-1, x2, x3]:
            XenvCompleteness = 'central'
        else:
            XenvCompleteness = 'constant'

        # checking completeness of the voxel neighborhood in y dimension    
        # if the voxel is on the edge of the microstructure
        if x2 == 0:
            YenvCompleteness = 'forward'
        # if the voxel is on the other edge of the microstructure
        elif x2 == featIDs.shape[1] - 1:
            YenvCompleteness = 'backward'
        # if the voxel is on the edge of a grain
        elif featIDs[x1, x2, x3] == featIDs[x1, x2+1, x3] & featIDs[x1, x2, x3] != featIDs[x1, x2-1, x3]:
            YenvCompleteness = 'forward'
        # if the voxel is on the other edge of a grain
        elif featIDs[x1, x2, x3] != featIDs[x1, x2+1, x3] & featIDs[x1, x2, x3] == featIDs[x1, x2-1, x3]:
            YenvCompleteness = 'backward'
        # if the voxel is somewhere in the middle of a grain
        elif featIDs[x1, x2, x3] == featIDs[x1, x2+1, x3] & featIDs[x1, x2, x3] == featIDs[x1, x2-1, x3]:
            YenvCompleteness = 'central'
        else:
            YenvCompleteness = 'constant'

        # checking completeness of the voxel neighborhood in z dimension 
        # if the voxel is on the edge of the microstructure
        if x3 == 0:
            ZenvCompleteness = 'forward'
        # if the voxel is on the other edge of the microstructure
        elif x3 == featIDs.shape[2] - 1:
            ZenvCompleteness = 'backward'
        # if the voxel is on the edge of a grain
        elif featIDs[x1, x2, x3] == featIDs[x1, x2, x3+1] & featIDs[x1, x2, x3] != featIDs[x1, x2, x3-1]:
            ZenvCompleteness = 'forward'
        # if the voxel is on the other edge of a grain
        elif featIDs[x1, x2, x3] != featIDs[x1, x2, x3+1] & featIDs[x1, x2, x3] == featIDs[x1, x2, x3-1]:
            ZenvCompleteness = 'backward'
        # if the voxel is somewhere in the middle of a grain
        elif featIDs[x1, x2, x3] == featIDs[x1, x2, x3+1] & featIDs[x1, x2, x3] == featIDs[x1, x2, x3-1]:
            ZenvCompleteness = 'central'
        else:
            ZenvCompleteness = 'constant'

        return XenvCompleteness,YenvCompleteness,ZenvCompleteness