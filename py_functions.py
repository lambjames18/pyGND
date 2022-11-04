import numpy as np


def eu2om_multi(eu: np.ndarray):
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
    return om.reshape((3, 3) + vol_shape)



def eu2om_mod(eu):
    """Euler angles bunge convention to orientation matrix
    Args:
        eu (np.ndarray): Euler angles in radians
    Returns:
        np.ndarray: Orientation matrix"""
    thr = 1e-10

    c1, c2, c3 = np.cos(eu)

    s1, s2, s3 = np.sin(eu)

    om = np.array([[ c1 * c3 - c2 * s1 * s3, c3 * s1 + c1 * c2 * s3, s3 * s2],
                   [-c1 * s3 - c2 * c3 * s1, c1 * c2 * c3 - s1 * s3, c3 * s2],
                   [                s2 * s1,               -c1 * s2,      c2]])

    om = np.where(np.abs(om) < thr, 0.0, om)
    return om


def eu2om(eu: np.ndarray):
    """Euler angles bunge convention to orientation matrix
    Args:
        eu (np.ndarray): Euler angles in degrees
    Returns:
        np.ndarray: Orientation matrix"""
    q = eu2om_mod(np.radians(eu))
    return q


def determine_neighborhood(featIDs, x1, x2, x3):
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


def deltathetakV4_old(gA, gB, k, symOp):
    # determine how many symmetry cases to evaluate
    numSym = symOp.shape

    # preallocate number of unique misorientations to calc so disorientation
    # can be found
    misori_matrix = np.zeros(numSym[2]**2)

    # preallocate iterators
    gA_iter = 0
    gB_iter = 0

    # if material points have same orientation, ignore this process
    if (gB != gA).any() and gA.sum() != 0:
        # lines 17-37 correspond to misorientation calc with ori matrices
        for delg_iter in range(misori_matrix.shape[0]):
            gA_temp = symOp[:, :, gA_iter].dot(gA)
            gB_temp = symOp[:, :, gB_iter].dot(gB)
            # delg = gB_temp / gA_temp
            delg = np.linalg.solve(gA_temp.conj().T, gB_temp.conj().T).conj().T
            
            # calculate delta theta, skip trace(delg) function for speed
            deltheta = np.arccos((np.diag(delg).sum() - 1) / 2)
            
            if deltheta == 0:
                misori_matrix[delg_iter] = 0
            elif k == 1:
                misori_matrix[delg_iter] = -(delg[1, 2] - delg[2, 1]) * (deltheta/(2*np.sin(deltheta)))
            elif k == 2:
                misori_matrix[delg_iter] = -(delg[2, 0] - delg[0, 2]) * (deltheta/(2*np.sin(deltheta)))
            elif k == 3:
                misori_matrix[delg_iter] = -(delg[0, 1] - delg[1, 0]) * (deltheta/(2*np.sin(deltheta)))
            else:
                misori_matrix[delg_iter] = 0

            # storing misorientation with specific symmetry operator applied

            if gB_iter == numSym[2] - 1:
                gB_iter = 0
                gA_iter += 1
            else:
                gB_iter += 1

        # finding lowest value of misorientation (disorientation)
        d_col = np.argmin(np.abs(misori_matrix))

        # return disorientation
        disori = np.abs(misori_matrix[d_col])
    else:
        disori = 0

    return disori


def deltathetakV4(gA, gB, k, symOp):
    numSym = symOp.shape
    misori_matrix = np.zeros(numSym[2]**2)
    gA_temps = np.einsum("ijl,jkl->ikl", symOp, np.einsum('ij,kjl->ikl', gA, symOp))
    gB_temps = np.einsum("ijl,jkl->ikl", symOp, np.einsum('ij,kjl->ikl', gB, symOp))
    gA_temps = np.moveaxis(gA_temps, 2, 0).transpose(0, 2, 1)
    gB_temps = np.moveaxis(gB_temps, 2, 0).transpose(0, 2, 1)
    indices = np.indices((numSym[2], numSym[2])).reshape(2, -1)
    gA_temps = gA_temps[indices[0]]
    gB_temps = gB_temps[indices[1]]
    delgs = np.linalg.solve(gA_temps.conj(), gB_temps.conj()).conj().transpose(0, 2, 1)
    delthetas = np.arccos((np.diagonal(delgs, axis1=1, axis2=2).sum(axis=1) - 1) / 2)
    
    # Where deltheta is zero, misorientation matrix is zero
    misori_matrix[delthetas == 0] = 0
    
    if k == 1:
        misori_matrix[delthetas != 0] = -(delgs[delthetas != 0, 0, 1] - delgs[delthetas != 0, 1, 0]) * (delthetas[delthetas != 0] / (2 * np.sin(delthetas[delthetas != 0])))
    elif k == 2:
        misori_matrix[delthetas != 0] = -(delgs[delthetas != 0, 2, 0] - delgs[delthetas != 0, 0, 2]) * (delthetas[delthetas != 0] / (2 * np.sin(delthetas[delthetas != 0])))
    elif k == 3:
        misori_matrix[delthetas != 0] = -(delgs[delthetas != 0, 1, 2] - delgs[delthetas != 0, 2, 1]) * (delthetas[delthetas != 0] / (2 * np.sin(delthetas[delthetas != 0])))
    else:
        misori_matrix[delthetas != 0] = 0
    
    d_col = np.argmin(np.abs(misori_matrix))
    disori = np.abs(misori_matrix[d_col])
    return disori


def determine_dthe(XenvCompleteness, YenvCompleteness, ZenvCompleteness, GAO, x1, x2, x3, symOp):

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
        dthe[0, 0] = deltathetakV4(gE, gA, 1, symOp)
        dthe[1, 0] = deltathetakV4(gE, gA, 2, symOp)
        dthe[2, 0] = deltathetakV4(gE, gA, 3, symOp)
        
    elif XenvCompleteness ==  'forward':
        gB = GAO[:, :, x1+1, x2, x3]  #setting Euler Angle at x + 1
        # First Nearest Neighbors 1st order forward difference-----
        diffOperatorX = 1
        # calc specific miorientation angles for kappa calc
        dthe[0, 0] = deltathetakV4(gA, gB, 1, symOp)
        dthe[1, 0] = deltathetakV4(gA, gB, 2, symOp)
        dthe[2, 0] = deltathetakV4(gA, gB, 3, symOp)
        
    elif XenvCompleteness == 'central':
        gB = GAO[:, :, x1+1, x2, x3]  #setting Euler Angle at x + 1
        gE = GAO[:, :, x1-1, x2, x3]  #setting Euler Angle at x - 1
        # central finite difference
        diffOperatorX = 2
        # calc specific miorientation angles for kappa calc
        dthe[0, 0] = deltathetakV4(gE, gB, 1, symOp)
        dthe[1, 0] = deltathetakV4(gE, gB, 2, symOp)
        dthe[2, 0] = deltathetakV4(gE, gB, 3, symOp)
        
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
        dthe[0, 1] = deltathetakV4(gF, gA, 1, symOp)
        dthe[1, 1] = deltathetakV4(gF, gA, 2, symOp)
        dthe[2, 1] = deltathetakV4(gF, gA, 3, symOp)
        
    elif YenvCompleteness == 'forward':
        gC = GAO[:, :, x1, x2+1, x3]  #setting Euler Angle at y + 1
        # First Nearest Neighbors 1st order forward difference----
        diffOperatorY = 1
        # calc specific miorientation angles for kappa calc
        dthe[0, 1] = deltathetakV4(gA, gC, 1, symOp)
        dthe[1, 1] = deltathetakV4(gA, gC, 2, symOp)
        dthe[2, 1] = deltathetakV4(gA, gC, 3, symOp)
        
    elif YenvCompleteness == 'central':
        gC = GAO[:, :, x1, x2+1, x3]  #setting Euler Angle at y + 1
        gF = GAO[:, :, x1, x2-1, x3]  #setting Euler Angle at y - 1
        # central finite difference
        diffOperatorY = 2
        # calc specific miorientation angles for kappa calc
        dthe[0, 1] = deltathetakV4(gF, gC, 1, symOp)
        dthe[1, 1] = deltathetakV4(gF, gC, 2, symOp)
        dthe[2, 1] = deltathetakV4(gF, gC, 3, symOp)
        
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
        dthe[0,2] = deltathetakV4(gG, gA, 1, symOp)
        dthe[1,2] = deltathetakV4(gG, gA, 2, symOp)
        dthe[2,2] = deltathetakV4(gG, gA, 3, symOp)
        
    elif ZenvCompleteness == 'forward':
        gD = GAO[:, :, x1, x2, x3+1]  #setting Euler Angle at z + 1
        # First Nearest Neighbors 1st order forward difference-----
        diffOperatorZ = 1
        # calc specific miorientation angles for kappa calc
        dthe[0, 2] = deltathetakV4(gA, gD, 1, symOp)
        dthe[1, 2] = deltathetakV4(gA, gD, 2, symOp)
        dthe[2, 2] = deltathetakV4(gA, gD, 3, symOp)
        
    elif ZenvCompleteness == 'central':
        gD = GAO[:, :, x1, x2, x3+1]  #setting Euler Angle at z + 1
        gG = GAO[:, :, x1, x2, x3-1]  #setting Euler Angle at z - 1
        diffOperatorZ = 2
        # calc specific miorientation angles for kappa calc
        dthe[0, 2] = deltathetakV4(gG, gD, 1, symOp)
        dthe[1, 2] = deltathetakV4(gG, gD, 2, symOp)
        dthe[2, 2] = deltathetakV4(gG, gD, 3, symOp)
        
    elif ZenvCompleteness == 'constant':
        # zero misorientation along axis if no neighbor
        diffOperatorZ = 1
        dthe[0, 2] = 0
        dthe[1, 2] = 0
        dthe[2, 2] = 0 
    return dthe, diffOperatorX, diffOperatorY, diffOperatorZ


def determine_kappaV5(dthe, diffOperators, spacing):
    # kappa must be calculated for material point
    #----------------------------------------------------------------------
    # kappa = np.zeros((3, 3))
    # Calc three kappa components for x direction
    # kappa[:, 0] = dthe[:, 0] / (diffOperatorX * spacing[0])
    # Calc three kappas for y direction
    # kappa[:, 1] = dthe[:,1] / (diffOperatorY * spacing[1])
    # Calc three kappas for z direction                    
    # kappa[:, 2] = dthe[:, 2] / (diffOperatorZ * spacing[2])
    #----------------------------------------------------------------------
    kappa = dthe / (diffOperators * spacing)
    return kappa


def L2_SparseV2(alpha, cs, A, B, burgers):
    # L2 minimization with sparse solver

    # Equation to be solved -> A*rho[array form] = Lambda[Nye in array form] 
    # Can solve via minimize‖Ax−b‖2
    # requires two steps
    # [c,R] = qr(A,Lambda);
    # rho = R\c

    # Nye tensor must be converted into array form Lambda
    #----------------------------------------------------------------------
    Lambda = np.array([alpha[0,0],
                       alpha[0,1],
                       alpha[0,2],
                       alpha[1,0],
                       alpha[1,1],
                       alpha[1,2],
                       alpha[2,0],
                       alpha[2,1],
                       alpha[2,2]])

    if cs == 2 | cs == 3:
        # two steps to solve via minimize‖Ax−b‖2
        #[c,R] = qr(transpose(A_sparse),transpose(Lambda))
        B = A.T.dot(np.linalg.inv(A * A.T))
        dd = B.dot(Lambda)
        numSlip = A.shape[1]
        # calc dislocation density (rho) using burgers vector
        if numSlip > 9 & cs == 3:
            burgers_ca = 4.68
            burgers_ca = burgers_ca*1E-10
            dd[:9] = dd[:9] / burgers
            dd[9:33] = dd[9:33] / burgers_ca
        else:
            dd = dd/burgers

    else:
        # explicitly solve for FCC dislocation density with linear operator
        dd = B.dot(Lambda)
        # calc dislocation density (rho) using burgers vector
        dd = dd/burgers
    #-----------------------------------------------------------------------
    return dd

# Loop over all points in the volume
# Need to have coordinates of point and be able to grab neighboring coordinates
# Need orientation matrix for each point
# Need to be able to grab neighboring orientation matrices
# Need to be able to grab neighboring featIDs

def GND(coords, featIDs, GAO, cs, symOp, spacing, A, B, burgers):
    # Prep calucation of GND
    # burgers, A, numModes, numSlip, cs, B = xtal()
    # burgers = burgers * 1e-10
    # symOp = symmetry_operators(cs)
    
    # Get coordinates of current point
    x1 = coords[0].astype(int)
    x2 = coords[1].astype(int)
    x3 = coords[2].astype(int)

    # no calculations if inside void or outside microstructure
    if GAO[:, :, x1, x2, x3].sum() != 0:
        # Determine what neighborhood the point has
        # Returns forward, backward, central, or constant for the three directions
        XenvCompleteness, YenvCompleteness, ZenvCompleteness = determine_neighborhood(featIDs, x1, x2, x3)

        # Determine Disorientation between material points and neighbors, influenced by neighborhood
        # Calculate kappa for material points
        dthe, diffOperatorX, diffOperatorY, diffOperatorZ = determine_dthe(XenvCompleteness, YenvCompleteness, ZenvCompleteness, GAO, x1, x2, x3, symOp)

        # Calculate average misorientation from dthe
        avg_misori = np.mean(np.abs(dthe))

        # kappaSR = determine_kappaV5(dthe, diffOperatorX, diffOperatorY, diffOperatorZ, spacing)
        diffOperators = np.array([diffOperatorX, diffOperatorY, diffOperatorZ])
        kappaSR = determine_kappaV5(dthe, diffOperators, spacing)
        
        # Convert Kappa to crystal coordinates since dislocations are
        # described in crystal coordinates
        # kappaSRprime = GAO[:, :, x1, x2, x3].T.dot(kappaSR).dot(GAO[:, :, x1, x2, x3])
        # kappaSRprime = GAO[:, :, x1, x2, x3].T.dot(kappaSR[:, ::-1]).dot(GAO[:, :, x1, x2, x3])
        kappaSRprime = GAO[:, :, x1, x2, x3].dot(kappaSR).dot(GAO[:, :, x1, x2, x3].T)
        
        # Calculate Nye Tensor (alpha) from curvature kappa  
        alphaSR = kappaSRprime.T - np.trace(kappaSRprime)

        #function used to determine a total value of gnd density at one particular material point

        # determine dislocation densities (dd -> rho) from misorientations
        ddSR = L2_SparseV2(alphaSR, cs, A, B, burgers)

        # determine total gnd density to be sum of dislocation density across all
        # slip systems
        totalGNDdensitySR = np.abs(ddSR).sum()
        ddSR = np.abs(ddSR).T
        
        # repeat misorientation calculations using LR approach
        # it is just zero (not sure why, but it was commented out in original code)
        # totalGNDdensityLR = 0
    else:
        # tame output for voxels where misorientation can't be calc
        avg_misori = 0
        totalGNDdensitySR = 0
        # totalGNDdensityLR = 0
        ddSR_dim = A.shape[1]
        ddSR = np.zeros((1,ddSR_dim))

    # return totalGNDdensitySR, totalGNDdensityLR, avgMisori, ddSR
    return totalGNDdensitySR, avg_misori, ddSR


def import_data(data_path: str, grain_path: str):
    """Import data from .csv files
    Needs to be changed to read in sample scaling/resolution"""
    micro = np.loadtxt(data_path, delimiter=',', dtype=float)
    GrainIDs = np.loadtxt(grain_path, delimiter=',', dtype=int)

    X_scaling = 1.5
    Y_scaling = 1.5
    Z_scaling = 1.5

    X_spacing = X_scaling * 10**-6  # in meters
    Y_spacing = Y_scaling * 10**-6
    Z_spacing = Z_scaling * 10**-6
    spacing = np.array([Z_spacing, Y_spacing, X_spacing])

    micro[:, 2] = np.int32(micro[:, 2] / X_scaling)  # x
    micro[:, 1] = np.int32(micro[:, 1] / Y_scaling)  # y
    micro[:, 0] = np.int32(micro[:, 0] / Z_scaling)  # z

    # micro: 1st column: z, 2nd column: y, 3rd column: x, 4th column: phi1, 5th column: Phi, 6th column: phi2
    return micro, GrainIDs, spacing


def symmetry_operators(cs):
    # Symmetry operators used to determine the disorientation between two
    # points

    # global symOp

    # define symmetry operators for cubic or hexagonal symmetries
    if cs == 1 or cs == 2:
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
        
    elif cs == 3:
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
    return symOp


def xtal():
    """xtal
    decide which crystallography is relevant for material of interest
    includes burgers vector mag, linear operator B, or A matrix"""

    # cs = int(input('Input crystallography: \n 1: FCC \n 2: BCC \n 3: HCP\n\n'))
    cs = 1

    # define burgers vector magnitude
    # burgers = float(input('Input Burgers Vector (A): \n2.86A for Tantalum BCC\n2.5A for IN718 & AlNiCo9\n2.95A for Ti\n\n'))
    burgers = 2.5

    # generate full A matrices

    # create linear operator B for FCC
    # constants used for linear operator
    a = np.sqrt(3)/9
    c = np.sqrt(3)/84
    d = 1/18
    f = 3/14

    # Linear operator used to calculate dd from Nye components

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
        
    # BCC
    if cs == 2:
        A_bcc, checknorm = BCC_A_matrix_generationV2()
        print('Include which slip modes?\n')
        A_matrix_choice = int(input('1: screw + [110]\n2: screw + [112]\n3: screw + [123]\n4: screw + [110] + [112]\n5: screw + [110] + [112] + [123]\n'))
        if A_matrix_choice == 1:
            a_bcc = np.float64(A_bcc[:, :16])
            numModes = 2
        elif A_matrix_choice == 2:
            a_bcc = np.float64([A_bcc[:,:4], A_bcc[:,16:28]])
            numModes = 2
        elif A_matrix_choice == 3:
            a_bcc = np.float64([A_bcc[:,:4], A_bcc[:,28:]])
            numModes = 2
        elif A_matrix_choice == 4:
            a_bcc = np.float64(A_bcc[:,:28])
            numModes = 3
        elif A_matrix_choice == 5:
            numModes = 4
            a_bcc = np.float64(A_bcc)
        
        # 2.86A for Tantalum BCC
        # burgers = 2.86E-10;
        # A_sparse = sparse(a_bcc)
        A_sparse = a_bcc
        numNye, numSlip = A_sparse.shape
        
    # HCP
    elif cs == 3:
        d1, d2, d3, d4, d5 = HCP_A_matrix_mk3()
        print('Include which slip modes? \n')
        A_matrix_choice = int(input('1: basal\n2: basal + prismatic\n3: basal + prismatic + pyramidal(c+a)\n'))
        if A_matrix_choice == 1:
            A_hcp = np.array([d1, d2])
            numModes = 2
        elif A_matrix_choice == 2:
            A_hcp = np.array([d1, d2, d3])
            numModes = 3
        elif A_matrix_choice == 3:
            A_hcp = np.array([d1, d2, d3, d4, d5])
            numModes = 5
        
        # 0.295nm for Ti 
        # burgers = 2.95E-10;
        # A_sparse = sparse(A_hcp)
        A_sparse = A_hcp
        numNye, numSlip = A_sparse.shape
        
    # FCC
    else:
        print('Defaulting to FCC.')

        # .25nm, see by neutron diffraction via Zhang et. al.
        # burgers = 2.5E-10;
        A_sparse = np.zeros((9,18)) # dummy variable
        
        #defining number of slip systems and slip modes
        numSlip = 18
        numModes = 4
    return (burgers, A_sparse, numModes, numSlip, cs, B)


def BCC_A_matrix_generationV2():
    # BCC A matrix formulation
    # b vectors for systems {110}{112}{321} 1->4 as screw
    # b vectors for systems {110} 5->16 as edge
    # b vectors for systems {112} 17->28 as edge
    # b vectors for systems {123} 29->52 as edge

    bedge = np.float32((1/np.sqrt(3))* np.array([[ 1,  1, -1],  # {110}<111> SLIP
                                                 [ 1,  1, -1],
                                                 [ 1,  1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],  # {112}<111> SLIP
                                                 [-1, -1,  1],
                                                 [-1, -1,  1],
                                                 [-1, -1,  1],
                                                 [-1,  1,  1],
                                                 [-1,  1,  1],
                                                 [-1,  1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],  # {123}<111> SLIP
                                                 [ 1,  1, -1],
                                                 [ 1,  1, -1],
                                                 [ 1,  1, -1],
                                                 [ 1,  1, -1],
                                                 [ 1,  1, -1],
                                                 [ 1,  1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1, -1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1, -1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1],
                                                 [ 1,  1,  1]]))

    nedge = np.zeros((48,3))
    # {110}<111> SLIP
    nedge[:12] = np.float32(1/np.sqrt(2) * np.array([[0,  1,  1],
                                                     [1,  0,  1],
                                                     [1, -1,  0], 
                                                     [0,  1, -1],
                                                     [1,  0,  1],
                                                     [1,  1,  0],
                                                     [0,  1,  1],
                                                     [1,  0, -1],
                                                     [1,  1,  0],
                                                     [0,  1, -1],
                                                     [1,  0, -1],
                                                     [1, -1,  0]]))

    # {112}<111> SLIP
    nedge[12:24] = np.float32(1/np.sqrt(6) * np.array([[-2,  1, -1],
                                                       [ 1, -2, -1],
                                                       [ 1,  1,  2],
                                                       [-2, -1, -1],
                                                       [ 1,  2, -1],
                                                       [ 1, -1,  2],
                                                       [ 2,  1, -1],
                                                       [-1, -2, -1],
                                                       [-1,  1,  2],
                                                       [ 2, -1, -1],
                                                       [-1,  2, -1],
                                                       [-1, -1,  2]]))

    # {123}<111> SLIP
    nedge[24:] = np.float32(1/np.sqrt(14) * np.array([[ 1,  2,  3],     
                                                      [-1,  3,  2],
                                                      [ 2,  1,  3],
                                                      [-2,  3,  1],
                                                      [ 3, -1,  2],
                                                      [ 3, -2,  1],
                                                      [-1,  2, -3],
                                                      [ 1,  3, -2],
                                                      [ 2, -1,  3],
                                                      [ 2,  3, -1],
                                                      [ 3,  1,  2],
                                                      [ 3,  2,  1],
                                                      [ 1, -2, -3],
                                                      [ 1,  3,  2],
                                                      [ 2, -1, -3],
                                                      [ 2,  3,  1],
                                                      [ 3,  1, -2],
                                                      [ 3,  2, -1],
                                                      [ 1,  2, -3],
                                                      [ 1, -3,  2],
                                                      [ 2,  1, -3],
                                                      [ 2, -3,  1],
                                                      [-3,  1,  2],
                                                      [-3,  2,  1]]))

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
    t[:12] = np.array([[-0.8165,  0.4082, -0.4082],
                       [-0.4082,  0.8165,  0.4082],
                       [ 0.4082,  0.4082,  0.8165],
                       [-0.8165, -0.4082, -0.4082],
                       [ 0.4082,  0.8165, -0.4082],
                       [-0.4082,  0.4082, -0.8165],
                       [ 0.8165,  0.4082, -0.4082],
                       [-0.4082, -0.8165, -0.4082],
                       [ 0.4082, -0.4082, -0.8165],
                       [ 0.8165, -0.4082, -0.4082],
                       [ 0.4082, -0.8165,  0.4082],
                       [-0.4082, -0.4082,  0.8165]])


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
    checknorm = 1

    return A_bcc, checknorm


def HCP_A_matrix_mk3():
    # Relevant Direcitons in [uvtw] notation

    # Basal Slip
    bMBbasal = np.array([[1, 1, -2, 0], [1, -2, 1, 0], [-2, 1, 1, 0]])
    nMBbasal = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]])

    # Prismatic Slip
    bMBprismatic = np.array([[2, -1, -1, 0], [-1, 2, -1, 0], [1, 1, -2, 0]])
    nMBprismatic = np.array([[0, 1, -1, 0], [1, 0, -1, 0], [1, -1, 0, 0]])

    # Pyramidal <c+a> Slip
    bMBpyramidalCplusA = np.array([[-1, -1,  2, 3],
                                   [-2,  1,  1, 3],
                                   [ 1,  1, -2, 3],
                                   [-1,  2, -1, 3],
                                   [ 2, -1, -1, 3],
                                   [ 1, -2,  1, 3],
                                   [ 2, -1, -1, 3],
                                   [ 1,  1, -2, 3],
                                   [-1, -1,  2, 3],
                                   [ 1, -2,  1, 3],
                                   [-2,  1,  1, 3],
                                   [-1,  2, -1, 3]])
                    
    nMBpyramidalCplusA = np.array([[ 1,  0, -1, 1],
                                   [ 1,  0, -1, 1],
                                   [ 0, -1,  1, 1],
                                   [ 0, -1,  1, 1],
                                   [-1,  1,  0, 1],
                                   [-1,  1,  0, 1],
                                   [-1,  0,  1, 1],
                                   [-1,  0,  1, 1],
                                   [ 0,  1, -1, 1],
                                   [ 0,  1, -1, 1],
                                   [ 1, -1,  0, 1],
                                   [ 1, -1,  0, 1]]) 
                    
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
