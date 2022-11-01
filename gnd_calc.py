import numpy as np
import h5py

import py_functions as pf

directory = "./test/"
# Name output file
ID = "Test_Output"
# Get crystallography
burgers, A, numModes, numSlip, cs, B = pf.xtal()

# Convert Burgers to m
burgers = burgers * 1e-10

# Get symmetry operations from crystallography
symOp = pf.symmetry_operators(cs)

# Read data
h = h5py.File("D:/Research/R2_Sample10-Shot5/Data/3D/R2S10S5.dream3d")
spacing = np.squeeze(h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...])
featIDs = np.squeeze(h["DataContainers/ImageDataContainer/CellData/FeatureIds"][...])
euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/EulerAngles"][...])
# Get xyz
x1, x2, x3 = np.indices(featIDs.shape)
coordinates = np.stack((x1, x2, x3), axis=-1)

# Make microstructure
# micro, GrainIDs, spacing = pf.import_data()

# preallocate multidimensional arrays
dd = np.zeros(numSlip)

# create multidimensional arrays for Euler Angles and Feature IDs
# uses the maximum spacial dimensions to create the arrays
# phi1 = np.zeros((micro_max[0]+1, micro_max[1]+1, micro_max[2]+1))  # shape (x_length, y_length, z_length)
# Phi  = np.zeros((micro_max[0]+1, micro_max[1]+1, micro_max[2]+1))  # shape (x_length, y_length, z_length)
# phi2 = np.zeros((micro_max[0]+1, micro_max[1]+1, micro_max[2]+1))  # shape (x_length, y_length, z_length)
# featIDs = np.zeros((micro_max[0]+1, micro_max[1]+1, micro_max[2]+1))  # shape (x_length, y_length, z_length)
# misori = np.zeros((micro_max[0]+1, micro_max[1]+1, micro_max[2]+1))  # shape (x_length, y_length, z_length)
# GAO  = np.zeros((3, 3, micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (3, 3, x_length, y_length, z_length)
phi1 = euler[:, :, :, 0]
Phi = euler[:, :, :, 1]
phi2 = euler[:, :, :, 2]
misori  = np.zeros(featIDs.shape)

# Create orientation matrix for each point
print("Creating orientation matrices...")
GAO = pf.eu2om_multi(euler)

# create array for total GND density at each material point
GNDarraySR = np.zeros(featIDs.size)
GNDarrayLR = np.zeros(featIDs.size)
GNDarraySS = np.zeros((featIDs.size, numSlip))

# create array for avg misorientation at each material point
misoriArray = np.zeros(featIDs.size)

# uncomment line for GNDs across GB 
# RESULTS IN LOSS OF FEATURE DATA
# featIDs[:, :, :] = 1

###########################################################################
# ---------------------- START OF MAIN LOOP -------------------------------
# main loop to iterate over all microstructure points and determine -------
# misorientations ---------------------------------------------------------
###########################################################################

print("Converting arrays to 1D...")
# convert arrays to 1D
# featIDs = featIDs.reshape(-1)
# euler = euler.reshape(-1, 3)
# GAO = GAO.reshape(3, 3, -1)
# Indicate start of GND computation 
print('\n\nStarting parallel computations....\n\n')
# for index in range(euler.shape[0]):
for i in range(featIDs.shape[0]):
    for j in range(featIDs.shape[1]):
        for k in range(featIDs.shape[2]):
            point_coords = coordinates[i, j, k]
            GND_SR, misori, GND_SS = pf.GND(point_coords, featIDs, GAO, cs, symOp, spacing, A, B, burgers)
            print(GND_SR.shape, GND_SR)
            print(misori.shape, misori)
            print(GND_SS.shape, GND_SS)
            exit()




exit()

###########################################################################
# ------------------------ END OF MAIN LOOP -------------------------------
###########################################################################
#
# resolve GND array into spatially resolved material points for
# visualization via .vtk output files

print('\n\nSaving Data...\n\n')

GND_SR = np.zeros(micro_max[2] + 1, micro_max[1] + 1,micro_max[0] + 1)
GND_LR = np.zeros(micro_max[2] + 1, micro_max[1] + 1,micro_max[0] + 1)
GND_SS = np.zeros(micro_max[2] + 1, micro_max[1] + 1,micro_max[0] + 1, numSlip)

if cs == 3 & numSlip == 33:
    GND_basal = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)
    GND_pris = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)
    GND_pyr = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)
elif cs == 2 & numSlip == 52:
    GND_s = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)
    GND_110 = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)
    GND_112 = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)
    GND_123 = np.zeros(micro_max[2] + 1, micro_max[1] + 1, micro_max[0] + 1)

microTEMP = np.copy(micro)
grainIDsTEMP = np.int32(GrainIDs[:, 0])

for index in range(indexmax):
    x = microTEMP[index, 2] + 1  #setting temp x coordinate
    y = microTEMP[index, 1] + 1  #setting temp y coordinate
    z = microTEMP[index, 0] + 1  #setting temp z coordinate
    
    # locating spatially resolved GND density
    GND_SR[x, y, z] = GNDarraySR[index, 0]
    
    # locating spatially resolved GND density
    GND_SS[x, y, z, :] = GNDarraySS[index]
    if cs == 3 & numSlip == 33:
        GND_basal[x, y, z] = GNDarraySS[index].sum()
        GND_pris[x, y, z] = GNDarraySS[index, :3].sum() + GNDarraySS[index, 6:9].sum()
        GND_pyr[x, y, z] = GNDarraySS[index,9:33].sum()
    elif cs == 2 & numSlip == 52:
        GND_s[x, y, z] = GNDarraySS[index, :4].sum()
        GND_110[x, y, z] = GNDarraySS[index, 4:16].sum()
        GND_112[x, y, z] = GNDarraySS[index, 16:28].sum()
        GND_123[x, y, z] = GNDarraySS[index, 28:].sum()
    
    # locating spatially resolved misorientations
    misori[x, y, z] = misoriArray[index, 0]

#Save data for post-processing
GNDtotOUTfilename = directory + ID + 'Data_output_GND_SR.txt'
GNDtotOUTfilenameLR = directory + ID + 'Data_output_GND_LR.txt'
GNDslipOUTfilename = directory + ID + 'Data_output_GNDslip.txt'
np.savetxt(GNDtotOUTfilename, GND_SR, delimiter='\t', fmt='%1.5f')
np.savetxt(GNDtotOUTfilenameLR, GND_LR, delimiter='\t', fmt='%1.5f')
np.savetxt(GNDslipOUTfilename, GND_SS, delimiter='\t', fmt='%1.5f')

if cs == 3 & numSlip == 33:
    GNDslipOUTfilename = directory + ID + 'Data_output_GNDbasal_.txt'
    GNDslipOUTfilename = directory + ID + 'Data_output_GNDpris_.txt'
    GNDslipOUTfilename = directory + ID + 'Data_output_GNDpyr_.txt'
    np.savetxt(GNDslipOUTfilename, GND_basal, delimiter='\t', fmt='%1.5f')
    np.savetxt(GNDslipOUTfilename, GND_pris, delimiter='\t', fmt='%1.5f')
    np.savetxt(GNDslipOUTfilename, GND_pyr, delimiter='\t', fmt='%1.5f')

elif cs == 2 & numSlip == 52:
    GNDslipOUTfilename = directory + ID + 'Data_output_GND_s_.txt'
    GNDslipOUTfilename = directory + ID + 'Data_output_GND110_.txt'
    GNDslipOUTfilename = directory + ID + 'Data_output_GND112_.txt'
    GNDslipOUTfilename = directory + ID + 'Data_output_GND123_.txt'
    np.savetxt(GNDslipOUTfilename, GND_s, delimiter='\t', fmt='%1.5f')
    np.savetxt(GNDslipOUTfilename, GND_110, delimiter='\t', fmt='%1.5f')
    np.savetxt(GNDslipOUTfilename, GND_112, delimiter='\t', fmt='%1.5f')
    np.savetxt(GNDslipOUTfilename, GND_123, delimiter='\t', fmt='%1.5f')


misoriOUTfilename = directory + ID + 'Data_output_misori_.txt'
featOUTfilename = directory + ID + 'Data_output_featID_.txt'
GAOOUTfilename = directory + ID + 'Data_output_GAO_.txt'
np.savetxt(misoriOUTfilename, misori, delimiter='\t', fmt='%1.5f')
np.savetxt(featOUTfilename, featIDs, delimiter='\t', fmt='%1.5f')
np.savetxt(GAOOUTfilename, GAO, delimiter='\t', fmt='%1.5f')

print('\n\nCalculation Complete\n\n')

# barchart_ssGND