import numpy as np
import h5py
import mpire

import py_functions as pf
import GND

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
spacing = np.squeeze(h["DataContainers/ImageDataContainer/_SIMPL_GEOMETRY/SPACING"][...]) * 1e-6
featIDs = np.squeeze(h["DataContainers/ImageDataContainer/CellData/FeatureIds"][...])
euler = np.squeeze(h["DataContainers/ImageDataContainer/CellData/EulerAngles"][...])
# Get xyz
x1, x2, x3 = np.indices(featIDs.shape)
coordinates = np.stack((x1, x2, x3), axis=-1)
coordinates = coordinates.reshape(-1, 3)

# Create orientation matrix for each point
print("Creating orientation matrices...")
GAO = pf.eu2om_multi(euler)

# uncomment line for GNDs across GB 
# RESULTS IN LOSS OF FEATURE DATA
# featIDs[:, :, :] = 1

# Create empty arrays to store results
GND_SR = np.zeros(featIDs.size, dtype=float)
GND_SS = np.zeros((featIDs.size, numSlip), dtype=float)
misori = np.zeros(featIDs.size, dtype=float)


###########################################################################
# ---------------------- START OF MAIN LOOP -------------------------------
# main loop to iterate over all microstructure points and determine -------
# misorientations ---------------------------------------------------------
###########################################################################

# Indicate start of GND computation 
print('\n\nStarting parallel computations....\n\n')


gnd = GND.GND(cs, burgers / 1e-10)
gnd.set_data(coordinates, euler, featIDs, spacing)

exit()
for i in track(range(coordinates.shape[0]), "Working on GNDs..."):
    point_coords = coordinates[i]
    GND_SR[i], misori[i], GND_SS[i] = pf.GND(point_coords, featIDs, GAO, cs, symOp, spacing, A, B, burgers)

exit()
print("Complete")
np.save(directory + ID + "_GND_SR.npy", GND_SR)
np.save(directory + ID + "_GND_SS.npy", GND_SS)
np.save(directory + ID + "_misori.npy", misori)

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