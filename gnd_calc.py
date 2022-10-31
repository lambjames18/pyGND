import numpy as np

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
micro, GrainIDs, spacing, featureData = pf.import_data()

micro_max = np.amax(micro, axis=0)  # provides the maximum x, y, z values
micro_min = np.amin(micro, axis=0)  # provides the minimum x, y, z values
indexmax = micro.shape[0]  # provides the number of entries

# preallocate multidimensional arrays
dd = np.zeros((numSlip,1))

# create multidimensional arrays for Euler Angles and Feature IDs
# uses the maximum spacial dimensions to create the arrays
phi1 = np.zeros((micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (x_length, y_length, z_length)
Phi  = np.zeros((micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (x_length, y_length, z_length)
phi2 = np.zeros((micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (x_length, y_length, z_length)
featIDs = np.zeros((micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (x_length, y_length, z_length)
misori  = np.zeros((micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (x_length, y_length, z_length)
# Create orientation matrix for each point
GAO  = np.zeros((3, 3, micro_max[2]+1, micro_max[1]+1, micro_max[0]+1))  # shape (3, 3, x_length, y_length, z_length)

# create array for total GND density at each material point
GNDarraySR = np.zeros(indexmax)
GNDarrayLR = np.zeros(indexmax)
GNDarraySS = np.zeros((indexmax, numSlip))

# create array for avg misorientation at each material point
misoriArray = np.zeros((indexmax,1))

microTEMP = np.copy(micro)
microTEMP[:, :3] = microTEMP[:, :3]

grainIDsTEMP = GrainIDs[:, 0]
grainIDsTEMP = np.int32(grainIDsTEMP)

#create 3D matrices with associated Euler angles and featureIDs
for index in range(indexmax):
    x = microTEMP[index, 2] + 1  #setting temp x coordinate
    y = microTEMP[index, 1] + 1  #setting temp y coordinate
    z = microTEMP[index, 0] + 1  #setting temp z coordinate
    phi1[x, y, z] = microTEMP[index,3]  #first Euler angle for 3D coordinate
    Phi[x, y, z] = microTEMP[index,4]
    phi2[x, y, z] = microTEMP[index,5]
    
    # convert to orientation matrices
    gA = pf.eu2om_mod(np.array([phi1[x,y,z], Phi[x,y,z], phi2[x,y,z]]))
    
    # store orientation of voxel
    GAO[:, :, x, y, z] = gA
    featIDs[x, y, z] = grainIDsTEMP[index, 0]
    
    # uncomment line for GNDs across GB 
    # RESULTS IN LOSS OF FEATURE DATA
    #featIDs(x,y,z) = 1;



###########################################################################
# ---------------------- START OF MAIN LOOP -------------------------------
# main loop to iterate over all microstructure points and determine -------
# misorientations ---------------------------------------------------------
###########################################################################

# make sure workstation is ready for parallel processing. Determine number-
# of workers needed for calculation. Prallalelization is only via parfor --
# loops. Multiple parfor loops occur sequentially to report progress w/o --
# impacting performance of parallel processing. ---------------------------
#
# Indicate start of GND computation 
print('\n\nStarting parallel computations....\n\n')
for index in range(micro.shape[0]):
    GNDarraySR, GNDarrayLR, misoriArray, GNDarraySS = pf.GND(index,
                                                             micro_max,
                                                             featIDs,
                                                             micro,
                                                             GAO,
                                                             cs,
                                                             indexmax,
                                                             symOp,
                                                             spacing,
                                                             A,
                                                             B,
                                                             burgers,
                                                             featureData)



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