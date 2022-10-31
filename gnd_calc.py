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
micro, GrainIDs, X_spacing, Y_spacing, Z_spacing, featureData = pf.import_data()

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
GNDarraySR = np.zeros((indexmax, 1))
GNDarrayLR = np.zeros((indexmax, 1))
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
limit1 = np.int32(indexmax / 8)  # used to define the indices to use for each part of the computation
microTEMP = micro[:limit1, :3]
zOffset1 = 1 + micro[limit1, 0]
featIDsTEMP = featIDs[:, :, :zOffset1+1]
GAOTEMP = GAO[:, :, :, :, :zOffset1+1]
zOffset2 = 0
zOffset1 = 0

#  ---------------------------------------------------------------------
for index in range(limit1):
    GNDarraySR[index, 0], GNDarrayLR[index, 0], misoriArray[index, 0] ,GNDarraySS[index] = pf.GND(index, micro_max, featIDsTEMP, microTEMP, GAOTEMP, cs, indexmax, symOp, X_spacing, Y_spacing, Z_spacing, A, B, burgers, featureData, zOffset1, zOffset2)

# Progress update ---------------------------
print('\n\nProgress: 12.5%%\n')
print('[>>>---------------------]\n')

limit2 = np.int32(indexmax/4)

# Memory management -------------------------
#
microTEMP = micro[limit1:limit2, :3]
microTEMP = microTEMP / reduction

zOffset1 = micro[limit1, 0] / reduction - 1
if zOffset1 < 1:
    zOffset1 = 1

zOffset2 = micro[limit2, 0] / reduction + 1
featIDsTEMP = featIDs[:, :, zOffset1:zOffset2]
GAOTEMP = GAO[:, :, :, :, zOffset1:zOffset2]
# ---------------------------------------------------------------------
parfor index = limit1:limit2
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit1+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);  

# Progress update
print('\n\nProgress: 25.0%%\n');
print('[>>>>>>------------------]\n');

limit3 = np.int32(indexmax/2 - indexmax/8)
microTEMP = micro[limit2:limit3, :3]
microTEMP = microTEMP/reduction

zOffset1 = (tallMicro(limit2,1)/reduction)-1;
zOffset1 = gather(zOffset1);
zOffset2 = (tallMicro(limit3,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);

% %%% ---------------------------------------------------------------------
parfor index = limit2:limit3
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit2+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);   
end

% Progress update
clc
fprintf('\n\nProgress: 37.5%%\n');
fprintf('[>>>>>>>>>---------------]\n');
%
limit4 = int32(indexmax/2);
microTEMP = tallMicro(limit3:limit4,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

zOffset1 = (tallMicro(limit3,1)/reduction)-1;
zOffset1 = gather(zOffset1);
zOffset2 = (tallMicro(limit4,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);
% %%% ---------------------------------------------------------------------
parfor index = limit3:limit4
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit3+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);    
end

% Progress update
clc
fprintf('\n\nProgress: 50.0%%\n');
fprintf('[>>>>>>>>>>>>------------]\n');

limit5 = int32(indexmax/2 + indexmax/8);
microTEMP = tallMicro(limit4:limit5,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

zOffset1 = (tallMicro(limit4,1)/reduction)-1;
zOffset1 = gather(zOffset1);
zOffset2 = (tallMicro(limit5,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);
% %%% ---------------------------------------------------------------------
parfor index = limit4:limit5
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit4+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end

% Progress update
clc
fprintf('\n\nProgress: 62.5%%\n');
fprintf('[>>>>>>>>>>>>>>>---------]\n');

limit6 = int32(indexmax/2 + indexmax/4);
microTEMP = tallMicro(limit5:limit6,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

zOffset1 = (tallMicro(limit5,1)/reduction)-1;
zOffset1 = gather(zOffset1);
zOffset2 = (tallMicro(limit6,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);
% %%% ---------------------------------------------------------------------
parfor index = limit5:limit6
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit5+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update
clc
fprintf('\n\nProgress: 75.0%%\n');
fprintf('[>>>>>>>>>>>>>>>>>>------]\n');

limit7 = int32(indexmax - indexmax/8);
microTEMP = tallMicro(limit6:limit7,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

zOffset1 = (tallMicro(limit6,1)/reduction)-1;
zOffset1 = gather(zOffset1);
zOffset2 = (tallMicro(limit7,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);
% %%% ---------------------------------------------------------------------
parfor index = limit6:limit7
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit6+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update
clc
fprintf('\n\nProgress: 87.5%%\n');
fprintf('[>>>>>>>>>>>>>>>>>>>>>---]\n');

microTEMP = tallMicro(limit7:indexmax,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

zOffset1 = (tallMicro(limit7,1)/reduction)-1;
zOffset1 = gather(zOffset1);
zOffset2 = (tallMicro(indexmax,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);
% %%% ---------------------------------------------------------------------
parfor index = limit7:indexmax
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit7+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A,B,burgers,reduction,featureData,zOffset1,zOffset2);   
end

% Progress update
clc
fprintf('\n\nProgress: 100%%\n');
fprintf('[>>>>>>>>>>>>>>>>>>>>>>>>]\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------------------------ END OF MAIN LOOP -------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% resolve GND array into spatially resolved material points for
% visualization via .vtk output files

fprintf('\n\nSaving Data...\n\n');

GND_SR = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GND_LR = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GND_SS = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1,numSlip);

if cs == 3 && numSlip == 33
    GND_basal = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
    GND_pris = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);    
    GND_pyr = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
elseif cs == 2 && numSlip == 52
    GND_s = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
    GND_110 = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
    GND_112 = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
    GND_123 = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
end

microTEMP = tallMicro(:,:);
microTEMP = gather(microTEMP);
microTEMP(:,1:3) = microTEMP(:,1:3)/reduction;

grainIDsTEMP = tallGrainIDs(:,1);
grainIDsTEMP = gather(grainIDsTEMP);
grainIDsTEMP = int32(grainIDsTEMP);

for index = 1:indexmax
    x = microTEMP(index,3)+1; %setting temp x coordinate
    y = microTEMP(index,2)+1; %setting temp y coordinate
    z = microTEMP(index,1)+1; %setting temp z coordinate
    
    % locating spatially resolved GND density
    GND_SR(x,y,z) = GNDarraySR(index,1);
    
    % locating spatially resolved GND density
    GND_SS(x,y,z,:) = GNDarraySS(index,:);
    if cs == 3 && numSlip == 33
        GND_basal(x,y,z) = sum(GNDarraySS(index,1:6));
        GND_pris(x,y,z) = sum(GNDarraySS(index,1:3)) + sum(GNDarraySS(index,7:9));
        GND_pyr(x,y,z) = sum(GNDarraySS(index,10:33));
    elseif cs == 2 && numSlip == 52
        GND_s(x,y,z) = sum(GNDarraySS(index,1:4));
        GND_110(x,y,z) = sum(GNDarraySS(index,5:16));
        GND_112(x,y,z) = sum(GNDarraySS(index,17:28));
        GND_123(x,y,z) = sum(GNDarraySS(index,29:52));
    end
    
    % locating spatially resolved GND density
    %GND_LR(x,y,z) = GNDarrayLR(index,1);
    
    % locating spatially resolved misorientations
    misori(x,y,z) = misoriArray(index,1);
end

% clear out data which lacks spatial resolution
clear GNDarray misoriArray
%
%--------------------------------------------------------------------------