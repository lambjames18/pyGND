% ************************************************************************%
% ------------------------------------------------------------------------%
% Crystallographic Geometrically Necessary Dislocation Density Calculator %
% ------------------ Wyatt A. Witzen, April 22, 2022 ---------------------%
% ------------------------------------------------------------------------%
% *********************************************************************** %
%                                                                         %
% This script will calculate crystallographic GND densities via L2 -------%
% minimization and resolve them upon specified slip systems for each -----%
% voxel in a 3D orientation dataset. Three .vtk files are produced for ---%
% visualization of the misorientation, total GND density, and featureID --%
% mapping. A separate post-processing script is necessary for extraction -%
% of feature-specific GND statistics, along with GAM and GOS calculations.%
% This post processing script will use load the GAO, misori, and GNDtotal-%
% variables as saved in the separate .mat files. -------------------------%
%                                                                         %
% Nearest neighbors are based on Von Neumann environments and calculate --% 
% derivatives based upon a central difference formulation. ---------------%
% This has been extended to account for both long range and short range --%
% GND densities, where short range are determined via neighboring material%
% points and long rang densities are determined via the finite differences%
% between a material point and a grains centroid.
%                                                                         %
% MODIFIED FOR CLUSTER USE. RUNS PARALLEL AND EXPECTS SUFFICIENT MEMORY. -%
% WILL NOT PROMPT USER FOR INPUT PARAMETERS. SPECIFY IN INPUT FILE. ------%
% COMPILE BY CALLING THIS SCRIPT IN A SEPARATE FUNCTION .m FILE. ---------%
%                                                                         %
% Create .csv file with z y x phi1 phi phi2 and separate grainID .csv ----%
% file and include in working directory. Label as needed and properly set %
% filename variables in micro_input file. --------------------------------%
% ------------------------------------------------------------------------%
% ************************************************************************%

clear

%Directory = 'G:/AlNiCo_Work/';
Directory = '//orthanc/data/Spall_AM_Ta_Work/';
fprintf('File location: %s \n',Directory)

% Name output files
%ID = '_Final_1pct_full_';
%ID = 'SS_test3_Ti7_3pct_full';
%ID = 'AlNiCo_disori_test2';
%ID = 'Ti7_SStest_r15_1pct_step160';
%ID = 'LOCAL_parallel_test5_AM_Ta_r15';
%ID = 'LOCAL_parallel_test1_Spall_Ta_';
ID = 'Final_Spall_Ta_AM_';
%ID = 'LOCAL_parallel_test1_r3_SRLR_1pct';
%ID = 'LOCAL_parallel_test1_CoNi_SS';
%ID = 'test_Bicrystal2';
%ID = 'synth_test_';
%ID = 'SS_GND_test';
fprintf('File ID: %s \n',ID)

% prompt user for crystallography
%xtal_cluster_HCP
%xtal_cluster_BCC
%xtal_cluster_FCC
xtal_cluster

% convert burgers vector to m
burgers = burgers*1E-10;

% determine symmetry operators from xtal
symmetry_operators

% toggle misorientation vs disorientation
%symOp = symOp(:,:,1);

% tell user data is being imported
clc
fprintf('\nImporting Data...\n\n');
% select data file and import data
microInputV2
tallMicroMax = max(tallMicro,[],1);
tallMicroMin = min(tallMicro,[],1);
indexmax = size(tallMicro,1);

% extract 3D coordinates of voxels and associated Euler angles
microMax = gather(tallMicroMax);
microMax = microMax/reduction;
microMin = gather(tallMicroMin);
% determine number of voxels in dataset
indexmax = gather(indexmax);

% preallocate multidimensional arrays -------------------------------------
dd = zeros(numSlip,1);
% initialize GND densities with value associated with "annealed"
% microstructures (zero dislocation density is not intuitive)
GND_SR = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GND_LR = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GND_SS = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1,numSlip);
GND_basal = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GND_pris = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GND_pyr = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
%GNDtotal(:,:,:) = 1E10;
%GNDmode = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1,numModes);
%GNDtotal(:,:,:,:) = 1E10;
%GNDslip = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1,numSlip);
%GNDslip(:,:,:,:) = 1E10;

% create multidimensional arrays for Euler Angles and Feature IDs
phi1 = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
Phi = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
phi2 = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
featIDs = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
misori = zeros(microMax(3)+1,microMax(2)+1,microMax(1)+1);
GAO = zeros(3,3,microMax(3)+1,microMax(2)+1,microMax(1)+1);

% create array for total GND density at each material point
GNDarraySR = zeros(indexmax,1);
GNDarrayLR = zeros(indexmax,1);
GNDarraySS = zeros(indexmax,numSlip);
% create array for avg misorientation at each material point
misoriArray = zeros(indexmax,1);

% indicate status
formatSpec = '\n\nInitializing Multidimensional Arrays...\n';

microTEMP = tallMicro(:,:);
microTEMP = gather(microTEMP);
microTEMP(:,1:3) = microTEMP(:,1:3)/reduction;

grainIDsTEMP = tallGrainIDs(:,1);
grainIDsTEMP = gather(grainIDsTEMP);
grainIDsTEMP = int32(grainIDsTEMP);

%create 3D matrices with associated Euler angles and featureIDs
for index = 1:indexmax
    x = microTEMP(index,3)+1; %setting temp x coordinate
    y = microTEMP(index,2)+1; %setting temp y coordinate
    z = microTEMP(index,1)+1; %setting temp z coordinate
    phi1(x,y,z) = microTEMP(index,4); %first Euler angle for 3D coordinate
    Phi(x,y,z) = microTEMP(index,5);
    phi2(x,y,z) = microTEMP(index,6);
    
    % convert to orientation matrices
    gA = eu2om_mod([phi1(x,y,z),Phi(x,y,z),phi2(x,y,z)]);
    
    % store orientation of voxel
    GAO(:,:,x,y,z) = gA;
    featIDs(x,y,z) = grainIDsTEMP(index,1);
    
    % uncomment line for GNDs across GB 
    % RESULTS IN LOSS OF FEATURE DATA
    %featIDs(x,y,z) = 1;
end

% clear redudant variables
%clear phi1 Phi phi2 microTEMP
% -------------------------------------------------------------------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ---------------------- START OF MAIN LOOP -------------------------------
% main loop to iterate over all microstructure points and determine -------
% misorientations ---------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% make sure workstation is ready for parallel processing. Determine number-
% of workers needed for calculation. Prallalelization is only via parfor --
% loops. Multiple parfor loops occur sequentially to report progress w/o --
% impacting performance of parallel processing. ---------------------------
%
% Indicate start of GND computation 
clc
fprintf('\n\nStarting parallel computations....\n\n');
%
limit1 = int32(indexmax/8);
%
microTEMP = tallMicro(1:limit1,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;
zOffset1 = (tallMicro(limit1,1)/reduction)+1;
zOffset1 = gather(zOffset1);
featIDsTEMP = featIDs(:,:,1:zOffset1+1);
GAOTEMP = GAO(:,:,:,:,1:zOffset1+1);
zOffset2 = 0;
zOffset1 = 0;
%
% %%% ---------------------------------------------------------------------
parfor index = 1:limit1
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update ---------------------------
clc
fprintf('\n\nProgress: 12.5%%\n');
fprintf('[>>>---------------------]\n');
%
limit2 = int32(indexmax/4);

% Memory management -------------------------
%
microTEMP = tallMicro(limit1:limit2,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

zOffset1 = (tallMicro(limit1,1)/reduction)-1;
zOffset1 = gather(zOffset1);
if zOffset1 < 1
    zOffset1 = 1;
end
zOffset2 = (tallMicro(limit2,1)/reduction)+1;
zOffset2 = gather(zOffset2);
featIDsTEMP = featIDs(:,:,zOffset1:zOffset2);
GAOTEMP = GAO(:,:,:,:,zOffset1:zOffset2);
% %%% ---------------------------------------------------------------------
parfor index = limit1:limit2
    [GNDarraySR(index,1),GNDarrayLR(index,1),misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit1+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update
clc
fprintf('\n\nProgress: 25.0%%\n');
fprintf('[>>>>>>------------------]\n');

limit3 = int32(indexmax/2 - indexmax/8);
microTEMP = tallMicro(limit2:limit3,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;

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
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);   
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
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);    
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
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
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
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
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
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
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
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);   
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
%Output Calculations as .mat and .vtk
%
outfilename = [Directory ID '_misorientation_mapping.vtk'];
vtkwrite(outfilename, 'structured_points', 'Misorientation',misori,...
    'binary');
outfilename2 = [Directory ID '_FeatureIDs.vtk'];
vtkwrite(outfilename2, 'structured_points', 'GrainID', featIDs,...
    'binary');
%
outfilename3 = [Directory ID '_GND_SR.vtk'];
vtkwrite(outfilename3, 'structured_points', 'GNDDensity', ...
    GND_SR,'binary');

if cs == 3 && numSlip == 33
    outfilename4 = [Directory ID '_GND_basal.vtk'];
    vtkwrite(outfilename4, 'structured_points', 'GNDDensity', ...
        GND_basal,'binary');

    outfilename5 = [Directory ID '_GND_pris.vtk'];
    vtkwrite(outfilename5, 'structured_points', 'GNDDensity', ...
        GND_pris,'binary');

    outfilename6 = [Directory ID '_GND_pyr.vtk'];
    vtkwrite(outfilename6, 'structured_points', 'GNDDensity', ...
        GND_pyr,'binary');
end
%{
outfilename4 = ['W:/Data/Ti7_Data/' ID '_GND_LR.vtk'];
vtkwrite(outfilename4, 'structured_points', 'GNDDensity', ...
    GND_LR,'binary');

%}
fileType = '.vtk';
for index_ss = 1:numSlip
    outfilename_ss = sprintf('%s%s_GND_SS_%.15g%s',Directory,ID,index_ss,fileType);
    dataLabel_ss = sprintf('GNDDensity%.15g',index_ss);
    vtkwrite(outfilename_ss, 'structured_points', dataLabel_ss, ...
        GND_SS(:,:,:,index_ss),'binary');
end
%--------------------------------------------------------------------------
%
%Save data for post-processing
GNDtotOUTfilename = [Directory ID 'Data_output_GND_SR_.mat'];
save(GNDtotOUTfilename,'GND_SR', '-v7.3');
GNDtotOUTfilenameLR = [Directory ID 'Data_output_GND_LR_.mat'];
save(GNDtotOUTfilename,'GND_LR', '-v7.3');
GNDslipOUTfilename = [Directory ID 'Data_output_GNDslip_.mat'];
save(GNDslipOUTfilename,'GND_SS', '-v7.3');
GNDslipOUTfilename = [Directory ID 'Data_output_GNDbasal_.mat'];
save(GNDslipOUTfilename,'GND_basal', '-v7.3');
GNDslipOUTfilename = [Directory ID 'Data_output_GNDpris_.mat'];
save(GNDslipOUTfilename,'GND_pris', '-v7.3');
GNDslipOUTfilename = [Directory ID 'Data_output_GNDpyr_.mat'];
save(GNDslipOUTfilename,'GND_pyr', '-v7.3');
misoriOUTfilename = [Directory ID 'Data_output_misori_.mat'];
save(misoriOUTfilename,'misori', '-v7.3');
featOUTfilename = [Directory ID 'Data_output_featID_.mat'];
save(featOUTfilename,'featIDs', '-v7.3');
GAOOUTfilename = [Directory ID 'Data_output_GAO_.mat'];
save(GAOOUTfilename,'GAO', '-v7.3');
%
% indicate completion of script
fprintf('\n\nCalculation Complete\n\n');

barchart_ssGND
%--------------------------------------------------------------------------