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

initVar

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
GNDarraySS = zeros(limit1,numSlip);
zOffset2 = 0;
zOffset1 = 0;
%
% %%% ---------------------------------------------------------------------
parfor index = 1:limit1
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update ---------------------------
clc
featOUTfilename = ['GNDarraySS_1_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
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
limit1_ss = limit1-1;

% %%% ---------------------------------------------------------------------
parfor index = limit1:limit2
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-limit1_ss,:)] = ...
        calcGND(index-limit1+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update
clc
featOUTfilename = ['GNDarraySS_2_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
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
limit2_ss = limit2-1;

% %%% ---------------------------------------------------------------------
parfor index = limit2:limit3
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-limit2_ss,:)] = ...
        calcGND(index-limit2+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);   
end

% Progress update
clc
featOUTfilename = ['GNDarraySS_3_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
%
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
index3_ss = index3-1;
%%
%
% %%% ---------------------------------------------------------------------
parfor index = limit3:limit4
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-index3_ss,:)] = ...
        calcGND(index-limit3+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);    
end

% Progress update
clc
featOUTfilename = ['GNDarraySS_4_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
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
index4_ss = index4-1;

% %%% ---------------------------------------------------------------------
parfor index = limit4:limit5
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-index4_ss,:)] = ...
        calcGND(index-limit4+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end

% Progress update
clc
featOUTfilename = ['GNDarraySS_5_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
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
index5_ss = index5-1;

% %%% ---------------------------------------------------------------------
parfor index = limit5:limit6
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-index5_ss,:)] = ...
        calcGND(index-limit5+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update
clc
featOUTfilename = ['GNDarraySS_6_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
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
index6_ss = index6-1;

% %%% ---------------------------------------------------------------------
parfor index = limit6:limit7
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-index6_ss,:)] = ...
        calcGND(index-limit6+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);  
end
%
% Progress update
clc
featOUTfilename = ['GNDarraySS_7_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
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
index7_ss = index7-1;

% %%% ---------------------------------------------------------------------
parfor index = limit7:indexmax
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index-index7_ss,:)] = ...
        calcGND(index-limit7+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);   
end

% Progress update
clc
featOUTfilename = ['GNDarraySS_8_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
fprintf('\n\nProgress: 100%%\n');
fprintf('[>>>>>>>>>>>>>>>>>>>>>>>>]\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ------------------------ END OF MAIN LOOP -------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

saveData_VtkMat

%
% indicate completion of script
fprintf('\n\nCalculation Complete\n\n');

barchart_ssGND
%--------------------------------------------------------------------------