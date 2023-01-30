clear

Directory = 'D:\\Research\\scripts\\TriBeam_GND\\';
fprintf('File location: %s \n',Directory)
ID = 'TaAMSpalled_mini_';
fprintf('File ID: %s \n',ID)

% prompt user for crystallography
xtal_cluster_Ta

% convert burgers vector to m
burgers = burgers*1E-10;

% determine symmetry operators from xtal
symmetry_operators

% tell user data is being imported
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
fprintf('\n\nInitializing Multidimensional Arrays...\n');

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
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

limit1 = int32(indexmax/8);
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
index = 1;
[GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
    calcGND(index,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
    X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);
return

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\n\nStarting parallel computations....\n\n');

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

% %%% ---------------------------------------------------------------------
parfor index = limit1:limit2
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
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

% %%% ---------------------------------------------------------------------
parfor index = limit2:limit3
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
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
%%
%
% %%% ---------------------------------------------------------------------
parfor index = limit3:limit4
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
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

% %%% ---------------------------------------------------------------------
parfor index = limit4:limit5
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
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

% %%% ---------------------------------------------------------------------
parfor index = limit5:limit6
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
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

% %%% ---------------------------------------------------------------------
parfor index = limit6:limit7
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
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

% %%% ---------------------------------------------------------------------
parfor index = limit7:indexmax
    [GNDarraySR(index,1),~,misoriArray(index,1),GNDarraySS(index,:)] = ...
        calcGND(index-limit7+1,microMax,featIDsTEMP,microTEMP,GAOTEMP,cs,indexmax,symOp,...
        X_spacing,Y_spacing,Z_spacing,A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2);   
end

% Progress update
clc
featOUTfilename = ['GNDarraySS_8_' ID '.mat'];
save(featOUTfilename,'GNDarraySS', '-v7.3')
fprintf('\n\nProgress: 100%%\n');
fprintf('[>>>>>>>>>>>>>>>>>>>>>>>>]\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
saveData_VtkMat

fprintf('\n\nCalculation Complete\n\n');

barchart_ssGND
