clear

Directory = '//orthanc/data/Spall_AM_Ta_Work/';
fprintf('File location: %s \n',Directory)

% Name output files
%ID = '_Final_1pct_full_';
%ID = 'SS_test3_Ti7_3pct_full';
%ID = 'AlNiCo_disori_test2';
%ID = 'Ti7_SStest_r15_1pct_step160';
%ID = 'LOCAL_parallel_test5_AM_Ta_r15';
ID = 'AM_Spall_Ta__FINAL_';
%ID = 'LOCAL_parallel_test1_Spall_Ta_AM';
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
xtal_cluster_Ta

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
%GNDarrayLR = zeros(indexmax,1);
%GNDarraySS = zeros(indexmax,numSlip);
%GNDarraySS = zeros(indexmax,numSlip);
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