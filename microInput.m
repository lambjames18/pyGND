% microInput_clusteruse - script to input files
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFIED FOR CLUSTER USE. WILL SEARCH CURRENT WORKING DIRECTORY. %%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filenameData = 'CoNi_Feature1_zyx-eulersFZ_wholedata.csv';
filenameIDs = 'CoNi_Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'r5_3pct_TaFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'r5_3pct_TaFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'SpallTaFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'SpallTaFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'Spall_AM_TaFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Spall_AM_TaFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'Ti73pctFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Ti73pctFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'Ti71pctFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Ti71pctFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'AlNiCoFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'AlNiCoFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'Bicrystal2Feature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Bicrystal2Feature1_IDs_zyx-eulersFZ_wholedata.csv';

filenameFeatData = 'CoNi16_FZ_featuredata.csv';

inputMatrix = csvread(filenameData);
featureData = csvread(filenameFeatData);
GrainIDs = csvread(filenameIDs);

X_scaling = 1.5;
Y_scaling = 1.5;
Z_scaling = 1.0;

% if resolution of TriBeam data was adjusted
reduction = 1; 

X_spacing = X_scaling*(10^(-6))*reduction; % in meters
Y_spacing = Y_scaling*(10^(-6))*reduction;
Z_spacing = Z_scaling*(10^(-6))*reduction;

% adjust memory allocation of variables
Mpos = zeros(length(inputMatrix),3);
Mpos(1:length(inputMatrix),3) = int16(inputMatrix(1:length(inputMatrix),(3))/(X_scaling*reduction));
Mpos(1:length(inputMatrix),2) = int16(inputMatrix(1:length(inputMatrix),(2))/(Y_scaling*reduction));
Mpos(1:length(inputMatrix),1) = int16(inputMatrix(1:length(inputMatrix),(1))/(Z_scaling*reduction));
Meuler = inputMatrix(1:length(inputMatrix),4:6);

clear inputMatrix;

micro = [Mpos Meuler GrainIDs];

% tidy up arrays
clear Mpos Meuler GrainIDs