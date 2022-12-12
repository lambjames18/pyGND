% microInput_clusteruse - script to input files
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFIED FOR CLUSTER USE. WILL SEARCH CURRENT WORKING DIRECTORY. %%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%filenameData = '//192.168.0.102/general/MATLAB_common/GND_Matlab/CoNi_Feature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.102/general/MATLAB_common/GND_Matlab/CoNi_Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.201/data/MATLAB_common/GND_Matlab/3Feature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.201/data/MATLAB_common/GND_Matlab/3Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.201/data/MATLAB_common/GND_Matlab/r5_3pct_Feature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.201/data/MATLAB_common/GND_Matlab/r5_3pct_Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.201/data/MATLAB_common/GND_Matlab/r2_3pct_Feature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.201/data/MATLAB_common/GND_Matlab/r2_3pct_Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.201/data/MATLAB_common/GND_Matlab/SpallTaFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.201/data/MATLAB_common/GND_Matlab/SpallTaFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.201/data/MATLAB_common/GND_Matlab/Spall_AM_TaFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.201/data/MATLAB_common/GND_Matlab/Spall_AM_TaFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'Ti73pctFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Ti73pctFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'Ti71pctFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Ti71pctFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.201/data/MATLAB_common/GND_Matlab/AlNiCoFeature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = '//192.168.0.201/data/MATLAB_common/GND_Matlab/AlNiCoFeature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = '//192.168.0.102/general/MATLAB_common/GND_Matlab/316L-refined.csv';
%filenameData = 'Bicrystal2Feature1_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'Bicrystal2Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameIDs = 'synth16_Feature1_IDs_zyx-eulersFZ_wholedata.csv';
%filenameData = 'synth16_Feature1_zyx-eulersFZ_wholedata.csv';
filenameIDs = 'R2S10S5_GrainIDs.csv';
filenameData = 'R2S10S5_Data.csv';

%filenameFeatData = '//192.168.0.102/general/MATLAB_common/GND_Matlab/CoNi16_FZ_featuredata.csv';
%Directory = 'D:/IMMI Work/Final_output/1pct_r15_addtl_2pct/';
%filenameData = [Directory 'Ti7_1pct_160_Feature1_zyx-eulersFZ_wholedata.csv'];
%filenameIDs = [Directory 'Ti7_1pct_160_Feature1_IDs_zyx-eulersFZ_wholedata.csv'];


micro = datastore(filenameData);
%featureData = csvread(filenameFeatData);
featureData = 0;
GrainIDs = datastore(filenameIDs);
%GrainIDs = ;

X_scaling = 0.5;
Y_scaling = 0.5;
Z_scaling = 0.5;

% if resolution of TriBeam data was adjusted
reduction = 1; 

X_spacing = X_scaling*(10^(-6))*reduction; % in meters
Y_spacing = Y_scaling*(10^(-6))*reduction;
Z_spacing = Z_scaling*(10^(-6))*reduction;

%fullMicro = readall(micro);
%fullMicro = table2array(fullMicro);
tallMicro = tall(micro);
tallMicro = table2array(tallMicro);
%tallMicro(:,3) = int32(tallMicro(:,3)/X_scaling);
%tallMicro(:,2) = int32(tallMicro(:,2)/Y_scaling);
%tallMicro(:,1) = int32(tallMicro(:,1)/Z_scaling);
tallMicro(:,3) = int32(tallMicro(:,3));
tallMicro(:,2) = int32(tallMicro(:,2));
tallMicro(:,1) = int32(tallMicro(:,1));

%fullGrainIDs = readall(GrainIDs);
tallGrainIDs = tall(GrainIDs);
tallGrainIDs = table2array(tallGrainIDs);

%{
 adjust memory allocation of variables
Mpos = zeros(length(inputMatrix),3);
Mpos(1:length(inputMatrix),3) = int16(inputMatrix(1:length(inputMatrix),(3))/(X_scaling*reduction));
Mpos(1:length(inputMatrix),2) = int16(inputMatrix(1:length(inputMatrix),(2))/(Y_scaling*reduction));
Mpos(1:length(inputMatrix),1) = int16(inputMatrix(1:length(inputMatrix),(1))/(Z_scaling*reduction));
Meuler = inputMatrix(1:length(inputMatrix),4:6);

clear inputMatrix;

micro = [Mpos Meuler GrainIDs];

% tidy up arrays
clear Mpos Meuler GrainIDs
%}