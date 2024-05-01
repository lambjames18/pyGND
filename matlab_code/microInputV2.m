% microInput_clusteruse - script to input files
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MODIFIED FOR CLUSTER USE. WILL SEARCH CURRENT WORKING DIRECTORY. %%%%%%%%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

filenameIDs = 'D:\\Research\\scripts\\TriBeam_GND\\matlab_code\\TaAMSpalled_mini_MatlabInput-FeatureIDs.csv';
filenameData = 'D:\\Research\\scripts\\TriBeam_GND\\matlab_code\\TaAMSpalled_mini_MatlabInput-Microstructure.csv';



micro = datastore(filenameData);
%featureData = csvread(filenameFeatData);
featureData = 0;
GrainIDs = datastore(filenameIDs);
%GrainIDs = ;

X_scaling = 1.5;
Y_scaling = 1.5;
Z_scaling = 1.5;

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