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
elseif cs == 2 && numSlip == 52
    outfilename4 = [Directory ID '_GND_s.vtk'];
    vtkwrite(outfilename4, 'structured_points', 'GNDDensity', ...
    GND_s,'binary');

    outfilename5 = [Directory ID '_GND_110.vtk'];
    vtkwrite(outfilename5, 'structured_points', 'GNDDensity', ...
    GND_110,'binary');

    outfilename6 = [Directory ID '_GND_112.vtk'];
    vtkwrite(outfilename6, 'structured_points', 'GNDDensity', ...
    GND_112,'binary');

    outfilename6 = [Directory ID '_GND_123.vtk'];
    vtkwrite(outfilename6, 'structured_points', 'GNDDensity', ...
    GND_123,'binary');
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

if cs == 3 && numSlip == 33
    GNDslipOUTfilename = [Directory ID 'Data_output_GNDbasal_.mat'];
    save(GNDslipOUTfilename,'GND_basal', '-v7.3');
    GNDslipOUTfilename = [Directory ID 'Data_output_GNDpris_.mat'];
    save(GNDslipOUTfilename,'GND_pris', '-v7.3');
    GNDslipOUTfilename = [Directory ID 'Data_output_GNDpyr_.mat'];
    save(GNDslipOUTfilename,'GND_pyr', '-v7.3');

elseif cs == 2 && numSlip == 52
    GNDslipOUTfilename = [Directory ID 'Data_output_GND_s_.mat'];
    save(GNDslipOUTfilename,'GND_s', '-v7.3');
    GNDslipOUTfilename = [Directory ID 'Data_output_GND110_.mat'];
    save(GNDslipOUTfilename,'GND_110', '-v7.3');
    GNDslipOUTfilename = [Directory ID 'Data_output_GND112_.mat'];
    save(GNDslipOUTfilename,'GND_112', '-v7.3');
    GNDslipOUTfilename = [Directory ID 'Data_output_GND123_.mat'];
    save(GNDslipOUTfilename,'GND_123', '-v7.3');

end

misoriOUTfilename = [Directory ID 'Data_output_misori_.mat'];
save(misoriOUTfilename,'misori', '-v7.3');
featOUTfilename = [Directory ID 'Data_output_featID_.mat'];
save(featOUTfilename,'featIDs', '-v7.3');
GAOOUTfilename = [Directory ID 'Data_output_GAO_.mat'];
save(GAOOUTfilename,'GAO', '-v7.3');