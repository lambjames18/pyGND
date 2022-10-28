% Script used to post-process featureID and miorientation relevant info
%{ID = 'test_Ti7_disori_full_1pct';
%ID = 'LOCAL_parallel_disori_test2_Ti7_1pct_r3';
% load data

%misoriOUTfilename = ['//192.168.0.102/general/Ti7Al_Data/' ID 'Data_output_featID_.mat'];
%load(misoriOUTfilename,'featIDs')

%}
cs = 3;
symmetry_operators;

% find voxels that match a featureID
featIDmax = max(featIDs,[],'all');

GOS = zeros(featIDmax,1);
gA_voxel = zeros(3,3,featIDmax);
numSlip = 33;
%total mean median 99 97 95 mode1 mode2 mode3 numSlip
GNDtot = sum(GND_SS,4);
GNDstats = zeros(featIDmax,2);

for featIDiter = 1:featIDmax
    % find all voxels associated with a feature
    featLoc = featIDs == featIDiter;
    
    % find mean orientation of a feature
    %gFeatMean = mean(GAO(:,:,featLoc),[3 4 5]);
    % identify how many voxels in a feature
    GNDstats(featIDiter,1) = mean(squeeze(nonzeros(GND_SS(:,:,:,1).*featLoc)));
    GNDstats(featIDiter,2) = mean(squeeze(nonzeros(GNDtot.*featLoc)));
    %{
    % calculate over all voxels in grain, ignore if grain has 1 voxel
    if (sum(numVoxelsInFeat) > 6)
        % find orientations of voxels
        gA_voxel = GAO(:,:,featLoc);

        for VoxelIter = 1:numVoxelsInFeat(1,3)
            GOS(featIDiter,1) = GOS(featIDiter,1) +...
                (abs(deltathetakV4(gA_voxel(:,:,VoxelIter),gFeatMean,1,symOp))+...
                abs(deltathetakV4(gA_voxel(:,:,VoxelIter),gFeatMean,2,symOp))+...
                abs(deltathetakV4(gA_voxel(:,:,VoxelIter),gFeatMean,3,symOp)))/3;
        end

        GOS(featIDiter,1) = 180/pi*GOS(featIDiter,1)/numVoxelsInFeat(1,3);
    
    end
    
    % indicate progress of calculations
    if (mod(featIDiter,10) == 0 || featIDiter == featIDmax)
        Completion = 100*featIDiter/featIDmax;
        formatSpec = 'Completion%%: %8.1f \n';
        fprintf(formatSpec,Completion)
    end
    %}
    clc
    completion = 100*featIDiter/featIDmax;
    fprintf('\nProgress: %5.0f',completion);
end
