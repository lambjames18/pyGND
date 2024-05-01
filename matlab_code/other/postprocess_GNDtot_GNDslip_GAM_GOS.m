% Script used to post-process featureID and miorientation relevant info
%{ID = 'test_Ti7_disori_full_1pct';
%ID = 'LOCAL_parallel_disori_test2_Ti7_1pct_r3';
% load data
misoriOUTfilename = [Directory ID 'Data_output_misori_.mat'];
load(misoriOUTfilename,'misori')
misoriOUTfilename = [Directory ID 'Data_output_featID_.mat'];
load(misoriOUTfilename,'featIDs')
misoriOUTfilename = [Directory ID 'Data_output_GAO_.mat'];
load(misoriOUTfilename,'GAO')
%}
cs = 3;
symmetry_operators;

% GAM - Grain Average Misorientation
% a measure of short range misorientation

% find the average the misorientation between voxels neighbors
% simply the average of variable misori

GAM = 180/pi*mean(misori,'all');

% GOS - Grain Orientation Spread
% a measure of long range misorientation

% find average misorientation between a grains voxels and the grain's
% average orientation

% find voxels that match a featureID
featIDmax = max(featIDs,[],'all');

GOS = zeros(featIDmax,1);
gA_voxel = zeros(3,3,featIDmax);

parfor featIDiter = 1:featIDmax
    % find all voxels associated with a feature
    featLoc = featIDs == featIDiter;
    
    % find mean orientation of a feature
    gFeatMean = mean(GAO(:,:,featLoc),[3 4 5]);
    % identify how many voxels in a feature
    numVoxelsInFeat = size(GAO(:,:,featLoc));
    
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
end

% Report GOS with distribution (histogram)

% Calc stats for GOS
m = mean(GOS);
med = median(GOS);
st = std(GOS);

% plot histogram of GOS with stats listed alongside GAM
GOS_histogram