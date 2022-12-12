function [totalGNDdensitySR,totalGNDdensityLR,avgMisori,ddSR] = calcGND(index,microMax,featIDs,...
    micro,GAO,cs,indexmax,symOp,X_spacing,Y_spacing,Z_spacing,...
    A_sparse,B,burgers,reduction,featureData,zOffset1,zOffset2)

x = micro(index,3)+1; %setting x coordinate
y = micro(index,2)+1; %setting y coordinate
z = micro(index,1)+1-zOffset1; %setting z coordinate
% no calculations if inside void or outside microstructure
if (GAO(:,:,x,y,z) ~= zeros(3))

    [XenvCompleteness,YenvCompleteness,ZenvCompleteness] = ...
        determine_neighborhood(microMax,featIDs,x,y,z);

    % Determine Disorientation between material points and neighbors, 
    % influenced by neighborhood -- Calculate kappa for material points              
    [dthe,diffOperatorX,diffOperatorY,diffOperatorZ] = determine_dthe(...
        XenvCompleteness,YenvCompleteness,ZenvCompleteness,GAO,x,y,z,symOp);

    % Calculate average misorientation from dthe
    % mean(abs(dthe),'all') appears too recent for use with 2018a on cluster
    % avgMisori = mean(abs(dthe),'all');
    avgMisori = (abs(dthe(1,1))+abs(dthe(1,2))+abs(dthe(1,3))+...
        abs(dthe(2,1))+abs(dthe(2,2))+abs(dthe(2,3))+...
        abs(dthe(3,1))+abs(dthe(3,2))+abs(dthe(3,3)))/9;

    kappaSR = determine_kappaV5(dthe,diffOperatorX,...
        diffOperatorY,diffOperatorZ,X_spacing,Y_spacing,Z_spacing);
    
    % Convert Kappa to crystal coordinates since dislocations are
    % described in crystal coordinates
    kappaSRprime = transpose(GAO(:,:,x,y,z))*kappaSR*GAO(:,:,x,y,z);
    %kappaSRprime = kappaSR;
    
    % Calculate Nye Tensor (alpha) from curvature kappa  
    alphaSR = transpose(kappaSRprime) - trace(kappaSRprime);

    %alphaSRprime = GAO(:,:,x,y,z)*alphaSR*transpose(GAO(:,:,x,y,z));
    %function used to determine a total value of gnd density at one particular
    %material point

    % determine dislocation densities (dd -> rho) from misorientations
    ddSR = L2_SparseV2(alphaSR,cs,A_sparse,B,burgers);

    % determine total gnd density to be sum of dislocation density across all
    % slip systems
    totalGNDdensitySR = sum(abs(ddSR));
    ddSR = transpose(abs(ddSR));
    
    % repeat misorientation calculations using LR approach
    %{
    % determine distance between voxel and grain centroid
    X_spacingLR = abs(X_spacing*reduction*microMax(1,3) +...
        featureData(featIDs(x,y,z),4) - (X_spacing*x*reduction));
    Y_spacingLR = abs(Y_spacing*reduction*microMax(1,2) +...
        featureData(featIDs(x,y,z),5) - (Y_spacing*y*reduction));
    Z_spacingLR = abs(Z_spacing*reduction*microMax(1,1) +...
        featureData(featIDs(x,y,z),6) - (Z_spacing*z*reduction));
    LR_spacing = sqrt(X_spacingLR^2 + Y_spacingLR^2 + Z_spacingLR^2);
    
    gB = eu2om([featureData(featIDs(x,y,z),1) ...
        featureData(featIDs(x,y,z),2) featureData(featIDs(x,y,z),3)]);
    
    % calc specific miorientation angles for kappa calc
    dtheLR(1,1) = deltathetakV4(gB,GAO(:,:,x,y,z),1,symOp);
    dtheLR(2,1) = deltathetakV4(gB,GAO(:,:,x,y,z),2,symOp);
    dtheLR(3,1) = deltathetakV4(gB,GAO(:,:,x,y,z),3,symOp);
    dtheLR(1,2) = dtheLR(1,1);
    dtheLR(2,2) = dtheLR(2,1);
    dtheLR(3,2) = dtheLR(3,1);
    dtheLR(1,3) = dtheLR(1,1);
    dtheLR(2,3) = dtheLR(2,1);
    dtheLR(3,3) = dtheLR(3,1);
    
    kappaLR = determine_kappaV5(dtheLR,1,...
        1,1,LR_spacing,LR_spacing,LR_spacing);
    
    % Calculate Nye Tensor (alpha) from curvature kappa  
    alphaLR = transpose(kappaLR) - trace(kappaLR);
    
    % Convert Nye Tensor to crystal coordinates since dislocations are
    % described in crystal coordinates
    alphaLRPrime = GAO(:,:,x,y,z)*alphaLR*transpose(GAO(:,:,x,y,z));
    
    % determine dislocation densities (dd -> rho) from LR misorientations
    ddLR = L2_SparseV2(alphaLRPrime,cs,A_sparse,B,burgers);

    % determine total gnd density to be sum of dislocation density across 
    % all slip systems
    totalGNDdensityLR = sum(abs(ddLR));
    %}
    totalGNDdensityLR = 0;
else
    % tame output for voxels where misorientation can't be calc
    avgMisori = 0;
    totalGNDdensitySR = 0;
    totalGNDdensityLR = 0;
    [~,ddSR_dim] = size(A_sparse);
    ddSR = zeros(1,ddSR_dim);
    %fprintf('\nGAO is zeros!\n')
end