clear

Directory = 'D:\\Research\\scripts\\TriBeam_GND\\';
fprintf('File location: %s \n',Directory)
ID = 'R2S10S5_';
fprintf('File ID: %s \n',ID)

% prompt user for crystallography
xtal_cluster

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
fprintf('\n\nStarting parallel computations....\n\n');
limit1 = 100000;
%
microTEMP = tallMicro(1:limit1,1:3);
microTEMP = gather(microTEMP);
microTEMP = microTEMP/reduction;
zOffset1 = (tallMicro(limit1,1)/reduction)+1;
zOffset1 = gather(zOffset1);
featIDsTEMP = featIDs(:,:,1:zOffset1+1);
GAOTEMP = GAO(:,:,:,:,1:zOffset1+1);
zOffset2 = 0;
zOffset1 = 0;

for index = 90000:90001
    x = microTEMP(index,3)+1; %setting x coordinate
    y = microTEMP(index,2)+1; %setting y coordinate
    z = microTEMP(index,1)+1-zOffset1; %setting z coordinate
    fprintf("\n\nID: %i", featIDsTEMP(x,y,z))
    
    % no calculations if inside void or outside microstructure
    if (GAOTEMP(:,:,x,y,z) ~= zeros(3))
    
        [XenvCompleteness,YenvCompleteness,ZenvCompleteness] = ...
            determine_neighborhood(microMax,featIDsTEMP,x,y,z);
    
        % Determine Disorientation between material points and neighbors, 
        % influenced by neighborhood -- Calculate kappa for material points              
        [dthe,diffOperatorX,diffOperatorY,diffOperatorZ] = determine_dthe(...
            XenvCompleteness,YenvCompleteness,ZenvCompleteness,GAOTEMP,x,y,z,symOp);
    
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
        kappaSRprime = transpose(GAOTEMP(:,:,x,y,z))*kappaSR*GAOTEMP(:,:,x,y,z);
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
        totalGNDdensityLR = 0;
        %fprintf("\n\nGAO\n");
        %fprintf("%f ", GAOTEMP(:,:,x,y,z));
        %fprintf("\n\ndthe\n");
        %fprintf("%f ", dthe);
        %fprintf("\n\nKappaSR\n");
        %fprintf("%f ", kappaSR);
        %fprintf("\n\nKappaSRprime\n");
        %fprintf("%f ", kappaSRprime);
        %fprintf("\n\nalphaSR\n");
        %fprintf("%f ", alphaSR);
        %fprintf("\n\nddSR\n");
        %fprintf("%f\n", ddSR);
        fprintf("\ntotalGNDdensitySR: %f", totalGNDdensitySR);
    else
        % tame output for voxels where misorientation can't be calc
        avgMisori = 0;
        totalGNDdensitySR = 0;
        totalGNDdensityLR = 0;
        [~,ddSR_dim] = size(A_sparse);
        ddSR = zeros(1,ddSR_dim);
        %fprintf('\nGAO is zeros!\n')
    end
end