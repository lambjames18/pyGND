function [dthe,diffOperatorX,diffOperatorY,diffOperatorZ] = ...
    determine_dthe(XenvCompleteness,YenvCompleteness,ZenvCompleteness,...
    GAO,x,y,z,symOp)

% determine misorientation and kappa based on material point neighborhood
dthe = zeros(3,3);

% orientation matrix of material point
gA = GAO(:,:,x,y,z);

% switch statement evaluating expression for x environment
switch XenvCompleteness
    case 'backward'
        gE = GAO(:,:,x-1,y,z); %setting Euler Angle at x - 1
        % First Nearest Neighbors 1st order backward difference----
        diffOperatorX = 1;
        % calc specific miorientation angles for kappa calc
        dthe(1,1) = deltathetakV4(gE,gA,1,symOp);
        dthe(2,1) = deltathetakV4(gE,gA,2,symOp);
        dthe(3,1) = deltathetakV4(gE,gA,3,symOp);
        
    case 'forward'
        gB = GAO(:,:,x+1,y,z); %setting Euler Angle at x + 1
        % First Nearest Neighbors 1st order forward difference-----
        diffOperatorX = 1;
        % calc specific miorientation angles for kappa calc
        dthe(1,1) = deltathetakV4(gA,gB,1,symOp);
        dthe(2,1) = deltathetakV4(gA,gB,2,symOp);
        dthe(3,1) = deltathetakV4(gA,gB,3,symOp);
        
    case 'central'
        gB = GAO(:,:,x+1,y,z); %setting Euler Angle at x + 1
        gE = GAO(:,:,x-1,y,z); %setting Euler Angle at x - 1
        % central finite difference
        diffOperatorX = 2;
        % calc specific miorientation angles for kappa calc
        dthe(1,1) = deltathetakV4(gE,gB,1,symOp);
        dthe(2,1) = deltathetakV4(gE,gB,2,symOp);
        dthe(3,1) = deltathetakV4(gE,gB,3,symOp);
        
    case 'constant'
        % in case no misorientation present
        diffOperatorX = 1;
        misoriX = 0;
        dthe(1,1) = 0;
        dthe(2,1) = 0;
        dthe(3,1) = 0;
end

% switch statement evaluating expression for y environment
switch YenvCompleteness
    case 'backward'
        gF = GAO(:,:,x,y-1,z); %setting Euler Angle at y - 1
        % First Nearest Neighbors 1st order backward difference----
        diffOperatorY = 1;
        % calc specific miorientation angles for kappa calc
        dthe(1,2) = deltathetakV4(gF,gA,1,symOp);
		dthe(2,2) = deltathetakV4(gF,gA,2,symOp);
		dthe(3,2) = deltathetakV4(gF,gA,3,symOp);
        
    case 'forward'
        gC = GAO(:,:,x,y+1,z); %setting Euler Angle at y + 1
        % First Nearest Neighbors 1st order forward difference----
        diffOperatorY = 1;
        % calc specific miorientation angles for kappa calc
        dthe(1,2) = deltathetakV4(gA,gC,1,symOp);
		dthe(2,2) = deltathetakV4(gA,gC,2,symOp);
		dthe(3,2) = deltathetakV4(gA,gC,3,symOp);
        
    case 'central'
        gC = GAO(:,:,x,y+1,z); %setting Euler Angle at y + 1
        gF = GAO(:,:,x,y-1,z); %setting Euler Angle at y - 1
        % central finite difference
        diffOperatorY = 2;
        % calc specific miorientation angles for kappa calc
        dthe(1,2) = deltathetakV4(gF,gC,1,symOp);
		dthe(2,2) = deltathetakV4(gF,gC,2,symOp);
		dthe(3,2) = deltathetakV4(gF,gC,3,symOp);
        
    case 'constant'
        % in case no misorientation, or no neighbor
        diffOperatorY = 1;
        misoriY = 0;
        dthe(1,2) = 0;
		dthe(2,2) = 0;
		dthe(3,2) = 0;
end

% switch statement evaluating expression for environment
switch ZenvCompleteness
    case 'backward'
        gG = GAO(:,:,x,y,z-1); %setting Euler Angle at z - 1
        % First Nearest Neighbors 1st order backward difference----
        diffOperatorZ = 1;
        % calc specific miorientation angles for kappa calc
        dthe(1,3) = deltathetakV4(gG,gA,1,symOp);
        dthe(2,3) = deltathetakV4(gG,gA,2,symOp);
        dthe(3,3) = deltathetakV4(gG,gA,3,symOp);
        
    case 'forward'
        gD = GAO(:,:,x,y,z+1); %setting Euler Angle at z + 1
        % First Nearest Neighbors 1st order forward difference-----
        diffOperatorZ = 1;
        % calc specific miorientation angles for kappa calc
        dthe(1,3) = deltathetakV4(gA,gD,1,symOp);
        dthe(2,3) = deltathetakV4(gA,gD,2,symOp);
        dthe(3,3) = deltathetakV4(gA,gD,3,symOp);
        
    case 'central'
        gD = GAO(:,:,x,y,z+1); %setting Euler Angle at z + 1
        gG = GAO(:,:,x,y,z-1); %setting Euler Angle at z - 1
        diffOperatorZ = 2;
        % calc specific miorientation angles for kappa calc
        dthe(1,3) = deltathetakV4(gG,gD,1,symOp);
        dthe(2,3) = deltathetakV4(gG,gD,2,symOp);
        dthe(3,3) = deltathetakV4(gG,gD,3,symOp);
        
    case 'constant'
        % zero misorientation along axis if no neighbor
        diffOperatorZ = 1;
        misoriZ = 0;
        dthe(1,3) = 0;
        dthe(2,3) = 0;
        dthe(3,3) = 0; 
end

end