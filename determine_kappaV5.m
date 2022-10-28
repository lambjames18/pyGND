function [kappa] = determine_kappaV5(dthe,...
    diffOperatorX,diffOperatorY,diffOperatorZ,...
    X_spacing,Y_spacing,Z_spacing)

%global X_spacing Y_spacing Z_spacing

% kappa must be calculated for material point
%----------------------------------------------------------------------

% Calc three kappa components for x direction
kappa(1,1) = dthe(1,1)/(diffOperatorX*X_spacing);
kappa(2,1) = dthe(2,1)/(diffOperatorX*X_spacing);
kappa(3,1) = dthe(3,1)/(diffOperatorX*X_spacing);

% Calc three kappas for y direction
kappa(1,2) = dthe(1,2)/(diffOperatorY*Y_spacing);
kappa(2,2) = dthe(2,2)/(diffOperatorY*Y_spacing);
kappa(3,2) = dthe(3,2)/(diffOperatorY*Y_spacing);
    
% Calc three kappas for z direction                    
kappa(1,3) = dthe(1,3)/(diffOperatorZ*Z_spacing);
kappa(2,3) = dthe(2,3)/(diffOperatorZ*Z_spacing);
kappa(3,3) = dthe(3,3)/(diffOperatorZ*Z_spacing);
%----------------------------------------------------------------------

end