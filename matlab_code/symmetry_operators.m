% Symmetry operators used to determine the disorientation between two
% points

global symOp

% define symmetry operators for cubic or hexagonal symmetries
if (cs == 1 || cs == 2)
    % there are 24 symmetry operators for cubic symmetries
    % 576 (24 x 24) axis/angle pairs exist for any two cubic crystal lattices
    sym1 = [1 0 0; 0 1 0; 0 0 1];
    sym2 = [0 0 1; 1 0 0; 0 1 0];
    sym3 = [0 1 0; 0 0 1; 1 0 0];
    sym4 = [0 -1 0; 0 0 1; -1 0 0];
    sym5 = [0 -1 0; 0 0 -1; 1 0 0];
    sym6 = [0 1 0; 0 0 -1; -1 0 0];
    sym7 = [0 0 -1; 1 0 0; 0 -1 0];
    sym8 = [0 0 -1; -1 0 0; 0 1 0];
    sym9 = [0 0 1; -1 0 0; 0 -1 0];
    sym10 = [-1 0 0; 0 1 0; 0 0 -1];
    sym11 = [-1 0 0; 0 -1 0; 0 0 1];
    sym12 = [1 0 0; 0 -1 0; 0 0 -1];
    sym13 = [0 0 -1; 0 -1 0; -1 0 0];
    sym14 = [0 0 1; 0 -1 0; 1 0 0];
    sym15 = [0 0 1; 0 1 0; -1 0 0];
    sym16 = [0 0 -1; 0 1 0; 1 0 0];
    sym17 = [-1 0 0; 0 0 -1; 0 -1 0];
    sym18 = [1 0 0; 0 0 -1; 0 1 0];
    sym19 = [1 0 0; 0 0 1; 0 -1 0];
    sym20 = [-1 0 0; 0 0 1; 0 1 0];
    sym21 = [0 -1 0; -1 0 0; 0 0 -1];
    sym22 = [0 1 0; -1 0 0; 0 0 -1];
    sym23 = [0 1 0; 1 0 0; 0 0 -1];
    sym24 = [0 -1 0; 1 0 0; 0 0 -1];
    
    symOp = cat(3, sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8,...
        sym9, sym10, sym11, sym12, sym13, sym14, sym15, sym16, sym17, sym18,...
        sym19, sym20, sym21, sym22, sym23, sym24);
    
elseif (cs == 3)
    % there are 12 symmetry operators for hexagonal symmetries
    % like A matrix for HCP, ortho-hexagonal coordinates
    % 144 (12 x 12) axis/angle pairs exist for any two hexagonal lattices
    a = sqrt(3)/2;
    
    sym1 = [1 0 0; 0 1 0; 0 0 1];
    sym2 = [-0.5 a 0; -a -0.5 0; 0 0 1];
    sym3 = [-0.5 -a 0; a -0.5 0; 0 0 1];
    sym4 = [0.5 a 0; -a 0.5 0; 0 0 1];
    sym5 = [-1 0 0; 0 -1 0; 0 0 1];
    sym6 = [0.5 -a 0; a 0.5 0; 0 0 1];
    sym7 = [-0.5 -a 0; -a 0.5 0; 0 0 -1];
    sym8 = [1 0 0; 0 -1 0; 0 0 -1];
    sym9 = [-0.5 a 0; a 0.5 0; 0 0 -1];
    sym10 = [0.5 a 0; a -0.5 0; 0 0 -1];
    sym11 = [-1 0 0; 0 1 0; 0 0 -1];
    sym12 = [0.5 -a 0; -a -0.5 0; 0 0 -1];
    
    symOp = cat(3, sym1, sym2, sym3, sym4, sym5, sym6, sym7, sym8,...
        sym9, sym10, sym11, sym12);
else
    fprintf('\nWarning! Crystal structure is not known. No symmetry operators have been defined.\n\n'); 
end