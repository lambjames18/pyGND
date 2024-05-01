% xtal --------------------------------------------------------------------
% decide which crystallography is relevant for material of interest
% includes burgers vector mag, linear operator B, or A matrix -------------

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% USER INPUT

cs = 2; % Crystallography (1 FCC, 2 BCC, 3 HCP)

burgers = 2.86; % Burgers vector magnitude (Ta 2.86 Angstroms)

A_matrix_choice = 1; % A matrix selection for BCC or HCP, determines which slip systems are active
% BCC options-> 1: screw+[110], 2: screw+[112], 3: screw+[123], 4: screw+[110]+[112], 5: screw+[110]+[112]+[123]
% HCP options-> 1: basal, 2: basal+prismatic, 3: basal+prismatic+pyramidal(c+a)
% FCC options-> doesn't matter, always uses all slip systems

%% END USER INPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate full A matrices
BCC_A_matrix_generationV2
HCP_A_matrix_mk3

% create linear operator B for FCC
% constants used for linear operator
a = sqrt(3)/9;
c = sqrt(3)/84;
d = 1/18;
f = 3/14;

% Linear operator used to calculate dd from Nye components

% See Arsenlis & Parks 1999
B = [a,7*c,-13*c,-7*c,-a,13*c,c,-c,0;
    -a,13*c,-7*c,-c,0,c,7*c,-13*c,a;
    0,c,-c,-13*c,a,7*c,13*c,-7*c,-a;
    a,-7*c,13*c,7*c,-a,13*c,-c,-c,0;
    -a,-13*c,7*c,c,0,c,-7*c,-13*c,a;
    0,-c,c,13*c,a,7*c,-13*c,-7*c,-a;
    a,-7*c,-13*c,7*c,-a,-13*c,c,c,0;
    -a,-13*c,-7*c,c,0,-c,7*c,13*c,a;
    0,-c,-c,13*c,a,-7*c,13*c,7*c,-a;
    a,7*c,13*c,-7*c,-a,-13*c,-c,c,0;
    -a,13*c,7*c,-c,0,-c,-7*c,13*c,-a;
    0,c,c,-13*c,a,-7*c,-13*c,7*c,-a;
    5*d,f,0,f,5*d,0,0,0,-d;
    5*d,0,f,0,-d,0,f,0,5*d;
    -d,0,0,0,5*d,f,0,f,5*d;
    5*d,-f,0,-f,5*d,0,0,0,-d;
    5*d,0,-f,0,-d,0,-f,0,5*d;
    -d,0,0,0,5*d,-f,0,-f,5*d];


if(cs == 2)
    switch A_matrix_choice
        case 1
            a_bcc = double(A_bcc(:,1:16));
            fprintf('You selected: case 1\n');
            numModes = 2;
        case 2
            a_bcc = double([A_bcc(:,1:4) A_bcc(:,17:28)]);
            fprintf('You selected: case 2\n');
            numModes = 2;
        case 3
            a_bcc = double([A_bcc(:,1:4) A_bcc(:,29:52)]);
            fprintf('You selected: case 3\n');
            numModes = 2;
        case 4
            a_bcc = double(A_bcc(:,1:28));
            fprintf('You selected: case 4\n');
            numModes = 3;
        case 5
            a_bcc = double(A_bcc);
            fprintf('You selected: case 5\n');
            numModes = 4;
    end
    
    %2.86A for Tantalum BCC
    %burgers = 2.86E-10;
    A_sparse = sparse(a_bcc);
    [numNye,numSlip] = size(A_sparse);
    
elseif(cs == 3)
    switch A_matrix_choice
        case 1
            A_hcp = [d1 d2];
            fprintf('You selected: basal\n');
            numModes = 2;
        case 2
            A_hcp = [d1 d2 d3];
            fprintf('You selected: basal + prismatic\n');
            numModes = 3;
        case 3
            A_hcp = [d1 d2 d3 d4 d5];
            fprintf('You selected: basal + prismatic + pyramidal(c+a)\n');
            numModes = 5;
    end
    

    A_sparse = sparse(A_hcp);
    [numNye,numSlip] = size(A_sparse);
    
else
    A_sparse = zeros(9,18); %dummy variable
    numSlip = 18;
    numModes = 4;
    
end