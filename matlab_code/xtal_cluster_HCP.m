% xtal --------------------------------------------------------------------
% decide which crystallography is relevant for material of interest
% includes burgers vector mag, linear operator B, or A matrix -------------

%cs = input('Input crystallography: \n 1: FCC \n 2: BCC \n 3: HCP\n\n');
fprintf('Input crystallography:\n')
cs = 3

%define burgers vector magnitude
%burgers = input(['Input Burgers Vector (A): \n2.86A for Tantalum BCC'...
%    '\n2.5A for IN718 & AlNiCo9\n2.95A for Ti'...
%    '\n   A for CoNi\n\n']);
fprintf('Burgers (A):\n')
burgers = 2.95

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
    
% prompt user based on xtal selection
if(cs == 2)
    fprintf('Include which slip modes?\n');
    %A_matrix_choice = input('1: screw + [110]\n2: screw + [112]\n3: screw + [123]\n4: screw + [110] + [112]\n5: screw + [110] + [112] + [123]\n');
    A_matrix_choice = 4
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
    fprintf('Include which slip modes? \n');
    %A_matrix_choice = input('1: basal\n2: basal + prismatic\n3: basal + prismatic + pyramidal(c+a)\n');
    A_matrix_choice = 3
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
    
    % 0.295nm for Ti 
    % burgers = 2.95E-10;
    A_sparse = sparse(A_hcp);
    [numNye,numSlip] = size(A_sparse);
    
else
    %fprintf('Defaulting to FCC.');

    %.25nm, see by neutron diffraction via Zhang et. al.
    % burgers = 2.5E-10;
    A_sparse = zeros(9,18); %dummy variable
    
    %defining number of slip systems and slip modes
    numSlip = 18;
    numModes = 4;
    
end