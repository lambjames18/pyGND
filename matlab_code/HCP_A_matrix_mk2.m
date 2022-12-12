% Relevant Direcitons in [uvtw] notation

% Basal Slip
bMBbasal = [1 1 -2 0;
    1 -2 1 0;
    -2 1 1 0];

nMBbasal = [0 0 0 1;
    0 0 0 1;
    0 0 0 1];

% Prismatic Slip
bMBprismatic = [2 -1 -1 0;
                -1 2 -1 0;
                1 1 -2 0];

nMBprismatic = [0 1 -1 0;
                1 0 -1 0;
                1 -1 0 0];

% Pyramidal <c+a> Slip
bMBpyramidalCplusA = [-1 -1 2 3;
                      -2 1 1 3;
                      1 1 -2 3;
                      -1 2 -1 3;
                      2 -1 -1 3;
                      1 -2 1 3;
                      2 -1 -1 3;
                      1 1 -2 3;
                      -1 -1 2 3;
                      1 -2 1 3;
                      -2 1 1 3;
                      -1 2 -1 3];
                  
nMBpyramidalCplusA = [1 0 -1 1;
                      1 0 -1 1;    
                      0 -1 1 1;     
                      0 -1 1 1;   
                      -1 1 0 1;     
                      -1 1 0 1;     
                      -1 0 1 1;     
                      -1 0 1 1;     
                      0 1 -1 1;    
                      0 1 -1 1;     
                      1 -1 0 1;    
                      1 -1 0 1];  
                
% Conversion to [UVW] Miller Indices

bMillerBASAL = zeros(3,3);
nMillerBASAL = zeros(3,3);
tMillerBASAL = zeros(3,3);

for index = 1:3

    u = bMBbasal(index,1);
    v = bMBbasal(index,2);
    t = bMBbasal(index,3);
    w = bMBbasal(index,4);

    bMillerBASAL(index,1) = (1/3)*(u - t);
    bMillerBASAL(index,2) = (1/3)*(v - t);
    bMillerBASAL(index,3) = (1/3)*w;
    
    u = nMBbasal(index,1);
    v = nMBbasal(index,2);
    t = nMBbasal(index,3);
    w = nMBbasal(index,4);

    nMillerBASAL(index,1) = (u - t);
    nMillerBASAL(index,2) = (v - t);
    nMillerBASAL(index,3) = w;

end

bMillerPRISMATIC = zeros(3,3);
nMillerPRISMATIC = zeros(3,3);
tMillerPRISMATIC = zeros(3,3);

for index = 1:3

    u = bMBprismatic(index,1);
    v = bMBprismatic(index,2);
    t = bMBprismatic(index,3);
    w = bMBprismatic(index,4);

    bMillerPRISMATIC(index,1) = (1/3)*(u - t);
    bMillerPRISMATIC(index,2) = (1/3)*(v - t);
    bMillerPRISMATIC(index,3) = (1/3)*w;
    
    u = nMBprismatic(index,1);
    v = nMBprismatic(index,2);
    t = nMBprismatic(index,3);
    w = nMBprismatic(index,4);

    nMillerPRISMATIC(index,1) = (u - t);
    nMillerPRISMATIC(index,2) = (v - t);
    nMillerPRISMATIC(index,3) = w;


end

bMillerPYRAMIDALcplusa = zeros(12,3);
nMillerPYRAMIDALcplusa = zeros(12,3);
tMillerPYRAMIDALcplusa = zeros(12,3);

for index = 1:12

    u = bMBpyramidalCplusA(index,1);
    v = bMBpyramidalCplusA(index,2);
    t = bMBpyramidalCplusA(index,3);
    w = bMBpyramidalCplusA(index,4);

    bMillerPYRAMIDALcplusa(index,1) = (1/3)*(u - t);
    bMillerPYRAMIDALcplusa(index,2) = (1/3)*(v - t);
    bMillerPYRAMIDALcplusa(index,3) = (1/3)*w;
    
    u = nMBpyramidalCplusA(index,1);
    v = nMBpyramidalCplusA(index,2);
    t = nMBpyramidalCplusA(index,3);
    w = nMBpyramidalCplusA(index,4);

    nMillerPYRAMIDALcplusa(index,1) = (u - t);
    nMillerPYRAMIDALcplusa(index,2) = (v - t);
    nMillerPYRAMIDALcplusa(index,3) = w;
    
end

% determining tangent vectors for screw dislocations

tMillerBASALscrew = bMillerBASAL;

tMillerPYRAMIDALcplusascrew = bMillerPYRAMIDALcplusa;

% determining tangent vectors for edge dislocations
for index3 = 1:3
    tMillerBASAL(index3,:) = cross(nMillerBASAL(index3,:),bMillerBASAL(index3,:));
end
for index1 = 1:3
    tMillerPRISMATIC(index1,:) = cross(nMillerPRISMATIC(index3,:),bMillerPRISMATIC(index3,:));
end
for index2 = 1:12
    tMillerPYRAMIDALcplusa(index2,:) = cross(nMillerPYRAMIDALcplusa(index3,:),bMillerPYRAMIDALcplusa(index3,:));
end

% prepping dislocation dyads matrix
d1 = zeros(9,3);
d2 = zeros(9,3);
d3 = zeros(9,3);
d4 = zeros(9,12);
d5 = zeros(9,12);

for index = 1:3
    d1(1,index) = bMillerBASAL(index,1)*tMillerBASALscrew(index,1);
    d1(2,index) = bMillerBASAL(index,1)*tMillerBASALscrew(index,2);
    d1(3,index) = bMillerBASAL(index,1)*tMillerBASALscrew(index,3);
    d1(4,index) = bMillerBASAL(index,2)*tMillerBASALscrew(index,1);
    d1(5,index) = bMillerBASAL(index,2)*tMillerBASALscrew(index,2);
    d1(6,index) = bMillerBASAL(index,2)*tMillerBASALscrew(index,3);
    d1(7,index) = bMillerBASAL(index,3)*tMillerBASALscrew(index,1);
    d1(8,index) = bMillerBASAL(index,3)*tMillerBASALscrew(index,2);
    d1(9,index) = bMillerBASAL(index,3)*tMillerBASALscrew(index,3);
    
    d2(1,index) = bMillerBASAL(index,1)*tMillerBASAL(index,1);
    d2(2,index) = bMillerBASAL(index,1)*tMillerBASAL(index,2);
    d2(3,index) = bMillerBASAL(index,1)*tMillerBASAL(index,3);
    d2(4,index) = bMillerBASAL(index,2)*tMillerBASAL(index,1);
    d2(5,index) = bMillerBASAL(index,2)*tMillerBASAL(index,2);
    d2(6,index) = bMillerBASAL(index,2)*tMillerBASAL(index,3);
    d2(7,index) = bMillerBASAL(index,3)*tMillerBASAL(index,1);
    d2(8,index) = bMillerBASAL(index,3)*tMillerBASAL(index,2);
    d2(9,index) = bMillerBASAL(index,3)*tMillerBASAL(index,3);
    
    d3(1,index) = bMillerPRISMATIC(index,1)*tMillerPRISMATIC(index,1);
    d3(2,index) = bMillerPRISMATIC(index,1)*tMillerPRISMATIC(index,2);
    d3(3,index) = bMillerPRISMATIC(index,1)*tMillerPRISMATIC(index,3);
    d3(4,index) = bMillerPRISMATIC(index,2)*tMillerPRISMATIC(index,1);
    d3(5,index) = bMillerPRISMATIC(index,2)*tMillerPRISMATIC(index,2);
    d3(6,index) = bMillerPRISMATIC(index,2)*tMillerPRISMATIC(index,3);
    d3(7,index) = bMillerPRISMATIC(index,3)*tMillerPRISMATIC(index,1);
    d3(8,index) = bMillerPRISMATIC(index,3)*tMillerPRISMATIC(index,2);
    d3(9,index) = bMillerPRISMATIC(index,3)*tMillerPRISMATIC(index,3);
end

for index = 1:12
    d4(1,index) = bMillerPYRAMIDALcplusa(index,1)*tMillerPYRAMIDALcplusa(index,1);
    d4(2,index) = bMillerPYRAMIDALcplusa(index,1)*tMillerPYRAMIDALcplusa(index,2);
    d4(3,index) = bMillerPYRAMIDALcplusa(index,1)*tMillerPYRAMIDALcplusa(index,3);
    d4(4,index) = bMillerPYRAMIDALcplusa(index,2)*tMillerPYRAMIDALcplusa(index,1);
    d4(5,index) = bMillerPYRAMIDALcplusa(index,2)*tMillerPYRAMIDALcplusa(index,2);
    d4(6,index) = bMillerPYRAMIDALcplusa(index,2)*tMillerPYRAMIDALcplusa(index,3);
    d4(7,index) = bMillerPYRAMIDALcplusa(index,3)*tMillerPYRAMIDALcplusa(index,1);
    d4(8,index) = bMillerPYRAMIDALcplusa(index,3)*tMillerPYRAMIDALcplusa(index,2);
    d4(9,index) = bMillerPYRAMIDALcplusa(index,3)*tMillerPYRAMIDALcplusa(index,3);
    
    d5(1,index) = bMillerPYRAMIDALcplusa(index,1)*tMillerPYRAMIDALcplusascrew(index,1);
    d5(2,index) = bMillerPYRAMIDALcplusa(index,1)*tMillerPYRAMIDALcplusascrew(index,2);
    d5(3,index) = bMillerPYRAMIDALcplusa(index,1)*tMillerPYRAMIDALcplusascrew(index,3);
    d5(4,index) = bMillerPYRAMIDALcplusa(index,2)*tMillerPYRAMIDALcplusascrew(index,1);
    d5(5,index) = bMillerPYRAMIDALcplusa(index,2)*tMillerPYRAMIDALcplusascrew(index,2);
    d5(6,index) = bMillerPYRAMIDALcplusa(index,2)*tMillerPYRAMIDALcplusascrew(index,3);
    d5(7,index) = bMillerPYRAMIDALcplusa(index,3)*tMillerPYRAMIDALcplusascrew(index,1);
    d5(8,index) = bMillerPYRAMIDALcplusa(index,3)*tMillerPYRAMIDALcplusascrew(index,2);
    d5(9,index) = bMillerPYRAMIDALcplusa(index,3)*tMillerPYRAMIDALcplusascrew(index,3);
end
