%BCC A matrix formulation
%{
%A_bcc = zeros(9,48);
% OLD NOTATION
% b vectors for systems {110} 1->12 as screw, 13->24 as edge
% b vectors for systems {112} 25->36 as screw, 37->48 as edge
%}
% b vectors for systems {110}{112}{321} 1->4 as screw
% b vectors for systems {110} 5->16 as edge
% b vectors for systems {112} 17->28 as edge
% b vectors for systems {123} 29->52 as edge
%{
bedge = [1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;%start of 123 systems
    1 1 -1;
    1 -1 1;
    -1 1 1;
    -1 1 1;
    1 1 -1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    -1 1 1;
    -1 1 1;
    1 -1 1;
    1 1 -1;
    1 -1 1;
    1 1 -1;
    1 -1 1;
    1 1 -1;
    1 1 -1;
    1 -1 1;
    -1 1 1;
    -1 1 1];
%}
% setting plane normals
%{
nedge = [0 1 1;
    1 0 1;
    1 -1 0;
    0 1 -1;
    1 0 1;
    1 1 0;
    0 1 1;
    1 0 -1;
    1 1 0;
    0 1 -1;
    1 0 -1;
    1 -1 0;
    2 -1 1;
    -1 2 1;
    1 1 2;
    2 1 1;
    1 2 -1;
    1 -1 2;
    2 1 -1;
    1 2 1;
    -1 1 2;
    -2 1 1;
    1 -2 1;
    1 -2 1;%start of 123 systems
    1 2 3;
    1 3 2;
    3 1 2;
    3 2 1;
    2 1 3;
    2 3 1;
    1 2 3;
    1 3 2;
    3 1 2;
    3 2 1;
    2 1 3;
    2 3 1;
    1 2 3;
    1 3 2;
    3 1 2;
    3 2 1;
    1 2 3;
    2 3 1;
    -1 2 3;
    -1 3 2;
    3 -1 2;
    3 2 -1;
    2 -1 3;
    2 3 -1];
%}
bedge = single((1/sqrt(3))*[1 1 -1;%{110}<111> SLIP
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;%{112}<111> SLIP
    -1 -1 1;
    -1 -1 1;
    -1 -1 1;
    -1 1 1;
    -1 1 1;
    -1 1 1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;%{123}<111> SLIP
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1]);

nedge(1:12,:) = single((1/sqrt(2))*[0 1 1;%{110}<111> SLIP
    1 0 1;     
    1 -1 0;     
    0 1 -1;     
    1 0 1;     
    1 1 0;     
    0 1 1;     
    1 0 -1;     
    1 1 0;     
    0 1 -1;     
    1 0 -1; 
    1 -1 0]);

    %{112}<111> SLIP
    nedge(13:24,:) = single((1/sqrt(6))*[-2 1 -1;           
    1 -2 -1;    
    1 1 2;    
    -2 -1 -1;    
    1 2 -1;    
    1 -1 2;    
    2 1 -1;     
    -1 -2 -1;     
    -1 1 2;     
    2 -1 -1;     
    -1 2 -1; 
    -1 -1 2]);

    %{123}<111> SLIP
    nedge(25:48,:) = single((1/sqrt(14))*[1 2 3;     
    -1 3 2;     
    2 1 3; 
    -2 3 1;   
    3 -1 2;  
    3 -2 1;  
    -1 2 -3;     
    1 3 -2;    
    2 -1 3;   
    2 3 -1;    
    3 1 2;    
    3 2 1;   
    1 -2 -3;    
    1 3 2;   
    2 -1 -3;   
    2 3 1;   
    3 1 -2;  
    3 2 -1;   
    1 2 -3;     
    1 -3 2;     
    2 1 -3;   
    2 -3 1;   
    -3 1 2; 
    -3 2 1]);
%{ 
t vectors for each b if screw
t = b;
% determining tangent vectors for edge dislocations
for index1 = 13:24
    t(index1,:) = cross(n(index1,:),b(index1,:));
end
for index2 = 37:48
    t(index2,:) = cross(n(index2,:),b(index2,:));
end
%}
b = round((1/sqrt(3))*[1 1 -1;     
    1 -1 -1;    
    1 -1 1;
    1 1 1],4);
%{
n = [0 1 1;
    1 0 1;
    1 -1 0;
    0 1 -1;
    1 0 1;
    1 1 0;];
    %}
tscrew = b;
t = zeros(48,3);
% prepping dislocation dyads matrix
d = zeros(9,42);

%Calc Screw Dislocation Density
for index = 1:4
    d(1,index) = round(b(index,1)*tscrew(index,1),8);
    d(2,index) = round(b(index,1)*tscrew(index,2),8);
    d(3,index) = round(b(index,1)*tscrew(index,3),8);
    d(4,index) = round(b(index,2)*tscrew(index,1),8);
    d(5,index) = round(b(index,2)*tscrew(index,2),8);
    d(6,index) = round(b(index,2)*tscrew(index,3),8);
    d(7,index) = round(b(index,3)*tscrew(index,1),8);
    d(8,index) = round(b(index,3)*tscrew(index,2),8);
    d(9,index) = round(b(index,3)*tscrew(index,3),8);
end

%Calc Edge Dislocation Density
%{
b = [1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 -1;
    1 1 -1;
    1 1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 -1;
    1 -1 1;
    1 -1 1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;%start of 123 systems
    1 1 -1;
    1 -1 1;
    -1 1 1;
    -1 1 1;
    1 1 -1;
    1 -1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    1 1 1;
    -1 1 1;
    -1 1 1;
    1 -1 1;
    1 1 -1;
    1 -1 1;
    1 1 -1;
    1 -1 1;
    1 1 -1;
    1 1 -1;
    1 -1 1;
    -1 1 1;
    -1 1 1];
%}
%{
% setting plane normals
n = [0 1 1;
    1 0 1;
    1 -1 0;
    0 1 -1;
    1 0 1;
    1 1 0;
    0 1 1;
    1 0 -1;
    1 1 0;
    0 1 -1;
    1 0 -1;
    1 -1 0;
    0 1 1;
    1 0 1;
    1 -1 0;
    0 1 -1;
    1 0 1;
    1 1 0;
    0 1 1;
    1 0 -1;
    1 1 0;
    0 1 -1;
    1 0 -1;
    1 -1 0;
    2 -1 1;
    -1 2 1;
    1 1 2;
    2 1 1;
    1 2 -1;
    1 -1 2;
    2 1 -1;
    1 2 1;
    -1 1 2;
    -2 1 1;
    1 -2 1;
    1 -2 1;
    2 -1 1;
    -1 2 1;
    1 1 2;
    2 1 1;
    1 2 -1;
    1 -1 2;
    2 1 -1;
    1 2 1;
    -1 1 2;
    -2 1 1;
    1 -2 1;
    1 -2 1;%start of 123 systems
    1 2 3;
    1 3 2;
    3 1 2;
    3 2 1;
    2 1 3;
    2 3 1;
    1 2 -3;
    1 -3 2;
    -3 1 2;
    -3 2 2;
    2 1 -3;
    2 -3 1;
    1 -2 3;
    1 3 -2;
    3 1 -2;
    3 -2 1;
    -2 1 3;
    -2 3 1;
    -1 2 3;
    -1 3 2;
    3 -1 2;
    3 2 -1;
    2 -1 3;
    2 3 -1];
    %}
%
% determining tangent vectors for edge dislocations
t(1:12,:) = [-0.8165 0.4082 -0.4082;
             -0.4082 0.8165 0.4082;
              0.4082 0.4082 0.8165;
             -0.8165 -0.4082 -0.4082;
              0.4082 0.8165 -0.4082;
             -0.4082 0.4082 -0.8165;
              0.8165 0.4082 -0.4082;
             -0.4082 -0.8165 -0.4082;
              0.4082 -0.4082 -0.8165;
              0.8165 -0.4082 -0.4082;
              0.4082 -0.8165 0.4082;
             -0.4082 -0.4082 0.8165];


for index1 = 13:48
    t(index1,:) = single(cross(nedge(index1,:),bedge(index1,:)));
end
%}
%{
for index2 = 37:48
    t(index2,:) = cross(n(index2,:),b(index2,:));
end
%}
%
for index = 5:52
    d(1,index) = bedge(index-4,1)*t(index-4,1);
    d(2,index) = bedge(index-4,1)*t(index-4,2);
    d(3,index) = bedge(index-4,1)*t(index-4,3);
    d(4,index) = bedge(index-4,2)*t(index-4,1);
    d(5,index) = bedge(index-4,2)*t(index-4,2);
    d(6,index) = bedge(index-4,2)*t(index-4,3);
    d(7,index) = bedge(index-4,3)*t(index-4,1);
    d(8,index) = bedge(index-4,3)*t(index-4,2);
    d(9,index) = bedge(index-4,3)*t(index-4,3);
end
%}
screws = d(1:9,1:4);
edges = d(1:9,5:52);
a_bcc = [round(screws,4) round(edges,6)];

A_bcc = single(a_bcc);

checknorm = 1;
%checknorm = norm(bedge(1,:));
%{
for checkindexbedge = 1:48
    if((norm(bedge(checkindexbedge,:)) - checknorm) == 0)
        fprintf('Norm of bedge %u is correct \n',checkindexbedge)
        fprintf('Norm is: %8.17f \n\n',norm(bedge(checkindexbedge,:)))
    else
        fprintf('Norm of bedge %u is incorrect \n',checkindexbedge)
        fprintf('Norm is: %8.17f \n\n',norm(bedge(checkindexbedge,:)))
    end
end

for checkindexnedge = 1:48
    if((norm(nedge(checkindexnedge,:)) - checknorm) == 0)
        fprintf('Norm of nedge %u is correct \n',checkindexnedge)
        fprintf('Norm is: %8.17f \n\n',norm(nedge(checkindexnedge,:)))
    else
        fprintf('Norm of nedge %u is incorrect \n',checkindexnedge)
        fprintf('Norm is: %8.17f \n\n',norm(nedge(checkindexnedge,:)))
    end
end

for checkindext = 1:48
    if((norm(t(checkindext,:)) - checknorm) == 0)
        fprintf('Norm of t %u is correct \n',checkindext)
        fprintf('Norm is: %8.17f \n\n',norm(t(checkindext,:)))
    else
        fprintf('Norm of t %u is incorrect \n',checkindext)
        fprintf('Norm is: %8.17f \n\n',norm(t(checkindext,:)))
    end
end

for checkindextscrew = 1:4
    if((norm(tscrew(checkindextscrew,:)) - checknorm) == 0)
        fprintf('Norm of tscrew %u is correct \n',checkindextscrew)
        fprintf('Norm is: %8.17f \n\n',norm(tscrew(checkindextscrew,:)))
    else
        fprintf('Norm of tscrew %u is incorrect \n',checkindextscrew)
        fprintf('Norm is: %8.17f \n\n',norm(tscrew(checkindextscrew,:)))
    end
end

for checkindexb = 1:4
    if((norm(tscrew(checkindexb,:)) - checknorm) == 0)
        fprintf('Norm of tscrew %u is correct \n',checkindexb)
        fprintf('Norm is: %8.17f \n\n',norm(tscrew(checkindexb,:)))
    else
        fprintf('Norm of tscrew %u is incorrect \n',checkindexb)
        fprintf('Norm is: %8.17f \n\n',norm(tscrew(checkindexb,:)))
    end
end
%}