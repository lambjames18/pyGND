function [XenvCompleteness,YenvCompleteness,ZenvCompleteness] = ...
    determine_neighborhood(microMax,featIDs,x,y,z)

%global microMax featIDs micro
%{
x = micro(index,3)+1; %setting x coordinate
y = micro(index,2)+1; %setting y coordinate
z = micro(index,1)+1; %setting z coordinate
%}
%----------------- X completeness ------------
% if condition used to calc based on 'full' env
% checking completeness of the voxel neighborhood in x dimension

if (x-1 == 0)
    XenvCompleteness = 'forward';
elseif (x == microMax(1,3)+1)
    XenvCompleteness = 'backward';
elseif (featIDs(x,y,z) == featIDs(x+1,y,z) && featIDs(x,y,z) ...
        ~= featIDs(x-1,y,z)) 
    XenvCompleteness = 'forward';
elseif (featIDs(x,y,z) ~= featIDs(x+1,y,z) && featIDs(x,y,z) ...
        == featIDs(x-1,y,z))
    XenvCompleteness = 'backward';
elseif (featIDs(x,y,z) == featIDs(x+1,y,z) && featIDs(x,y,z) ...
        == featIDs(x-1,y,z))
    XenvCompleteness = 'central';
else
    XenvCompleteness = 'constant';
end

%--------------------------------------------------------------------------
%----------------- Y completeness ------------
% checking completeness of the voxel neighborhood in y dimension    

if (y-1 == 0)
    YenvCompleteness = 'forward';
elseif (y == microMax(1,2)+1)
    YenvCompleteness = 'backward';
elseif (featIDs(x,y,z) == featIDs(x,y+1,z) && featIDs(x,y,z) ...
        ~= featIDs(x,y-1,z))
    YenvCompleteness = 'forward';
elseif (featIDs(x,y,z) ~= featIDs(x,y+1,z) && featIDs(x,y,z) ...
        == featIDs(x,y-1,z)) 
    YenvCompleteness = 'backward';
elseif (featIDs(x,y,z) == featIDs(x,y+1,z) && featIDs(x,y,z) ...
        == featIDs(x,y-1,z))
    YenvCompleteness = 'central';
else
    YenvCompleteness = 'constant';
end

%--------------------------------------------------------------------------
%----------------- Z completeness ------------
% checking completeness of the voxel neighborhood in z dimension 

if (z-1 == 0)
    ZenvCompleteness = 'forward';
elseif (z == microMax(1,1)+1)
    ZenvCompleteness = 'backward';
elseif (featIDs(x,y,z) == featIDs(x,y,z+1) && featIDs(x,y,z) ...
        ~= featIDs(x,y,z-1)) 
    ZenvCompleteness = 'forward';
elseif (featIDs(x,y,z) ~= featIDs(x,y,z+1) && featIDs(x,y,z) ...
        == featIDs(x,y,z-1))
    ZenvCompleteness = 'backward';
elseif (featIDs(x,y,z) == featIDs(x,y,z+1) && featIDs(x,y,z) ...
        == featIDs(x,y,z-1))
    ZenvCompleteness = 'central';  
else
    ZenvCompleteness = 'constant';
end
