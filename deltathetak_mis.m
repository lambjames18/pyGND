function disori = deltathetak_mis(gA,gB,k)

% if material points have same orientation, ignore this process

%logical check
comp = gB == gA;

if sum(comp,'all') ~= 9
    
        delg = gB/gA;
        
        % calculate delta theta, skip trace(delg) function for speed
        deltheta = acos(((sum(diag(delg)))-1)/2);
        
        if deltheta == 0
            misori = 0;
        elseif k == 1
            misori = -(delg(2,3)-...
                delg(3,2))*(deltheta/(2*sin(deltheta)));
        elseif k == 2
            misori = -(delg(3,1)-...
                delg(1,3))*(deltheta/(2*sin(deltheta)));
        elseif k == 3
            misori = -(delg(1,2)-...
                delg(2,1))*(deltheta/(2*sin(deltheta)));
        else
            misori = 0;
        end
        % storing misorientation with specific symmetry operator applied

    % return disorientation
    disori = misori;
else
    disori = 0;
end

end