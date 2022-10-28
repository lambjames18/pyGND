function disori = deltathetakV4(gA,gB,k,symOp)

% determine how many symmetry cases to evaluate
if sum(size(symOp)) == 6
    numSym = [1 1 1];
else
    numSym = size(symOp);
end
% preallocate number of unique misorientations to calc so disorientation
% can be found
misori_matrix = zeros((numSym(1,3))^2,1);

% preallocate iterators
gA_iter = 1;
gB_iter = 1;

% ensure calculation doesn't happen with orientation matrix of zeros
%gA_check = sum(sum(gA));
%gB_check = sum(sum(gB));
%gA_gB_check = sum(sum(gB-gA));

% if material points have same orientation or gA is zeros,
% ignore this process
comp = gB == gA;

if sum(comp,'all') ~= 9 && sum(gA,'all') ~= 0

    % lines 17-37 correspond to misorientation calc with ori matrices
    for delg_iter = 1:(numSym(1,3)^2)
        gA_temp = symOp(:,:,gA_iter)*gA;
        gB_temp = symOp(:,:,gB_iter)*gB;
        delg = gB_temp/gA_temp;
        
        % calculate delta theta, skip trace(delg) function for speed
        deltheta = acos(((sum(diag(delg)))-1)/2);
        
        if deltheta == 0
            misori_matrix(delg_iter,1) = 0;
        elseif k == 1
            misori_matrix(delg_iter,1) = -(delg(2,3)-...
                delg(3,2))*(deltheta/(2*sin(deltheta)));
        elseif k == 2
            misori_matrix(delg_iter,1) = -(delg(3,1)-...
                delg(1,3))*(deltheta/(2*sin(deltheta)));
        elseif k == 3
            misori_matrix(delg_iter,1) = -(delg(1,2)-...
                delg(2,1))*(deltheta/(2*sin(deltheta)));
        else
            misori_matrix(delg_iter,1) = 0;
        end
        % storing misorientation with specific symmetry operator applied

        if (gB_iter == numSym(1,3))
            gB_iter = 1;
            gA_iter = gA_iter + 1;
        else
            gB_iter = gB_iter +1;
        end
    end

    % finding lowest value of misorientation (disorientation)
    [~,d_col] = min(abs(misori_matrix),[],[1,2],'linear');

    % return disorientation
    disori = abs(misori_matrix(d_col));
else
    disori = 0;
end

end