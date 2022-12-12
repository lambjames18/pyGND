 function dd = L2_SparseV2(alphaPrime,cs,A_sparse,B,burgers)

%global A_sparse B burgers

% L2 minimization with sparse solver

% Equation to be solved -> A*rho[array form] = Lambda[Nye in array form] 
% Can solve via minimize‖Ax−b‖2
% requires two steps
% [c,R] = qr(A,Lambda);
% rho = R\c

% Nye tensor must be converted into array form Lambda
%----------------------------------------------------------------------
Lambda = [alphaPrime(1,1);alphaPrime(1,2);alphaPrime(1,3);alphaPrime(2,1);
    alphaPrime(2,2);alphaPrime(2,3);alphaPrime(3,1);alphaPrime(3,2);alphaPrime(3,3)];

if (cs == 2 || cs == 3)
    % two steps to solve via minimize‖Ax−b‖2
    %[c,R] = qr(transpose(A_sparse),transpose(Lambda));
    B = transpose(A_sparse)*inv(A_sparse*transpose(A_sparse));
    dd = B*Lambda;
    [~,numSlip] = size(A_sparse);
    % calc dislocation density (rho) using burgers vector
    if numSlip > 9 && cs == 3
        burgers_ca = 4.68;
        burgers_ca = burgers_ca*1E-10;
        dd(1:9) = dd(1:9)/burgers;
        dd(10:33) = dd(10:33)/burgers_ca;
    else
        dd = dd/burgers;
    end

else
    % explicitly solve for FCC dislocation density with linear operator
    dd = B*Lambda;
    
    % calc dislocation density (rho) using burgers vector
    dd = dd/burgers;
end
%-----------------------------------------------------------------------
end