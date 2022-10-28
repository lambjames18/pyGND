% Find void fraction in some layer of microstructure
[~,~,cssX,cssY,cssZ] = size(GAO);

% For xy layers
voidFracZ = zeros(cssZ,1);

for indexZ = 1:cssZ
    voidNum = 0;
    matrlNum = 0;
    for indexX = 1:cssX
        for indexY = 1:cssY
            if sum(GAO(:,:,indexX,indexY,indexZ),'all') == 0
                voidNum = voidNum +1;
            else
                matrlNum = matrlNum +1;
            end
        end
    end
    clc
    voidFracZ(indexZ,1) = voidNum/matrlNum;
    completion = 100*indexZ/cssZ;
    fprintf('\nCompletion %%: %5.1f',completion);
end

% For yz layers
voidFracX = zeros(cssX,1);

for indexX = 1:cssX
    voidNum = 0;
    matrlNum = 0;
    for indexY = 1:cssY
        for indexZ = 1:cssZ
            if sum(GAO(:,:,indexX,indexY,indexZ),'all') == 0
                voidNum = voidNum +1;
            else
                matrlNum = matrlNum +1;
            end
        end
    end
    clc
    voidFracX(indexX,1) = voidNum/matrlNum;
    completion = 100*indexX/cssX;
    fprintf('\nCompletion %%: %5.1f',completion);
end

% For xz layers
voidFracY = zeros(cssY,1);

for indexY = 1:cssY
    voidNum = 0;
    matrlNum = 0;
    for indexX = 1:cssX
        for indexZ = 1:cssZ
            if sum(GAO(:,:,indexX,indexY,indexZ),'all') == 0
                voidNum = voidNum +1;
            else
                matrlNum = matrlNum +1;
            end
        end
    end
    clc
    voidFracY(indexY,1) = voidNum/matrlNum;
    completion = 100*indexY/cssY;
    fprintf('\nCompletion %%: %5.1f',completion);
end

void_frac_plotting