% Script used to input spatial GND info into EVPFFT model
[xLim,yLim,zLim] = size(GND_SR);
GND_SR_array = zeros(xLim*yLim*zLim,1);
GND_SR_array_iter = 1;

for xIndex = 1:xLim
    for yIndex = 1:yLim
        for zIndex = 1:zLim
            GND_SR_array(GND_SR_array_iter) = GND_SR(xIndex,yIndex,zIndex);
            GND_SR_array_iter = GND_SR_array_iter + 1;
        end
    end
end

%filename = [Directory ID '_EVPFFT_GND_input.in'];
filename = [Directory 'GND.in'];
fileID = fopen(filename,'w');
formatSpec = '%8.2e\n';
fprintf(fileID,formatSpec,GND_SR_array);