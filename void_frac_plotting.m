figure
meanGNDxy = squeeze(mean(GND_SR,[1 2]));
plot(meanGNDxy)
hold on
meanGNDyz = squeeze(mean(GND_SR,[2 3]));
plot(meanGNDyz)
meanGNDxz = squeeze(mean(GND_SR,[1 3]));
plot(meanGNDxz)
%set(gca,'yscale','log')
ylim([(2.5*10^13) (7*10^13)])
hold off
figure
plot(voidFracZ)
hold on
plot(voidFracX)
plot(voidFracY)
ylim([0 0.7])