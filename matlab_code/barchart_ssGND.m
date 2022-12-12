% Script used to evaluate average ss values for GND density
figure
mean_GNDss = squeeze(mean(GND_SS,[1 2 3]));
bar(mean_GNDss);
hold on
set(gca,'yscale','log')
ylim([10^10 10^12])

sum_GNDss = squeeze(sum(GND_SS,[1 2 3]));

avgBasalGND = mean(mean_GNDss(1:6))
avgPrismaticGND = mean(mean(mean_GNDss(7:9)) + mean(mean_GNDss(1:3)))
avgPyramidalGND = mean(mean_GNDss(10:33))
avgGND = mean(mean_GNDss)

totBasalGND = mean(GND_basal,'all')
totPrismaticGND = mean(GND_pris,'all')
totPyramidalGND = mean(GND_pyr,'all')
totGND = mean(GND_SR,'all')

GND_report = [avgBasalGND avgPrismaticGND avgPyramidalGND avgGND totBasalGND totPrismaticGND totPyramidalGND totGND]