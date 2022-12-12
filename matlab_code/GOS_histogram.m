% GOS histogram
f = figure;
f.Position(3:4) = [760 440]
h = histogram(GOS_nz,20,'FaceAlpha',0.75,'Normalization','Probability');
h.EdgeColor = [1 1 1];
xlim([0.00 0.2]);
%xlim([0.01 0.12])
xl = xline(m,'-',sprintf('GOS Average = %6.4f', m));
xl.LabelHorizontalAlignment = 'right';
xl.LabelVerticalAlignment = 'bottom';
x2 = xline(med,'-',sprintf('GOS Median = %6.4f', med));
x2.LabelHorizontalAlignment = 'left';
x2.LabelVerticalAlignment = 'bottom';
x3 = xline(GAM,'--r',sprintf('GAM = %6.4f', GAM));
x3.LabelHorizontalAlignment = 'right';
x3.LabelVerticalAlignment = 'top';
xlabel('Misorientation [Deg]','FontSize',18)
ylabel('Relative Frequency Fraction','FontSize',18)


x4 = xline(m-st,'-b','-1 GOS Standard Dev');
x4.LabelHorizontalAlignment = 'left';
x5 = xline(m+st,'-b','+1 GOS Standard Dev');
x5.LabelHorizontalAlignment = 'right';