fname = 'D:\IMMI Work\FFT_Inputs\FFT_Ti7_1pct_r2_REDUX.txt';
cs = crystalSymmetry.load('Ti-Titanium-alpha.cif');
ori1 = orientation.load(fname,'columnNames',{'phi1','Phi','phi2','x','y','z'},cs);
odf1 = calcDensity(ori1);
h = [Miller(0,0,0,1,cs,'uvtw'),Miller(1,0,-1,0,cs,'uvtw'),Miller(2,-1,-1,0,cs,'uvtw')];
setMTEXpref('xAxisDirection','east');setMTEXpref('zAxisDirection','outOfPlane');
figure;plotPDF(odf1,h,'antipodal','projection','eangle','contourf','minmax');