function q = eu2om(eu)

thr = 1e-10;

c1 = cosd(eu(1));
c2  = cosd(eu(2));
c3 = cosd(eu(3));

s1 = sind(eu(1));
s2  = sind(eu(2));
s3 = sind(eu(3));

q = [ c1*c3-s1*c2*s3,  s1*c3+c1*c2*s3, s2*s3; ...
     -c1*s3-s1*c2*c3, -s1*s3+c1*c2*c3, s2*c3; ...
           s1*s2    ,      -c1*s2    ,  c2   ];

for i=1:3
  for j=1:3
    if (abs(q(i,j))< thr) 
        q(i,j) = 0.0;
    end
  end
end