rows = 40;
cols = 5000;
W = randn(rows);
Th = (randn(rows,1));
W = W.*(W>0);
Th = Th.*(Th>0);
Th_p = diag(Th);
H = randn(rows,cols);
H = H.*(H>0);
X = W*Th_p*H;

save('data1.mat');