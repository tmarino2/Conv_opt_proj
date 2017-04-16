load('data1.mat');

% randomly create the inital points
W0 = randn(rows,rows);
W0 = W0.*(W0 > 0);
H0 = randn(rows,cols);
H0 = H0.*(H0 > 0);
Th0 = diag(randn(rows,1));
Th0 = Th0.*(Th0 > 0);

[ Obj, Ws, Ths, Hs, ObjW, ObjTh , ObjWLL, ObjThLL, ObjHLL] = alt_min( X, W0, Th0, H0, 0.1, 1, 0.2 );