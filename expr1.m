load('data1.mat');

% randomly create the inital points
W0 = randn(rows,rows);
W0 = W0.*(W0 > 0);
H0 = randn(rows,cols);
H0 = H0.*(H0 > 0);
Th0 = randn(rows,1);
Th0 = Th0.*(Th0 > 0) + .001;  % make nothing be zero exactly
Th0 = diag(Th0);

[ Obj, Ws, Ths, Hs, ObjWLL, ObjThLL, ObjHLL] = alt_min( X, W0, Th0, H0, 0.05, 0.5, 0.5 );

