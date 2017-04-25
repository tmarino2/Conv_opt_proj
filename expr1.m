load('data1.mat');

% randomly create the inital points
W0 = randn(rows,rows);
W0 = W0.*(W0 > 0);
H0 = randn(rows,cols);
H0 = H0.*(H0 > 0);
Th0 = diag(randn(rows,1));
Th0 = Th0.*(Th0 > 0);

for alpha=[0.001,0.05,0.1,0.5]
    for lambda=[0.01,0.1,0.5,1,10]
        for eta = [0.01,0.1,0.5,1,10]
        [ Obj, Ws, Ths, Hs, ObjWLL, ObjThLL, ObjHLL] = alt_min2( X, W0, Th0, H0, alpha, lambda, eta );
        exp_name = sprintf('step_%0.5f_l_%0.2f_e_%0.2f.mat',alpha,lambda,eta);
        save(exp_name)
        end
    end
end