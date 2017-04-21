% proj GD method

function [] = expr_all()
rows = 40;
cols = 5000;
W = [];
Th = [];
W = [];
Th = [];
Th_p = [];
H = [];
H = [];
X = [];
load('data1.mat');


    function [Obj, ObjWLL, ObjThLL, ObjHLL] = run_pgd(alpha, lambda, eta, step_over_k, stop_early_gnorm)

% randomly create the inital points
W0 = randn(rows,rows);
W0 = W0.*(W0 > 0);
H0 = randn(rows,cols);
H0 = H0.*(H0 > 0);
Th0 = randn(rows,1);
Th0 = Th0.*(Th0 > 0) + .001;  % make nothing be zero exactly
Th0 = diag(Th0);

    [ Obj, Ws, Ths, Hs, ObjWLL, ObjThLL, ObjHLL] = alt_min( X, W0, Th0, H0, alpha, lambda, eta, step_over_k , stop_early_gnorm);

    end

    function [Obj, ObjWLL, ObjTHLL, ObjHLL] = run_sda(lambda, eta)

% randomly create the inital points
W0 = randn(rows,rows);
W0 = W0.*(W0 > 0);
H0 = randn(rows,cols);
H0 = H0.*(H0 > 0);
Th0 = randn(rows,1);
Th0 = Th0.*(Th0 > 0) + .001;  % make nothing be zero exactly
Th0 = diag(Th0);

[ Obj, Ws, Ths, Hs, ObjWLL, ObjThLL, ObjHLL] = alt_min_sda( X, W0, Th0, H0, 1, 0.5 );
    end


% perform the experiments on the Proj GD
 % changing the step size, fixed
 stepPlots = [];
 for step = .1.^(1:4)
    [Obj, ObjWLL, ObjThLL, ObjHLL] = run_pgd(step, .5, .5, 0, 0);
    stepPlots = [stepPlots;Obj];
 end
 % TODO: might change these to log plot since there is a large range
 % between items
plot(1:size(stepPlots, 2), stepPlots);
title('Project gradient, fixed step size');
legend('.1', '.01', '.001', '.0001');
print('pgd-fixed-step.png', '-dpng');
stepPlots = [];  % clear

% step inner 1/k
[Obj, ObjWLL, ObjThLL, ObjHLL] = run_pgd(.01, .5, .5, 1, 0);
plot(1:size(Obj), Obj);
title('Projected gradient, 1/k step scaling');
print('pgd-dec-step.png', '-dpng');

% different lambda
lambdaPlots = [];
for lambda = .2 * (1:5)
    [Obj, ObjWLL, ObjThLL, ObjHLL] = run_pgd(.01, lambda, .5, 0, 0);
    lambdaPlots = [lambdaPlots;Obj];
end
plot(1:size(lambdaPlots, 2), lambdaPlots);
title('Project gradient, changing lambda');
legend('.2', '.4', '.6', '.8', '1');
print('pgd-lambda.png', '-dpng');
lambdaPlots = [];

% different eta
etaPlots = [];
for eta = .2 * (1:5)
    [Obj, ObjWLL, ObjThLL, ObjHLL] = run_pgd(.01, .5, eta, 0, 0);
    etaPlots = [etaPlots;Obj];
end
plot(1:size(etaPlots, 2), etaPlots);
title('Project gradient, changing eta');
legend('.2', '.4', '.6', '.8', '1');
print('pgd-eta.png', '-dpng');
etaPlots = [];

% stopping the inner loops using the norm of their gradients
[Obj, ObjWLL, ObjThLL, ObjHLL] = run_pgd(.01, .5, .5, 0, 1);
plot(1:size(Obj, 1), Obj);
title('Early stopping with gradient norm');
print('pgd-early-stop.png', '-dpng');



% perofrm the experiments on the SDA
   

end