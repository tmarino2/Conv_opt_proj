function [ Obj, W, Th, H, ObjWLL, ObjThLL, ObjHLL] = alt_min( X, W, Th, H, alpha, lambda, eta, step_over_k, stop_early_gnorm )
Obj = [norm(X - W*Th*H,'fro')^2];
%have better stopping criterion
max_iters = 250;
[ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta );  % this is just used to create some filler space
ObjWLL = [];
ObjThLL = [];
ObjHLL = [];

if stop_early_gnorm == 1
    eps = 0.1;
else
    eps = 0.0001;
end
num_inner_iters = 200;
for i = 1:max_iters
    [W, ll] = proj_sub_grad( W, grW, alpha, Th, H, X, lambda, eta, 1, num_inner_iters, step_over_k, eps);
%     ObjW = [ObjW, norm(X - W*Th*H,'fro')^2];
    ObjWLL = [ObjWLL; ll];
    [H, ll]= proj_sub_grad( H, grH, alpha, W, Th, X, lambda, eta, 3, num_inner_iters, step_over_k, eps);  
    ObjHLL = [ObjHLL; ll];
    [Th, ll] = proj_sub_grad( Th, grTh, 0.00001*alpha, W, H, X, lambda, eta, 2, num_inner_iters, step_over_k, eps);
%     ObjTh = [ObjTh, norm(X - W*Th*H,'fro')^2];
    ObjThLL = [ObjThLL; ll];
    Obj = [Obj, norm(X - W*Th*H,'fro')^2/size(X,2)]
end
end

