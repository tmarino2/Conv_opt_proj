function [Norm_diff Obj, W, Th, H, ObjWLL, ObjThLL, ObjHLL] = alt_min3( X, W, Th, H, alpha, lambda, eta )
% Obj = [norm(X - W*Th*H,'fro')^2/size(X,2)];
Obj = [compute_f(X,W,Th,H,lambda,eta)];
%have better stopping criterion
max_iters = 200; %100;
Norm_diff = zeros(1,max_iters);
% [ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta );  % this is just used to create some filler space
ObjWLL = {1,max_iters};
ObjThLL = {1,max_iters};
ObjHLL = {1,max_iters};
for i = 1:max_iters
    [W, ll] = proj_sub_grad3( W, alpha, Th, H, X, lambda, eta, 1, 200);
%     ObjW = [ObjW, norm(X - W*Th*H,'fro')^2];
    ObjWLL{i} = ll;
    [H, ll]= proj_sub_grad3( H, alpha, W, Th, X, lambda, eta, 3, 500 );  
    ObjHLL{i} = ll;
    [Th, ll] = proj_sub_grad3( Th, 0.1*alpha, W, H, X, lambda, eta, 2, 200 );
%     ObjTh = [ObjTh, norm(X - W*Th*H,'fro')^2];
    ObjThLL{i} = ll;
    Norm_diff(i) = norm(X - W*Th*H,'fro')^2/size(X,2);
    norm(X - W*Th*H,'fro')^2/size(X,2)
%     Obj = [Obj, norm(X - W*Th*H,'fro')^2/size(X,2)];
    Obj = [Obj, compute_f(X,W,Th,H,lambda,eta)];   
end
end