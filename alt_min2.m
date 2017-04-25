function [ Obj, W, Th, H, ObjWLL, ObjThLL, ObjHLL] = alt_min2( X, W, Th, H, alpha, lambda, eta )
% Obj = [norm(X - W*Th*H,'fro')^2/size(X,2)];
Obj = [compute_f(X,W,Th,H,lambda,eta)];
%have better stopping criterion
max_iters = 200; %100;
% [ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta );  % this is just used to create some filler space
ObjWLL = [];
ObjThLL = [];
ObjHLL = [];
for i = 1:max_iters
    [W, ll] = proj_sub_grad2( W, alpha, Th, H, X, lambda, eta, 1, 200);
%     ObjW = [ObjW, norm(X - W*Th*H,'fro')^2];
    ObjWLL = [ObjWLL, ll];
    [H, ll]= proj_sub_grad2( H, alpha, W, Th, X, lambda, eta, 3, 500 );  
    ObjHLL = [ObjHLL, ll];
    [Th, ll] = proj_sub_grad2( Th, 0.00001*alpha, W, H, X, lambda, eta, 2, 200 );
%     ObjTh = [ObjTh, norm(X - W*Th*H,'fro')^2];
    ObjThLL = [ObjThLL, ll];
    norm(X - W*Th*H,'fro')^2/size(X,2)
%     Obj = [Obj, norm(X - W*Th*H,'fro')^2/size(X,2)];
    Obj = [Obj, compute_f(X,W,Th,H,lambda,eta)];
end
end

