function [ Obj, W, Th, H, ObjW, ObjTh, ObjWLL, ObjThLL, ObjHLL] = alt_min( X, W, Th, H, alpha, lambda, eta )
Obj = [norm(X - W*Th*H,'fro')^2];
ObjW = [norm(X - W*Th*H,'fro')^2];
ObjTh = [norm(X - W*Th*H,'fro')^2];
%have better stopping criterion
max_iters = 500; %100;
[ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta );  % this is just used to create some filler space
ObjWLL = [];
ObjThLL = [];
ObjHLL = [];
for i = 1:max_iters
    [W, ll] = proj_sub_grad( W, grW, alpha, Th, H, X, lambda, eta, 1, 200);
%     ObjW = [ObjW, norm(X - W*Th*H,'fro')^2];
    ObjWLL = [ObjWLL; ll];
    [H, ll]= proj_sub_grad( H, grH, alpha, W, Th, X, lambda, eta, 3, 200 );  
    ObjHLL = [ObjHLL; ll];
    [Th, ll] = proj_sub_grad( Th, grTh, 0.00001*alpha, W, H, X, lambda, eta, 2, 200 );
%     ObjTh = [ObjTh, norm(X - W*Th*H,'fro')^2];
    ObjThLL = [ObjThLL; ll];
    Obj = [Obj, norm(X - W*Th*H,'fro')^2]
end
end

