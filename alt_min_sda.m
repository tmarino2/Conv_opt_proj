function [ Obj, W, Th, H, ObjWLL, ObjThLL, ObjHLL] = alt_min_sda( X, W, Th, H, lambda, eta )
Obj = [norm(X - W*Th*H,'fro')^2];
%have better stopping criterion
max_iters = 250; %100;
[ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta );  % this is just used to create some filler space
ObjWLL = [];
ObjThLL = [];
ObjHLL = [];
for i = 1:max_iters
    [W, ll] = SDA(W, grW, Th, H, X, lambda, eta, 1, 200);
    ObjWLL = [ObjWLL; ll];
    [H, ll]= SDA( H, grH, W, Th, X, lambda, eta, 3, 200 );  
    ObjHLL = [ObjHLL; ll];
    [Th, ll] = SDA( Th, grTh, W, H, X, lambda, eta, 2, 200 );
    ObjThLL = [ObjThLL; ll];
    Obj = [Obj, norm(X - W*Th*H,'fro')^2]
end
end

