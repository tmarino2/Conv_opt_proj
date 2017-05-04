function [ Obj, W, Th, H, ObjWLL, ObjThLL, ObjHLL] = alt_min_sda( X, W, Th, H, lambda, eta, inner_iters)
Obj = [norm(X - W*Th*H,'fro')^2];
%have better stopping criterion
max_iters = 250; %100;
[ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta );  % this is just used to create some filler space
ObjWLL = {1,max_iters};
ObjThLL = {1,max_iters};
ObjHLL = {1,max_iters};
for i = 1:max_iters
<<<<<<< HEAD
    [W, ll] = SDA(W, grW, Th, H, X, lambda, eta, 1, 300);
    ObjWLL = [ObjWLL; ll];
    [H, ll]= SDA( H, grH, W, Th, X, lambda, eta, 3, 300 );  
    ObjHLL = [ObjHLL; ll];
    [Th, ll] = SDA( Th, grTh, W, H, X, lambda, eta, 2, 300 );
    ObjThLL = [ObjThLL; ll];
    Obj = [Obj, norm(X - W*Th*H,'fro')^2/size(X,2)]
=======
    [W, ll] = SDA(W, grW, Th, H, X, lambda, eta, 1, inner_iters);
    ObjWLL{i} = ll;
    [H, ll]= SDA( H, grH, W, Th, X, lambda, eta, 3, inner_iters);  
    ObjHLL{i} = ll;
    [Th, ll] = SDA( Th, grTh, W, H, X, lambda, eta, 2, inner_iters);
    ObjThLL{i} =  ll;
>>>>>>> 99b596ec48184271a496f3148f48ce602c47e82b
end
end

