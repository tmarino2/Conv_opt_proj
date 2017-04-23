function [ x_k_hat , objs ] = SDA( x_k, grad_x_k, A, B, X, lambda, eta, var, max_iters )
%Alg3 in lecture 4
eps = 0.0001;
k = 1;
objs = [];
x_0 = x_k;
x_k_hat = x_k;
s_k = grad_x_k;
betas = [1,1];
while norm(grad_x_k)>eps && k <= max_iters
    if var == 1
        [ grad_x_k,~,~ ] = sub_grads( x_k ,A ,B , X, lambda, eta );
    elseif var == 2
       [ ~,grad_x_k,~ ] = sub_grads( A ,x_k ,B , X, lambda, eta );
    else
        [ ~,~,grad_x_k ] = sub_grads( A ,B ,x_k , X, lambda, eta );
    end
    if norm(grad_x_k)<=eps
        break
    else
        s_k = s_k + grad_x_k;
        beta_k = sum(betas);
        betas = [betas, 1/beta_k];
        x_k_tilde = x_0-s_k/beta_k;
        x_k = x_k_tilde.*(x_k_tilde>0);
        x_k_hat = x_k_hat*k/(k+1)+x_k/(k+1);
        k = k+1;
    end
    if mod(k, 10) == 0
        if var == 1
             ll = norm(X - x_k_hat*A*B,'fro')^2/size(X,2);
        elseif var == 2
             ll = norm(X-A*x_k_hat*B,'fro')^2/size(X,2);
        else
             ll = norm(X-A*B*x_k_hat,'fro')^2/size(X,2);
        end
    objs = [objs, ll];
    end
    
end
end

