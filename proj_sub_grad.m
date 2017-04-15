function [ x_k ] = proj_sub_grad( x_k, grad_x_k, alpha_k, A, B, X, lambda, eta, var )
%Do a projected subgradient onto the positive cone
%A and B are the fixed matrices and x_k is the matrix over which we
%optimize, grad_x_k is the initial gradient with respect to x_k, lambda is
%the regularization parameter for l1 penalty, eta is the penalty parameter 
%for smoothness constraint, var just specifies over which variable we are
%minimizing i.e. var == 1 we minimize over W, var == 2 over Th, var == 3
%over H
max_iters = 300;
eps = 0.0001;
k = 1;
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
        x_k = x_k - alpha_k*grad_x_k/k;
        x_k = x_k.*(x_k>0);
        k = k+1;
    end
%     if var == 1
%         norm(X - x_k*A*B,'fro')^2;
%     elseif var == 2
%         nnz(x_k);
%         norm(X-A*x_k*B,'fro')^2
%     else
%         norm(X-A*B*x_k,'fro')^2
%     end
end


end

