function [ x_k , objs ] = proj_sub_grad2( x_k, alpha_k, A, B, X, lambda, eta, var, max_iters )
%Do a projected subgradient onto the positive cone
%A and B are the fixed matrices and x_k is the matrix over which we
%optimize, grad_x_k is the initial gradient with respect to x_k, lambda is
%the regularization parameter for l1 penalty, eta is the penalty parameter 
%for smoothness constraint, var just specifies over which variable we are
%minimizing i.e. var == 1 we minimize over W, var == 2 over Th, var == 3
%over H
% max_iters = 500;
eps = 0.00001;
k = 1;
objs = [];
while k <= max_iters
    if var == 1
        [ grad_x_k,~,~ ] = sub_grads( x_k ,A ,B , X, lambda, eta );
        x_temp = x_k;
        x_k = x_k - alpha_k*grad_x_k;
        x_k = x_k.*(x_k>0);
%         while compute_f(X,x_k,A,B,lambda,eta) > compute_f(X,x_temp,A,B,lambda,eta) + trace(grad_x_k'*(x_k-x_temp)) + (1/(2*alpha_k))*norm(x_k-x_temp,'fro')^2
%             alpha_k = 0.5*alpha_k;
%         end
        if max(max(abs(x_temp-x_k))) < eps %even though this is very bad for the Th update we have nothing better
            disp('good W')
            break;
        end
        k = k+1;        
    elseif var == 2
        [ ~,grad_x_k,~ ] = sub_grads( A ,x_k ,B , X, lambda, eta );
        x_temp = x_k;
        x_k = x_k - alpha_k*grad_x_k;
        x_k = x_k.*(x_k>0);
        if max(max(abs(x_temp-x_k))) < eps %even though this is very bad for the Th update we have nothing better
            disp('good Th') 
            break;
        end
        k = k+1;       
    else
        [ ~,~,grad_x_k ] = sub_grads( A ,B ,x_k , X, lambda, eta );
        x_temp = x_k;
        x_k = x_k - alpha_k*grad_x_k;
        x_k = x_k.*(x_k>0);
%         while compute_f(X,A,B,x_k,lambda,eta) > compute_f(X,A,B,x_temp,lambda,eta) + trace(grad_x_k'*(x_k-x_temp)) + (1/(2*alpha_k))*norm(x_k-x_temp,'fro')^2
%             alpha_k = 0.5*alpha_k;
%         end
        if max(max(abs(x_temp-x_k))) < eps %even though this is very bad for the Th update we have nothing better
            disp('good H')
            break;
        end
        k = k+1;
    end
%     x_temp = x_k;
%     %         x_k = x_k - alpha_k*grad_x_k/k;
%     x_k = x_k - alpha_k*grad_x_k;
%     x_k = x_k.*(x_k>0);
%     if max(max(abs(x_temp-x_k))) < eps %even though this is very bad for the Th update we have nothing better
%        break;
%     end
%     k = k+1;
    if mod(k, 10) == 0
        if var == 1
%              ll = norm(X - x_k*A*B,'fro')^2/size(X,2);
            ll = compute_f(X,x_k,A,B,lambda,eta);
        elseif var == 2
%              ll = norm(X-A*x_k*B,'fro')^2/size(X,2);
            ll = compute_f(X,A,x_k,B,lambda,eta);
        else
%              ll = norm(X-A*B*x_k,'fro')^2/size(X,2);
            ll = compute_f(X,A,B,x_k,lambda,eta);
        end
    objs = [objs, ll];
    end
    
end
end

