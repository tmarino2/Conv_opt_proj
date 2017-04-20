function [ x_k, objs ] = aug_lagr( x_k, A, B, X, lambda, eta, tau, gamma, var, max_iters )
%augmented lagrangian
%need to implement diagonal constraint for var==2 (i.e Theta case)
mu_k = randn(size(x_k));
mu_k = mu_k.*(mu_k>0);
k = 1;
rho_k = rand(1);
objs = [];
V_k = zeros(size(x_k));
while k<=max_iters
    if var == 1
        lagr = @(x)norm(X - x*A*B,'fro')^2/size(X,2) + eta*norm(x(:,1:end-1)...
            - x(:,2:end),'fro')^2 + norm((mu_k/rho_k-x).*((mu_k/rho_k-x)>0),'fro')^2;
        [x_k,~] = fminunc(lagr,x_k)
        mu_k = (mu_k - rho_k*x_k).*((mu_k - rho_k*x_k)>0);
        if k == 1
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = (1-rho_k)*rand(1) + rho_k;
        elseif tau*norm(x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k),'fro') >= norm(V_k,'fro')
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = (1-rho_k)*rand(1) + rho_k;
        else
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = gamma*rho_k;
        end
    elseif var == 2
        lagr = @(x)norm(X - A*x*B,'fro')^2/size(X,2) + lambda*(sum(sum(abs(x)))) + norm((mu_k/rho_k-x).*((mu_k/rho_k-x)>0),'fro')^2;
        [x_k,~] = fminunc(lagr,x_k);
        mu_k = (mu_k - rho_k*x_k).*((mu_k - rho_k*x_k)>0);
        if k == 1
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = (1-rho_k)*rand(1) + rho_k;
        elseif tau*norm(x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k),'fro') >= norm(V_k,'fro')
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = (1-rho_k)*rand(1) + rho_k;
        else
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = gamma*rho_k;
        end
    else
        lagr = @(x)norm(X - A*x*B,'fro')^2/size(X,2) + norm((mu_k/rho_k-x).*((mu_k/rho_k-x)>0),'fro')^2;
        [x_k,~] = fminunc(lagr,x_k);
        mu_k = (mu_k - rho_k*x_k).*((mu_k - rho_k*x_k)>0);
        if k == 1
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = (1-rho_k)*rand(1) + rho_k;
        elseif tau*norm(x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k),'fro') >= norm(V_k,'fro')
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = (1-rho_k)*rand(1) + rho_k;
        else
            V_k = x_k.*(x_k < mu_k/rho_k) + (mu_k/rho_k).*(x_k >= mu_k/rho_k);
            rho_k = gamma*rho_k;
        end
    end
    k = k+1;
    if mod(k, 10) == 0
        if var == 1
             ll = norm(X - x_k_hat*A*B,'fro')^2;
        elseif var == 2
             ll = norm(X-A*x_k_hat*B,'fro')^2;
        else
             ll = norm(X-A*B*x_k_hat,'fro')^2;
        end
    objs = [objs, ll]
    end
end
end
