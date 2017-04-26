function [ val ] = compute_f( X,W,Th,H,lambda,eta )
W_col_diff = 0;
for i = 1:size(W,2)-1
    W_col_diff = W_col_diff + norm(W(:,i) - W(:,i+1))^2;
end
val = (norm(X-W*Th*H,'fro')^2)/size(X,2) + lambda*sum(sum(abs(Th))) + eta*W_col_diff;


end

