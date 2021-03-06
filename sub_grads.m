function [ grW,grTh,grH ] = sub_grads( W ,Th ,H , X, lambda, eta )
%Compute the subgradients with respect to W, Theta and H
tilde_W = zeros(size(W));
for j = 1:size(W,2)
    tilde_W(1,j) = 2*(W(1,j) - W(2,j));
    tilde_W(size(W,1),j) = 2*(W(size(W,1),j) - W(size(W,1)-1,j));
    for i = 2:size(W,1)-1
        tilde_W(i,j) = 2*(2*W(i,j) - (W(i+1,j) + W(i-1,j)));
    end
end
grW = 2*(W*Th*H - X)*(Th*H)'/size(X,2) + eta*tilde_W;
% grTh = (2*W'*(W*Th*H - X)*H' + lambda*(eye(size(Th,1)).*Th>zer)).*eye(size(Th,1));
grTh = (2*W'*(W*Th*H - X)*H'/size(X,2) + lambda*(Th>0)).*eye(size(Th));
grH = 2*(W*Th)'*(W*Th*H - X)/size(X,2);
norm(grW,'fro')
% grW = grW/size(grH,2);
% grTh = grTh/size(grH,2);
% grH = grH/size(grH,2);
end

