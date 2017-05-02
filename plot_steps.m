% for alpha=[0.001,0.05,0.1,0.5]
%     for lambda=[0.01,0.1,0.5,1,10]
%         for eta = w[0.01,0.1,0.5,1,10]
% %             exp_name = sprintf('step_%0.5f_l_%0.2f_e_%0.2f.mat',alpha,lambda,eta);
% %             load(exp_name);
% %             
% %             plot_per_group(ObjThLL);
% %             plot_per_group(ObjWLL);
% %             plot_per_group(ObjHLL);
%         end
%     end
% end


% generate the alpha plot
max_length = 0;
alphas=[0.001,0.05,0.1,0.5];
alphasv=[];
for alpha=alphas
    lambda = .01;
    eta = .01;
    exp_name = sprintf('step_%0.5f_l_%0.2f_e_%0.2f.mat',alpha,lambda,eta);
    load(exp_name);
    alphasv = [alphasv; Obj];
end
plot(alphasv');
title('Fixed step size, changing alpha');
legend('.001', '.05', '.1', '.5');
print('fixed-step-alpha.png', '-dpng');

% generate the lambda plot
max_length = 0;
lambdas=[0.01,0.1,0.5,1,10];
lambdasv=[];
for lambda=lambdas
    alpha=0.001;
    eta=.01;
    exp_name = sprintf('step_%0.5f_l_%0.2f_e_%0.2f.mat',alpha,lambda,eta);
    load(exp_name);
    lambdasv= [lambdasv; Obj];
end
plot(lambdasv');
title('Fixed step size, chnaging lambda');
legend('.01','.1','.5','1','10');
print('fixed-step-lambda.png', '-dpng');


etas=[0.01,0.1,0.5,1,10];
etasv=[];
for eta=etas
    alpha=0.001;
    lambda=0.01;
    exp_name = sprintf('step_%0.5f_l_%0.2f_e_%0.2f.mat',alpha,lambda,eta);
    load(exp_name);
    etasv=[etasv;Obj];
end
plot(etasv');
title('Fixed step size, changing eta');
legend('0.01','0.1','0.5','1','10');
print('fixed-step-eta.png', '-dpng');