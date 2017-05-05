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
etas=[0.01,0.1,0.5,1,10];
lambdas=[0.01,0.1,0.5,1,10];

hold on;
lc = cell(64,1);
n = 1;
for alpha=alphas
    for lambda=lambdas
        for eta=etas            
            exp_name = sprintf('deck_step_%0.5f_l_%0.2f_e_%0.2f.mat',alpha,lambda,eta);
            load(exp_name);
            if max(Obj) < 1e3
            plot(min(Obj, 1e3));
            lc{n} = sprintf('Alpha=%0.5f Lambda=%0.2f Eta=%0.2f',alpha,lambda,eta);
            n = n + 1;
            end
        end
    end    
end

legend(lc{1:n-1});