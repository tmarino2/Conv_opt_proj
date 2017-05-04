lambdas=[0.01,0.1,0.5,1,10];
etas=[0.01,0.1,0.5,1]; % not plotting 10

lambdasv = [];
for lambda=lambdas
    eta=0.01;
    exp_name = sprintf('sda_l_%0.2f_e_%0.2f.mat', lambda, eta);
    load(exp_name);
    O = [];
    for i=1:length(ObjThLL)
        O = [O, ObjThLL{i}(end)];
    end
    lambdasv = [lambdasv; O];
end
plot(lambdasv');
title('SDA, changing lambda');
legend('0.01','0.1','0.5','10');
print('sda-lambda.png', '-dpng');

etasv = [];
for eta=etas
    lambda=0.01;
    exp_name = sprintf('sda_l_%0.2f_e_%0.2f.mat', lambda, eta);
    load(exp_name);
    O = [];
    for i=1:length(ObjThLL)
        O = [O, ObjThLL{i}(end)];
    end
    etasv = [etasv; O];
end
plot(etasv');
title('SDA, changing eta');
legend('0.01','0.1','0.5','1');
print('sda-eta.png', '-dpng');

load('sda_l_0.01_e_0.01.mat');
plot_per_group(ObjWLL);
title('Internal optimization steps of W for alpha=.001 lambda=.01 eta=.01');
print('sda-internal-W.png', '-dpng');

plot_per_group(ObjHLL);
title('Internal optimization steps of H for alpha=.001 lambda=.01 eta=.01');
print('sda-internal-H.png', '-dpng');

plot_per_group(ObjThLL);
title('Internal optimization steps of Theta for alpha=.001 lambda=.01 eta=.01');
print('sda-internal-Th.png', '-dpng');
