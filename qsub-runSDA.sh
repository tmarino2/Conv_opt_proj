#!/bin/bash

#$ -o /export/a05/mfran/convex_results/outputs
#$ -e /export/a05/mfran/convex_results/outputs
#$ -pe smp 8
#$ -l 'h=b*'

cd ~/proj/convex-proj

echo "running sda $lambda $eta $iters"
matlab -nodisplay -nosplash  << EOF
maxNumCompThreads(8);
load('data1.mat');
W0 = randn(rows,rows);
W0 = W0.*(W0 > 0);
H0 = randn(rows,cols);
H0 = H0.*(H0 > 0);
Th0 = diag(randn(rows,1));
Th0 = Th0.*(Th0 > 0);
disp('matlab running $alpha $lambda $eta');
[ Obj, Ws, Ths, Hs, ObjWLL, ObjThLL, ObjHLL] = alt_min_sda( X, W0, Th0, H0, $lambda, $eta, $iters );
exp_name = sprintf('sda_l_%0.2f_e_%0.2f_it_%0.2f.mat', $lambda, $eta, $iters);
save(exp_name);
exit;
EOF
