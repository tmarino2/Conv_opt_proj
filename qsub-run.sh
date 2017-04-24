#!/bin/bash

#$ -o /export/a05/mfran/convex_results/outputs
#$ -e /export/a05/mfran/convex_results/outputs


cd ~/proj/convex-proj

echo "running $idx"
matlab -nodisplay -nosplash -r "expr_all($idx); exit"
