#!/bin/bash

for alpha in 0.001 0.05 0.1 0.5
do
    for lambda in 0.01 0.1 0.5 1 10
    do
        for eta in 0.01 0.1 0.5 1 10
        do
            echo "submitting $alpha $lambda $eta"
            qsub -v alpha="$alpha",lambda="$lambda",eta="$eta" ./qsub-run3.sh
        done
    done
done
