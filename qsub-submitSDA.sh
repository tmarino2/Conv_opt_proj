#!/bin/bash

for lambda in 0.01 0.1 0.5 1 10
do
    for eta in 0.01 0.1 0.5 1 10
    do
        echo "submitting SDA $lambda $eta"
        qsub -v lambda="$lambda",eta="$eta" ./qsub-runSDA.sh
    done
done
