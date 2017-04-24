#!/bin/bash

for i in {1..7}
do
    echo "submitting $i"
    qsub -v idx="$i" ./qsub-run.sh
done
