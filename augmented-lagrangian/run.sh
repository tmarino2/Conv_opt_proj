
	#!/bin/bash

for lambda in 0.01
do
    for eta in 0.01 0.1 0.5 1
    do
	echo lambda=${lambda}, eta=${eta}
	python opt.py --X ../data1.mat --log lambda=${lambda},eta=${eta}.log2 --L ${lambda} --eta ${eta} &
    done
    for eta in 10
    do
	echo lambda=${lambda}, eta=${eta}
	python opt.py --X ../data1.mat --log lambda=${lambda},eta=${eta}.log2 --L ${lambda} --eta ${eta} 
    done
done
