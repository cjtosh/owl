#!/bin/bash

conda activate owl

for i in {1..50}
do
    seed=$(($i + 100))

    ## Logistic regression
    python simulations/logreg_simulations.py --seed $seed --dataset 'mnist' --corr_type 'max'
    python simulations/logreg_simulations.py --seed $seed --dataset 'mnist' --corr_type 'rand'

    python simulations/logreg_simulations.py --seed $seed --dataset 'random' --corr_type 'max'
    python simulations/logreg_simulations.py --seed $seed --dataset 'random' --corr_type 'rand'

    python simulations/logreg_simulations.py --seed $seed --dataset 'enron' --corr_type 'max'
    python simulations/logreg_simulations.py --seed $seed --dataset 'enron' --corr_type 'rand'


    ## Linear regression
    python simulations/linreg_simulations.py --seed $seed --dataset 'random' --corr_type 'max'
    python simulations/linreg_simulations.py --seed $seed --dataset 'random' --corr_type 'rand'

    python simulations/linreg_simulations.py --seed $seed --dataset 'qsar' --corr_type 'max'
    python simulations/linreg_simulations.py --seed $seed --dataset 'qsar' --corr_type 'rand'


    ## Clustering
    python simulations/gmm_simulation.py --seed $seed --corr_type 'max'
    python simulations/gmm_simulation.py --seed $seed --corr_type 'rand'

    python simulations/bmm_simulation.py --seed $seed --corr_type 'max' --dataset 'simul'
    python simulations/bmm_simulation.py --seed $seed --corr_type 'rand' --dataset 'simul'


    ## Gaussian
    python simulations/gaussian_simulations.py --seed $seed --corr_type 'max' --n 200 --scale 10.0 --dim 2
    python simulations/gaussian_simulations.py --seed $seed --corr_type 'rand' --n 200 --scale 10.0 --dim 2

    python simulations/gaussian_simulations.py --seed $seed --corr_type 'max' --n 200 --scale 10.0 --dim 25
    python simulations/gaussian_simulations.py --seed $seed --corr_type 'rand' --n 200 --scale 10.0 --dim 25

    python simulations/gaussian_simulations.py --seed $seed --corr_type 'max' --n 200 --scale 10.0 --dim 50
    python simulations/gaussian_simulations.py --seed $seed --corr_type 'rand' --n 200 --scale 10.0 --dim 50
done 

## RNA experiments
for i in {1..150}
do
    python simulations/rna_seq.py --seed $i
done 


## Microcredit experiments
python simulations/microcredit_initial.py

for i in {1..2500}
do
    python simulations/microcredit_boot.py --i $i
done

python simulations/plots.py