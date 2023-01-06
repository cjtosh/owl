#!/bin/bash

# rm output_logs/*


## Logistic
bsub -J 'rand[1-50]' -W 1:00 -n 1 -R "rusage[mem=5]" -e output_logs/rand_%I.err -o output_logs/rand_%I.out sh scripts/randlr_simulations.sh

bsub -J 'mnist[1-50]' -W 1:30 -n 1 -R "rusage[mem=5]" -e output_logs/mnist_%I.err -o output_logs/mnist_%I.out sh scripts/mnist_simulations.sh

bsub -J 'enron[1-50]' -W 1:30 -n 1 -R "rusage[mem=5]" -e output_logs/enron_%I.err -o output_logs/enron_%I.out sh scripts/enron_simulations.sh


## Linear

bsub -J 'qsar[1-50]' -W 2:00 -n 3 -e output_logs/qsar_%I.err -o output_logs/qsar_%I.out sh scripts/qsarlin_simulations.sh

bsub -J 'randlin[1-50]' -W 1:00 -n 1 -R "rusage[mem=5]" -e output_logs/randlin_%I.err -o output_logs/randlin_%I.out sh scripts/randlin_simulations.sh


## Clustering
bsub -J 'gmm[1-50]' -W 5:00 -n 3 -e output_logs/gmm_%I.err -o output_logs/gmm_%I.out sh scripts/gmm_simulation.sh

# bsub -J 'bmm[1-50]' -W 1:30 -n 3 -e output_logs/bmm_%I.err -o output_logs/bmm_%I.out sh scripts/bmm_simulation.sh

bsub -J 'rna[1-270]' -W 1:00 -e output_logs/rna_%I.err -o output_logs/rna_%I.out sh scripts/rna_elbows.sh

## Gaussian

bsub -J 'gaussian[1-50]' -W 1:00 -n 2 -e output_logs/gaussian_%I.err -o output_logs/gaussian_%I.out sh scripts/gaussian_simulation.sh

bsub -J 'gaussian_h[1-50]' -W 1:00 -n 2 -e output_logs/gaussian_h%I.err -o output_logs/gaussian_h%I.out sh scripts/gaussian_simulation_high.sh
