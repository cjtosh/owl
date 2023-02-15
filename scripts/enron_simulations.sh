#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl

seed=$(($LSB_JOBINDEX + 100))

python simulations/logreg_simulations.py --seed $seed --dataset 'enron' --corr_type 'max'
python simulations/logreg_simulations.py --seed $seed --dataset 'enron' --corr_type 'rand'