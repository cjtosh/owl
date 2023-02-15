#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl

seed=$(($LSB_JOBINDEX + 100))

python src/logreg_simulations.py --seed $seed --dataset 'random' --corr_type 'max'
python src/logreg_simulations.py --seed $seed --dataset 'random' --corr_type 'rand'