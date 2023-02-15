#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl

seed=$(($LSB_JOBINDEX + 100))

python simulations/linreg_simulations.py --seed $seed --dataset 'qsar' --corr_type 'max'
python simulations/linreg_simulations.py --seed $seed --dataset 'qsar' --corr_type 'rand'