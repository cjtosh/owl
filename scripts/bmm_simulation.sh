#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl

seed=$(($LSB_JOBINDEX + 100))

python src/bmm_simulation.py --seed $seed --corr_type 'max' --dataset 'simul'
python src/bmm_simulation.py --seed $seed --corr_type 'rand' --dataset 'simul'