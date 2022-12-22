#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate py39

seed=$(($LSB_JOBINDEX + 100))

python src/linreg_simulations.py --seed $seed --dataset 'random' --corr_type 'max'
python src/linreg_simulations.py --seed $seed --dataset 'random' --corr_type 'rand'