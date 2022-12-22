#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate py39

seed=$(($LSB_JOBINDEX + 100))

python src/logreg_simulations.py --seed $seed --dataset 'mnist' --corr_type 'max'
python src/logreg_simulations.py --seed $seed --dataset 'mnist' --corr_type 'rand'