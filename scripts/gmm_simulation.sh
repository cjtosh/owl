#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate py39

seed=$(($LSB_JOBINDEX + 100))

python src/gmm_simulation.py --seed $seed --corr_type 'max' --n 100 --stdv 0.25 --p 2
python src/gmm_simulation.py --seed $seed --corr_type 'rand' --n 100 --stdv 0.25 --p 2