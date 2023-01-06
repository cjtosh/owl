#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate py39

seed=$(($LSB_JOBINDEX + 100))

python src/gaussian_simulations.py --seed $seed --corr_type 'max' --n 200 --scale 10.0 --dim 2
python src/gaussian_simulations.py --seed $seed --corr_type 'rand' --n 200 --scale 10.0 --dim 2