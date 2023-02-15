#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl

seed=$(($LSB_JOBINDEX + 100))

python simulations/gaussian_simulations.py --seed $seed --corr_type 'max' --n 200 --scale 10.0 --dim 25
python simulations/gaussian_simulations.py --seed $seed --corr_type 'rand' --n 200 --scale 10.0 --dim 25