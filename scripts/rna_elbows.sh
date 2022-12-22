#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate py39


python src/rna_seq.py --seed $LSB_JOBINDEX



