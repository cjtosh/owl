#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl


python simulations/microcredit.py --i $LSB_JOBINDEX
