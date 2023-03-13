#!/bin/bash

source /home/toshc/anaconda3/bin/activate
conda activate owl


python simulations/rna_seq.py --i $LSB_JOBINDEX
