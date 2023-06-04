#!/bin/bash
eval "$(conda shell.bash hook)"

conda activate FCOS
# proxy_on

python train_voc.py --n_cpu=8 --n_gpu=0 --batch_size=8