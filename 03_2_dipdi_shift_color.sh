#!/bin/bash

exp_seed=132

folds=1
runs=10
models=4
ratios_A="100-0"
ratios_B="0-100"
dataset="CelebA"
network="ResNet"
task="classification"

# shift: RGB, Grayscale
shift="RGB"

output_dir="./results/shift_color/dipdi"
ckpt_dir="./results/shift_color/trained_models/RGB"

python -m scripts.dipdi_shift_color \
    --dataset $dataset \
    --network $network \
    --task $task \
    --shift $shift \
    --output_dir $output_dir \
    --ckpt_dir $ckpt_dir \
    --ratios_A $ratios_A \
    --ratios_B $ratios_B \
    --n_folds $folds \
    --n_runs $runs \
    --n_models $models

