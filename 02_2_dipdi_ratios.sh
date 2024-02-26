#!/bin/bash

exp_seed=132

folds=10
runs=1
groups=4
ratios_A="100-0 75-25 50-50 25-75 0-100"
ratios_B="0-100 25-75 50-50 75-25 100-0"

# ChestX-ray14: DenseNet, regression
# UTKFace: VGG, regression
# CelebA: ResNet, classification

dataset="CelebA"
network="ResNet"
task="classification"

output_dir="./results/ratios_group/dipdi"
ckpt_dir="./results/ratios_group/trained_models/"$dataset

python -m scripts.dipdi_ratios \
    --dataset $dataset \
    --network $network \
    --task $task \
    --output_dir $output_dir \
    --ckpt_dir $ckpt_dir \
    --ratios_A $ratios_A \
    --ratios_B $ratios_B \
    --n_folds $folds \
    --n_runs $runs \
    --n_groups $groups

