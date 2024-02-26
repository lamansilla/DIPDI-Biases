#!/bin/bash

exp_seed=132

exp_dir="./results/ratios_group/trained_models"
folds=10
runs=1
models=4
ratios="100-0 75-25 50-50 25-75 0-100"

# ChestX-ray14: DenseNet, regression
# UTKFace: VGG, regression
# CelebA: ResNet, classification

dataset="CelebA"
network="ResNet"
task="classification"

for ratio in $ratios; do
    for model in $(seq 1 $models); do
        for fold in $(seq 1 $folds); do
            for run in $(seq 1 $runs); do
                output_dir=$exp_dir"/"$dataset"/"$ratio"/model_"$model"/fold_"$fold"/run_"$run
                seed=$(($exp_seed+$fold$run))
                python -m scripts.train_group_ratios \
                    --dataset $dataset \
                    --network $network \
                    --task $task \
                    --output_dir $output_dir \
                    --seed $seed \
                    --ratio $ratio \
                    --fold $fold \
                    --n_models=$models \
                    --model $model
            done
        done
    done
done