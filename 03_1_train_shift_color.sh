#!/bin/bash

exp_seed=132

exp_dir="./results/shift_color/trained_models"
folds=1
runs=10
models=2
ratios="100-0 0-100"
dataset="CelebA"
network="ResNet"
task="classification"

shift="RGB"

for ratio in $ratios; do
    for model in $(seq 1 $models); do
        for fold in $(seq 1 $folds); do
            for run in $(seq 1 $runs); do
                output_dir=$exp_dir"/"$shift"/"$ratio"/model_"$model"/fold_"$fold"/run_"$run
                seed=$(($exp_seed+$fold$run))
                python -m scripts.train_shift_color \
                    --dataset $dataset \
                    --task $task \
                    --network $network \
                    --shift $shift \
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

