#!/bin/bash

## ChestX-ray14 dataset

# Male
for fold in 0 1 2 3 4; do
    python -m scripts.models.chestxray14.train_test_full --fold $fold --option "male"
done

# Female
 for fold in 0 1 2 3 4; do
     python -m scripts.models.chestxray14.train_test_full --fold $fold --option "female"
 done

# Mixed
for fold in 0 1 2 3 4; do
   python -m scripts.models.chestxray14.train_test_full --fold $fold --option "mixed"
done


## UTKFace dataset

# Male
for fold in 0 1 2 3 4; do
    python -m scripts.models.utkface.train_test_full --fold $fold --option "male"
done

# Female
 for fold in 0 1 2 3 4; do
     python -m scripts.models.utkface.train_test_full --fold $fold --option "female"
 done

# Mixed
for fold in 0 1 2 3 4; do
   python -m scripts.models.utkface.train_test_full --fold $fold --option "mixed"
done