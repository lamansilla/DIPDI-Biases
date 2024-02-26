#!/bin/bash

## UTKFace dataset

# Male
for fold in 0 1 2 3 4; do
    python -m scripts.models.utkface.test_dipdi_shift --fold $fold --shift "male"
done

# Female
 for fold in 0 1 2 3 4; do
     python -m scripts.models.utkface.test_dipdi_shift --fold $fold --shift "female"
 done
