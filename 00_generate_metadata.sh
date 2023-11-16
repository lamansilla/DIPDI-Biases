#!/bin/bash

# ChestX-ray14 dataset
python -m scripts.data.generate_metadata --dataset chestxray14

# UTKFace dataset
python -m scripts.data.generate_metadata --dataset utkface