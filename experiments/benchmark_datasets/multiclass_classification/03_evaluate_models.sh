#!/bin/bash

for dataset in covertype drybean iris pendigits wine
do
  echo "Running 03_evaluate_models.py (multiclass classification) for dataset=$dataset"
  python 03_evaluate_models.py --dataset "$dataset"
done