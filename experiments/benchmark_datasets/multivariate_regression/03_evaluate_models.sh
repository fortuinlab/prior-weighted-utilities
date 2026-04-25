#!/bin/bash

for dataset in air energy flare sgemm parkinsons
do
  echo "Running 03_evaluate_models.py (multivariate regression) for dataset=$dataset"
  python 03_evaluate_models.py --dataset "$dataset"
done