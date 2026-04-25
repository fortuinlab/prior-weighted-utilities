#!/bin/bash

for dataset in air auto energy power wine
do
  echo "Running 03_evaluate_models.py (regression) for dataset=$dataset"
  python 03_evaluate_models.py --dataset "$dataset"
done