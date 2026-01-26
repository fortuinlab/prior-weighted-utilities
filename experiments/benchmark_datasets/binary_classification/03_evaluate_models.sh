#!/bin/bash

for dataset in bank heartdisease ionosphere mushroom sonar
do
  echo "Running 03_evaluate_models.py (binary classification) for dataset=$dataset"
  python 03_evaluate_models.py --dataset "$dataset"
done