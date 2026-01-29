#!/bin/bash

for dataset in bank heartdisease ionosphere mushroom sonar
do
  echo "Running 04_sensitivity_analysis.py (binary classification) for dataset=$dataset"
  python 04_sensitivity_analysis.py --dataset "$dataset"
done