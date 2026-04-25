#!/bin/bash

for dataset in air auto energy power wine
do
  echo "Running 04_sensitivity_analysis.py (univariate regression) for dataset=$dataset"
  python 04_sensitivity_analysis.py --dataset "$dataset"
done