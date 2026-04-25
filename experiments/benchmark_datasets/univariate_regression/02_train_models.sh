#!/bin/bash

DATASETS=(air auto energy power wine)

R=100
K=5

for dataset in "${DATASETS[@]}"
do
  for ((repeat=0; repeat<${R}; repeat++))
  do
    for ((fold=1; fold<=${K}; fold++))
    do
      echo "Running 02_train_models.py (regression) for dataset=$dataset, repeat=$repeat, fold=$fold"
      python 02_train_models.py \
        --dataset "$dataset" \
        --repeat "$repeat" \
        --fold "$fold"
    done
  done
done