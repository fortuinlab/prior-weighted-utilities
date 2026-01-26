#!/bin/bash

R=100
K=5


for ((repeat=0; repeat<${R}; repeat++))
do
  for ((fold=1; fold<=${K}; fold++))
  do
    echo "Running 02_train_models.py (p2p lending) for repeat=$repeat, fold=$fold"
    python 02_train_models.py \
      --repeat "$repeat" \
      --fold "$fold"
  done
done
