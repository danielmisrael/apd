#!/bin/bash

# Figure 3

CUDA_DEVICE=0
LIMIT=500

for r_value in $(LC_NUMERIC=C seq 0.1 0.1 0.8)
do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias dream --task gsm8k --alg apd --apd_mixture_weight $r_value --limit $LIMIT
done
