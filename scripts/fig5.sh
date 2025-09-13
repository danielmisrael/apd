#!/bin/bash

# Figure 5

CUDA_DEVICE=0
LIMIT=500

for w_value in 0 1 4 16 64 128 256
do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias dream --task gsm8k --alg apd --kv_window $w_value --limit $LIMIT
done