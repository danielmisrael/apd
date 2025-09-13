#!/bin/bash

# Figure 2

CUDA_DEVICE=0
LIMIT=500

for tokens_per_step in {1..8}
do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias dream --task gsm8k --alg leftright --tokens_per_step $tokens_per_step --limit $LIMIT
done

for tokens_per_step in {1..8}
do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias llada --task gsm8k --alg leftright --tokens_per_step $tokens_per_step --limit $LIMIT
done