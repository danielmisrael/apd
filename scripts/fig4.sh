#!/bin/bash

# Figure 4

CUDA_DEVICE=3
LIMIT=500

for m_value in 256 128 64 32 16 8 4
do
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias dream --task gsm8k --alg apd --max_lookahead $m_value --limit $LIMIT
done