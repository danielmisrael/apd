#!/bin/bash

# Table 2

CUDA_DEVICE=0
LIMIT=2

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias dream --task gsm8k --alg origin --num_steps 256 --limit $LIMIT
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias dream --task gsm8k --alg entropy --num_steps 128 --limit $LIMIT
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias dream --task gsm8k --alg entropy --num_steps 256 --limit $LIMIT
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias dream --task gsm8k --alg leftright --tokens_per_step 1 --limit $LIMIT

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias llada --task gsm8k --alg random --num_steps 256 --limit $LIMIT
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias llada --task gsm8k --alg low_confidence --num_steps 128 --limit $LIMIT
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias llada --task gsm8k --alg low_confidence --num_steps 256 --limit $LIMIT
CUDA_VISIBLE_DEVICES=$CUDA_DEVICE  python eval.py --model_alias llada --task gsm8k --alg leftright --tokens_per_step 1 --limit $LIMIT