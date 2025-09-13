#!/bin/bash

# Figure 6

CUDA_DEVICE=0
LIMIT=500
MATH_LIMIT=100 # The math dataset has many subtasks

for task in gsm8k gpqa #math humaneval
do
    # Set the appropriate limit based on the task
    if [ "$task" = "math" ]; then
        CURRENT_LIMIT=$MATH_LIMIT
    else
        CURRENT_LIMIT=$LIMIT
    fi
    
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias qwen7b --task $task --alg leftright --limit $CURRENT_LIMIT
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias qwen0.5b --task $task --alg leftright --limit $CURRENT_LIMIT
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias dream --task $task --alg leftright --tokens_per_step 1 --limit $CURRENT_LIMIT
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias dream --task $task --alg apd --apd_mixture_weight 0.6 --kv_window 32 --max_lookahead 200 --limit $CURRENT_LIMIT
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval.py --model_alias dream --task $task --alg apd --apd_mixture_weight 0.7 --kv_window 16 --max_lookahead 100 --limit $CURRENT_LIMIT
done

