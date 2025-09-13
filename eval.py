import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"
import argparse
import json
import logging
import torch
from lm_eval import evaluator
from harness import DreamEvalHarness, ProfileEvalHarness, LladaEvalHarness
from utils import parse_results
from transformers import AutoModel, AutoTokenizer
from dream.modeling_dream import DreamModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply custom task configurations
from eval_config.monkey_patch import apply_custom_task_configs
apply_custom_task_configs()

# Define the tasks you want to support
TASKS = {'gsm8k': 'gsm8k', 'math': 'hendrycks_math', 'gpqa': 'gpqa_main_generative_n_shot', 'humaneval': 'humaneval_instruct'} # Use the exact task names from lm-eval task registry


def get_model(args):
    
    model_alias = args.model_alias
    alg = args.alg
    # Canonical names only
    tokens_per_step = args.tokens_per_step
    max_lookahead = args.max_lookahead
    kv_window = args.kv_window
    apd_mixture_weight = args.apd_mixture_weight
    num_steps = args.num_steps
    task_name = args.task

    logger.info(f"Configuring model details for alias: {model_alias}")

    if model_alias == "qwen7b":
        if args.qwen_7b_ckpt is None:
            raise ValueError("--qwen_7b_ckpt is required when --model_alias=qwen7b")
        model = ProfileEvalHarness(pretrained=args.qwen_7b_ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", max_length=16384)
    elif model_alias == "qwen0.5b":
        if args.qwen_small_ckpt is None:
            raise ValueError("--qwen_small_ckpt is required when --model_alias=qwen0.5b")
        model = ProfileEvalHarness(pretrained=args.qwen_small_ckpt, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", max_length=16384)
    elif model_alias == "dream":
        if args.dream_ckpt is None:
            raise ValueError("--dream_ckpt is required when --model_alias=dream")
        if args.alg == "apd" and args.qwen_small_ckpt is None:
            raise ValueError("--qwen_small_ckpt is required when --model_alias=dream and --alg=apd")
        dream = DreamModel.from_pretrained(args.dream_ckpt, 
                                               trust_remote_code=True,  
                                               attn_implementation="sdpa", 
                                               torch_dtype=torch.bfloat16, 
                                               device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
        model = DreamEvalHarness(
            pretrained=dream,
            tokenizer=tokenizer,
            alg=alg,
            tokens_per_step=tokens_per_step,
            max_lookahead=max_lookahead,
            kv_window=kv_window,
            apd_mixture_weight=apd_mixture_weight,
            num_steps=num_steps,
            max_gen_toks=512 if task_name=="math" else 256,
            verifier_ckpt=args.qwen_small_ckpt,
        )
        
    elif model_alias == "llada":
        if args.llada_ckpt is None:
            raise ValueError("--llada_ckpt is required when --model_alias=llada")
        llada = AutoModel.from_pretrained(args.llada_ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        tokenizer = AutoTokenizer.from_pretrained(args.llada_ckpt, trust_remote_code=True)
        model = LladaEvalHarness(pretrained=llada, tokenizer=tokenizer, alg=alg, tokens_per_step=tokens_per_step, num_steps=num_steps)

    else:
        raise ValueError(f"Unknown model alias: {model_alias}. Must be one of 'qwen7b', 'qwen0.5b', 'dream'.")

    return model

def main():
    parser = argparse.ArgumentParser(description="Evaluate language models using EleutherAI Eval Harness.")
    parser.add_argument(
        "--model_alias",
        required=True,
        choices=["qwen7b", "qwen0.5b", "dream", "llada"],
        help="Alias of the model to evaluate."
    )
    
    parser.add_argument(
        "--task",
        required=True,
        choices=["gsm8k", "math", "gpqa", "humaneval"],
        help="Please choose one of the following tasks: gsm8k, math, gpqa, humaneval"
    )
    parser.add_argument(
        "--output_dir", 
        default="results",
        help="Directory to save evaluation results json file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples per task for quick testing (e.g., 10 or 0.1 for 10%%)."
    )
    parser.add_argument(
        "--alg",
        choices=["leftright", "low_confidence", "random", "origin", "entropy", "apd"]
    )
    parser.add_argument("--qwen_small_ckpt", type=str, default="Qwen/Qwen2.5-0.5B", help="Checkpoint or repo id for Qwen (used by qwen0.5b and as APD verifier)")
    parser.add_argument("--qwen_7b_ckpt", type=str, default="Qwen/Qwen2.5-7B", help="Checkpoint or repo id for Qwen 7B (used by qwen7b)")
    parser.add_argument("--dream_ckpt", type=str, default="Dream-org/Dream-v0-Instruct-7B", help="Checkpoint or repo id for Dream model")
    parser.add_argument("--llada_ckpt", type=str, default="GSAI-ML/LLaDA-8B-Instruct", help="Checkpoint or repo id for LLaDA model")
    parser.add_argument("--tokens_per_step", type=int, default=None, help="The number of tokens to generate per step K")
    parser.add_argument("--kv_window", type=int, default=None, help="APD Parameter W")
    parser.add_argument("--max_lookahead", type=int, default=None, help="APD Parameter M")
    parser.add_argument("--apd_mixture_weight", type=float, default=None, help="APD Parameter R")
    
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--tag",
        type=str,
        default="",
    )

    args = parser.parse_args()
    
    # Validate algorithm choices based on model alias
    valid_algs = {
        "llada": ["low_confidence", "leftright", "random"],
        "dream": ["leftright", "apd", "entropy", "origin"],
        "qwen7b": ["leftright", None],
        "qwen0.5b": ["leftright", None]
    }
    
    if args.model_alias in valid_algs:
        if args.alg not in valid_algs[args.model_alias]:
            valid_alg_str = ", ".join([str(alg) for alg in valid_algs[args.model_alias]])
            raise ValueError(f"Invalid algorithm '{args.alg}' for model '{args.model_alias}'. "
                           f"Valid algorithms for {args.model_alias}: {valid_alg_str}")
    
    model = get_model(args)
    os.makedirs(args.output_dir, exist_ok=True)
    
    task_str = args.task
    if args.alg is not None:
        task_str += f"_{args.alg}"
    # Label outputs with canonical flags only
    if args.tokens_per_step is not None:
        task_str += f"_K={args.tokens_per_step}"
    if args.max_lookahead is not None:
        task_str += f"_M={args.max_lookahead}"
    if args.kv_window is not None:
        task_str += f"_W={args.kv_window}"
    if args.apd_mixture_weight is not None:
        task_str += f"_R={args.apd_mixture_weight}"
    if args.num_steps is not None:
        task_str += f"_num_steps={args.num_steps}"
    if args.tag:
        task_str += f"_{args.tag}"
    output_filename = f"{args.model_alias}_{task_str}_limit{args.limit}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    logger.info(f"Results will be saved to: {output_path}")
    
    
    task_name = args.task
    
    if task_name == "math":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: boxed{ANSWER}."
    elif task_name == "gpqa":
        system_instruction = "You are a helpful assistant. Justify your final answer by first explaining your step-by-step derivation or reasoning. Conclude by presenting the final answer in the format: (LETTER)."
    else:
        system_instruction = "You are a helpful assistant."
        
    if "qwen" in args.model_alias:
        system_instruction = "You are a helpful assistant." # Normal prompt for qwen models

    task = [TASKS[task_name]]
    
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task,
        batch_size=1,
        limit=args.limit,
        log_samples=True,    
        write_out=True,    
        num_fewshot=0, 
        apply_chat_template=True,
        system_instruction=system_instruction,
        gen_kwargs="max_length=16384" if "qwen" in args.model_alias else None,
        confirm_run_unsafe_code=True
    )

    results["profile"] = model.get_profile()
    
    parsed_results = parse_results(results, task_name=task_name)
    
    with open(output_path, 'w') as f:
        json.dump(parsed_results, f, indent=4)
    

if __name__ == "__main__":
    main()
    
    