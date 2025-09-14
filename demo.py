import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from dream.modeling_dream import DreamModel
import torch
import time
from transformers import set_seed
from prettytable import PrettyTable

set_seed(1)

COLORS = [
    '\033[91m',  # Red
    '\033[92m',  # Green
    '\033[93m',  # Yellow
    '\033[94m',  # Blue
    '\033[95m',  # Magenta
    '\033[96m',  # Cyan
]

WHITE = '\033[97m'
RESET = '\033[0m'
BOLD = '\033[1m'

def parse_args():
    parser = argparse.ArgumentParser(description="Dream model demonstration with APD/Speculative decoding")
    
    # Model arguments
    parser.add_argument("--dream_ckpt", type=str, default="Dream-org/Dream-v0-Instruct-7B", 
                       help="Path to Dream model checkpoint")
    parser.add_argument("--verifier_ckpt", type=str, default="Qwen/Qwen2.5-0.5B",
                       help="Path to verifier model checkpoint")
    
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on")
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, 
                       default="Toulouse has twice as many sheep as Charleston. Charleston has 4 times as many sheep as Seattle. How many sheep do Toulouse, Charleston, and Seattle have together if Seattle has 20 sheep?", # Answer is 260
                       help="Input prompt for generation")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum number of new tokens to generate")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95,
                       help="Top-p for generation")
    
    # APD-specific arguments
    parser.add_argument("--apd_mixture_weight", type=float, default=0.7,
                       help="APD mixture weight (R parameter, between 0.0 and 1.0, higher is faster)")
    parser.add_argument("--kv_window", type=int, default=16,
                       help="APD KV window size (W parameter, greater than 0, lower is faster)")
    parser.add_argument("--max_lookahead", type=int, default=100,
                       help="APD maximum lookahead (M parameter, greater than 0, lower is faster)")
    
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading models...")
    print(f"Dream model: {args.dream_ckpt}")
    print(f"Verifier model: {args.verifier_ckpt}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.dream_ckpt, trust_remote_code=True)
    model = DreamModel.from_pretrained(args.dream_ckpt, trust_remote_code=True, 
                                      attn_implementation="sdpa", 
                                      torch_dtype=torch.bfloat16, 
                                      device_map=args.device) 
    verifier_model = AutoModelForCausalLM.from_pretrained(args.verifier_ckpt, trust_remote_code=True, 
                                                         attn_implementation="sdpa", 
                                                         torch_dtype=torch.bfloat16, 
                                                         device_map=args.device)
    
    print(f"\nPrompt: {args.prompt}")
    print(f"APD parameters - R: {args.apd_mixture_weight}, W: {args.kv_window}, M: {args.max_lookahead}")
    
    messages = [{"role": "user", "content": args.prompt}]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt")
    
    input_ids = input_ids.to(model.device)

    def generation_tokens_with_color_hook_func(step, x, acceptance_counts):
        if acceptance_counts is None:
            return x
            
        # Clear previous output and show step info at top
        print(f"\033[2J\033[H", end="")  # Clear screen and move cursor to top
        
        # Decode the current sequence
        full_text = tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0]
        
        # Remove mask tokens completely
        full_text = full_text.replace(tokenizer.mask_token, "")
        
        # Decode the original prompt for comparison
        prompt_text = tokenizer.decode(input_ids[0].tolist()).split(tokenizer.eos_token)[0]
        
        # Find where the new generation starts (after the prompt)
        if full_text.startswith(prompt_text):
            generated_text = full_text[len(prompt_text):]
            
            # If we have acceptance counts, colorize only the generated segments
            if acceptance_counts and len(acceptance_counts) > 0:
                # Convert generated text to tokens for precise coloring
                generated_tokens = tokenizer.tokenize(generated_text)
                colored_generated = ""
                token_idx = 0
                
                for segment_idx, count in enumerate(acceptance_counts):
                    if token_idx >= len(generated_tokens):
                        break
                    color = COLORS[segment_idx % len(COLORS)]
                    segment_tokens = generated_tokens[token_idx:token_idx + count]
                    segment_text = tokenizer.convert_tokens_to_string(segment_tokens)
                    colored_generated += f"{color}{segment_text}{RESET}"
                    token_idx += count
                    
                # Add any remaining tokens in default color
                if token_idx < len(generated_tokens):
                    remaining_tokens = generated_tokens[token_idx:]
                    remaining_text = tokenizer.convert_tokens_to_string(remaining_tokens)
                    colored_generated += remaining_text
                
                # Store the colored generation for final output
                generation_tokens_with_color_hook_func.final_colored_text = f"{WHITE}{prompt_text}{RESET}{colored_generated}"
                
                # Display prompt in white/default + colored generation
                print(f"{WHITE}{prompt_text}{RESET}{colored_generated}")
            else:
                # Store uncolored version
                generation_tokens_with_color_hook_func.final_colored_text = f"{WHITE}{prompt_text}{RESET}{generated_text}"
                print(f"{WHITE}{prompt_text}{RESET}{generated_text}")
        else:
            generation_tokens_with_color_hook_func.final_colored_text = full_text
            print(f"{full_text}")
            
        return x
    
    # Initialize the final_colored_text attribute
    generation_tokens_with_color_hook_func.final_colored_text = ""


    start_time = time.time()
    
    # Generate
    decode_outputs = model.diffusion_generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            temperature=args.temperature,
            top_p=args.top_p,
            alg="apd",
            generation_tokens_hook_func=generation_tokens_with_color_hook_func,
            verifier_model=verifier_model,
            kv_window=args.kv_window,
            max_lookahead=args.max_lookahead,
            apd_mixture_weight=args.apd_mixture_weight,
        )
    
    end_time = time.time()
    
    # Clear screen one final time for clean final output
    print(f"\033[2J\033[H", end="")
    
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}           FINAL OUTPUT{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")
    
    # Use the stored colored text if available, otherwise decode normally
    if hasattr(generation_tokens_with_color_hook_func, 'final_colored_text') and generation_tokens_with_color_hook_func.final_colored_text:
        print(generation_tokens_with_color_hook_func.final_colored_text)
    else:
        final_output = tokenizer.decode(decode_outputs.sequences[0].tolist(), skip_special_tokens=True)
        print(final_output)
    
    print(f"\n{BOLD}PERFORMANCE METRICS{RESET}")
    
    generation_time = end_time - start_time
    
    # Create pretty table
    table = PrettyTable()
    table.field_names = ["Metric", "Value"]
    table.add_row(["Generation Time", f"{generation_time:.2f} s"])
    
    if hasattr(decode_outputs, 'profile') and decode_outputs.profile:
        profile = decode_outputs.profile
        throughput = profile.num_tokens_generated / profile.total_time
        table.add_row(["Throughput", f"{throughput:.1f} tok/s"])
    
    print(table)
    
if __name__ == "__main__":
    main()

