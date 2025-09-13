from lm_eval.models.huggingface import HFLM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import stop_sequences_criteria
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math
import time
import numpy as np


class ProfileEvalHarness(HFLM):
    def __init__(self, **args):
        super().__init__(**args)
        self.profile = {}
        
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        
        start = time.time()
        output = self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            **generation_kwargs,
        )
        stop_time = time.time()
        
        
        self.log_profile({
            "num_tokens_generated": output.shape[1] - context.shape[1],
            "total_time": stop_time - start,
        })
        
        return output
    
    def log_profile(self, profile):
        
        for k, v in profile.items():
            if k not in self.profile:
                self.profile[k] = []
            self.profile[k].append(v)
            
    def get_profile(self):
        num_tokens_generated = np.array(self.profile["num_tokens_generated"])
        total_times = np.array(self.profile["total_time"])
        throughputs = num_tokens_generated / total_times
        
        throughput_mean = throughputs.mean()
        throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
        num_tokens_generated_mean = num_tokens_generated.mean()
        num_tokens_generated_stderr = num_tokens_generated.std(ddof=1) / math.sqrt(len(num_tokens_generated))
        total_time_mean = total_times.mean()
        total_time_stderr = total_times.std(ddof=1) / math.sqrt(len(total_times))
        
        result = {"throughput_mean": throughput_mean,
                    "throughput_stderr": throughput_stderr,
                    "total_time_mean": total_time_mean,
                    "total_time_stderr": total_time_stderr,
                    "num_tokens_generated_mean": num_tokens_generated_mean,
                    "num_tokens_generated_stderr": num_tokens_generated_stderr}

        return result
    
class LladaEvalHarness(HFLM):
    def __init__(self, pretrained, tokenizer, **args):
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, **args)
        self.model_alias = "llada"
        self.alg = args.get("alg", None)
        self.tokens_per_step = args.get("tokens_per_step", None)
        self.num_steps = args.get("num_steps", None)
        self.profile = {}
            

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        
        start = time.time()
        from llada.llada_generate import llada_diffusion_generate, llada_ar_generate
        
        if self.alg == "leftright":
            outputs = llada_ar_generate(
                self.model,
                context,
                num_steps=256,
                gen_length=256,
                block_length=256,
                temperature=0.0,
                cfg_scale=0.0,
                tokens_per_step=self.tokens_per_step
            )
        else:

            assert self.alg == "low_confidence" or self.alg == "random"
            outputs = llada_diffusion_generate(
                self.model,
                context,
                num_steps=self.num_steps,
                gen_length=256,
                block_length=256,
                temperature=0.0,
                cfg_scale=0.0,
                remasking=self.alg,
            )
        end = time.time()
        
        
        num_tokens_generated = -context.shape[-1]
        for value in outputs[0]:
            if value == self.tokenizer.eos_token_id:
                break
            num_tokens_generated+=1
            
        profile = {"num_tokens_generated": num_tokens_generated,
                   "total_time": end - start}
        
        self.log_profile(profile)
        
        # self.log_profile({
        #     "num_tokens_generated": output.shape[1] - context.shape[1],
        #     "total_time": stop_time - start,
        # })
        
        return outputs
    
    def log_profile(self, profile):
        
        for k, v in profile.items():
            if k not in self.profile:
                self.profile[k] = []
            self.profile[k].append(v)
            
    def get_profile(self):
        num_tokens_generated = np.array(self.profile["num_tokens_generated"])
        total_times = np.array(self.profile["total_time"])
        throughputs = num_tokens_generated / total_times
        
        throughput_mean = throughputs.mean()
        throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
        num_tokens_generated_mean = num_tokens_generated.mean()
        num_tokens_generated_stderr = num_tokens_generated.std(ddof=1) / math.sqrt(len(num_tokens_generated))
        total_time_mean = total_times.mean()
        total_time_stderr = total_times.std(ddof=1) / math.sqrt(len(total_times))
        
        result = {"throughput_mean": throughput_mean,
                    "throughput_stderr": throughput_stderr,
                    "total_time_mean": total_time_mean,
                    "total_time_stderr": total_time_stderr,
                    "num_tokens_generated_mean": num_tokens_generated_mean,
                    "num_tokens_generated_stderr": num_tokens_generated_stderr}

        return result
    



class DreamEvalHarness(HFLM):
    def __init__(self, pretrained, tokenizer, **args):
        super().__init__(pretrained=pretrained, tokenizer=tokenizer, **args)
        self.model_alias = "dream"
        self.alg = args.get("alg", "origin")
        self.tokens_per_step = args.get("tokens_per_step", None)
        self.max_lookahead = args.get("max_lookahead", None)
        self.kv_window = args.get("kv_window", None)
        self.apd_mixture_weight = args.get("apd_mixture_weight", None)
        self.num_steps = args.get("num_steps", None)
        
        if self.num_steps is None:
            self.num_steps = 256
        
        if self.apd_mixture_weight is None and self.tokens_per_step is None:
            self.tokens_per_step = 1
        self.max_gen_toks_value = args.get("max_gen_toks", 256)
        
        # Accept external verifier checkpoint when APD is used
        if self.alg == "apd":
            verifier_ckpt = args.get("verifier_ckpt", None)
            if verifier_ckpt is None:
                raise ValueError("--qwen_ckpt (verifier_ckpt) is required when alg=apd")
            self.verifier_model = AutoModelForCausalLM.from_pretrained(
                verifier_ckpt, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda"
            )
        
        self.profile = {}
            
        
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError

    def loglikelihood_rolling(self, requests: list[Instance]):
        raise NotImplementedError
    
    def log_profile(self, profile):
        
        for k, v in profile.items():
            if k not in self.profile:
                self.profile[k] = []
            if type(v) == list:
                self.profile[k].extend(v)
            else:
                self.profile[k].append(v)
            
    
    def get_profile(self):
        
        if self.alg != "leftright" and self.alg != "apd":
            num_tokens_generated = np.array(self.profile["num_tokens_generated"])
            total_times = np.array(self.profile["total_time"])
            
            throughputs = num_tokens_generated / total_times
            throughput_mean = throughputs.mean()
            throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
            
            total_time_mean = np.mean(total_times)
            total_time_stderr = np.std(total_times, ddof=1) / math.sqrt(len(total_times))
        
            num_tokens_generated_mean = np.mean(num_tokens_generated)
            num_tokens_generated_stderr = np.std(num_tokens_generated, ddof=1) / math.sqrt(len(num_tokens_generated))
            
            result = {"throughput_mean": throughput_mean,
                    "throughput_stderr": throughput_stderr,
                    "total_time_mean": total_time_mean,
                    "total_time_stderr": total_time_stderr,
                    "num_tokens_generated_mean": num_tokens_generated_mean,
                    "num_tokens_generated_stderr": num_tokens_generated_stderr}
            
            return result
                    
        num_forward_evals = np.array(self.profile["num_forward_evals"])
        num_tokens_generated = np.array(self.profile["num_tokens_generated"])
        verification_time = np.array(self.profile["verification_time"])
        total_times = np.array(self.profile["total_time"])
        num_accepted = np.array(self.profile["acceptance_counts"])
        
        
        num_foward_evals_mean = np.mean(num_forward_evals)
        num_foward_evals_stderr = np.std(num_forward_evals, ddof=1) / math.sqrt(len(num_forward_evals))
        num_tokens_generated_mean = np.mean(num_tokens_generated)
        num_tokens_generated_stderr = np.std(num_tokens_generated, ddof=1) / math.sqrt(len(num_tokens_generated))
        verification_time_mean = np.mean(verification_time)
        verification_time_stderr = np.std(verification_time, ddof=1) / math.sqrt(len(verification_time))
        total_time_mean = np.mean(total_times)
        total_time_stderr = np.std(total_times, ddof=1) / math.sqrt(len(total_times))
        num_accepted_mean = np.mean(num_accepted)
        num_accepted_stderr = np.std(num_accepted, ddof=1) / math.sqrt(len(num_accepted))
        num_accepted_max = int(max(num_accepted))

        throughputs = num_tokens_generated / total_times
        throughput_mean = throughputs.mean()
        throughput_stderr = throughputs.std(ddof=1) / math.sqrt(len(throughputs))
        
        result = {"throughput_mean": throughput_mean,
                    "throughput_stderr": throughput_stderr,
                    "total_time_mean": total_time_mean,
                    "total_time_stderr": total_time_stderr,
                    "num_tokens_generated_mean": num_tokens_generated_mean,
                    "num_tokens_generated_stderr": num_tokens_generated_stderr,
                    "num_forward_evals_mean": num_foward_evals_mean,
                    "num_forward_evals_stderr": num_foward_evals_stderr,
                    "verification_time_mean": verification_time_mean,
                    "verification_time_stderr": verification_time_stderr,
                    "num_accepted_mean": num_accepted_mean,
                    "num_accepted_stderr": num_accepted_stderr,
                    "num_accepted_max": num_accepted_max}
        
        return result
    
    @property
    def max_gen_toks(self) -> int:
        return self.max_gen_toks_value
    
    def _model_generate(self, context, max_length, stop, **generation_kwargs):

        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
            
        if self.alg == "apd":
            
            # pass verifier model
            
            outputs = self.model.diffusion_generate(
                context,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                steps=self.num_steps,
                temperature=0.2,
                top_p=0.95,
                alg="apd",
                alg_temp=0.,
                tokens_per_step=self.tokens_per_step,
                max_lookahead=self.max_lookahead,
                kv_window=self.kv_window,
                apd_mixture_weight=self.apd_mixture_weight,
                verifier_model=self.verifier_model,
                return_dict_in_generate=True
                )
            
            self.log_profile(outputs.profile)
            
            return outputs.sequences
        
        elif self.alg == "leftright":
            
            outputs = self.model.diffusion_generate(
                context,
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                steps=self.num_steps,
                temperature=0.2,
                top_p=0.95,
                alg="leftright",
                alg_temp=0.,
                tokens_per_step=self.tokens_per_step,
                max_lookahead=self.max_lookahead,
                kv_window=self.kv_window,
                apd_mixture_weight=self.apd_mixture_weight,
                return_dict_in_generate=True
                )
            
            self.log_profile(outputs.profile)
            
            return outputs.sequences
        
        start = time.time()
        outputs = self.model.diffusion_generate(
            context,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            steps=self.num_steps,
            temperature=0.2,
            top_p=0.95,
            alg=self.alg,
            alg_temp=0.,
        )
        end = time.time()
        
        num_tokens_generated = -context.shape[-1]
        for value in outputs[0]:
            if value == self.tokenizer.eos_token_id:
                break
            num_tokens_generated+=1
            
        profile = {"num_tokens_generated": num_tokens_generated,
                   "total_time": end - start}
        
        self.log_profile(profile)
        return outputs

