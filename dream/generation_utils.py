# coding=utf-8
# Copyright 2024 The Dream team, HKUNLP Group and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)
from typing import List
logger = logging.get_logger(__name__)


import time


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            confidence, x0 = probs.max(dim=-1)
    else:
        confidence, x0 = probs.max(dim=-1)
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0] 
        top2_probs = sorted_probs[:, 1] 
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs 
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0



def add_gumbel_noise(logits, temperature, gumbel_noise=None):

    if temperature == 0:
        return logits, None
    
    if gumbel_noise is None:
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (- torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise, gumbel_noise
    else:
        return logits.exp() / gumbel_noise, gumbel_noise

def sample_with_gumbel(logits, gumbel_noise=None, temperature=0.0, top_p=None, top_k=None):

    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    

    logits_with_noise, gumbel_noise = add_gumbel_noise(logits, temperature, gumbel_noise)
    x0 = torch.argmax(logits_with_noise, dim=-1)
        
    return gumbel_noise, x0


def apd_accept_criterion(x, diffusion_logits, verifier_logits, gumbel_noise, apd_mixture_weight):
    if x.shape[-1] > diffusion_logits.shape[1]:
        x = x[:, -diffusion_logits.shape[1]:]
    if verifier_logits.shape[1] > diffusion_logits.shape[1]:
        verifier_logits = verifier_logits[:, -diffusion_logits.shape[1]:, :]
        
    target = apd_mixture_weight * diffusion_logits + (1-apd_mixture_weight) * verifier_logits
    _, target_samples = sample_with_gumbel(target, gumbel_noise=gumbel_noise)
    
    accept = x == target_samples
    accept[:, 0] = 1
    accept = torch.cumprod(accept, dim=-1)
    return accept

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    
@dataclass
class ProfileOutput(ModelOutput):
    num_forward_evals: int = None
    num_tokens_generated: int = None
    verification_time: float = None
    total_time: float = None
    acceptance_counts: List[int] = None
    
    
@dataclass
class DreamModelOutputWithProfile(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None
    profile: Optional[ProfileOutput] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)
        # Diffusion parameters (canonical names)
        self.tokens_per_step: Optional[int] = kwargs.pop("tokens_per_step", None)
        self.kv_window: Optional[int] = kwargs.pop("kv_window", None)
        self.max_lookahead: Optional[int] = kwargs.pop("max_lookahead", None)
        self.apd_mixture_weight: Optional[float] = kwargs.pop("apd_mixture_weight", None)
        

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)
        

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.

        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        
        verifier_model = kwargs.pop("verifier_model", None)
        
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        generation_tokens_hook_func = kwargs.pop("generation_tokens_hook_func", lambda step, x, logits: x)
        generation_logits_hook_func = kwargs.pop("generation_logits_hook_func", lambda step, x, logits: logits)

        # 2. Define model inputs
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)

        # 3. Prepare `max_length`.
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        # 4. Check input_ids
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and 
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )

        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask 
        )

        if generation_config.alg == "apd" or generation_config.alg == "leftright":
            result = self.apd_sample(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func,
                verifier_model=verifier_model
                
            )
        else:
            result = self._sample(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                generation_tokens_hook_func=generation_tokens_hook_func,
                generation_logits_hook_func=generation_logits_hook_func
            )
        
        
        return result

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        eps = generation_config.eps
        alg = generation_config.alg
        alg_temp = generation_config.alg_temp
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"
        
        timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

        # this allows user-defined token control of the intermediate steps
        x = generation_tokens_hook_func(None, x, None)
        for i in range(steps):
            mask_index = (x == mask_token_id)
            logits = self(x, attention_mask, tok_idx).logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

            # this allows user-defined logits control of the intermediate steps
            logits = generation_logits_hook_func(i, x, logits)

            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
        
            if alg == 'origin':
                p_transfer = 1 - s / t if i < steps - 1 else 1
                x0 = torch.zeros_like(x[mask_index], device=self.device, dtype=torch.long) + mask_token_id
                transfer_index_t_s = torch.rand(*x0.shape, device=self.device) < p_transfer
                _, x0[transfer_index_t_s]= sample_tokens(mask_logits[transfer_index_t_s], temperature=temperature, top_p=top_p, top_k=top_k)
                x[mask_index] = x0.clone()
            else:
                if alg == 'maskgit_plus':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif alg == 'topk_margin':
                    confidence, x0 = sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True)
                elif alg == 'entropy':
                    confidence, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True)
                else:
                    raise RuntimeError(f"Unknown alg: {alg}")
                num_mask_token = mask_index.sum()
                number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else num_mask_token
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(confidence, number_transfer_tokens)
                    else:
                        confidence = confidence / alg_temp
                        confidence = F.softmax(confidence, dim=-1)
                        transfer_index = torch.multinomial(confidence, num_samples=number_transfer_tokens)
                    x0_ = torch.zeros_like(x0, device=self.device, dtype=torch.long) + mask_token_id
                    x0_[transfer_index] = x0[transfer_index].clone()
                    x[mask_index] = x0_

            # this allows user-defined token control of the intermediate steps
            x = generation_tokens_hook_func(i, x, logits)

            if histories is not None:
                histories.append(x.clone())
        
        if return_dict_in_generate:
            return DreamModelOutput(
                sequences=x,
                history=histories,
            )
        else:
            return x
        
    def apd_sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        generation_tokens_hook_func,
        generation_logits_hook_func,
        verifier_model
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        
        total_time =  time.time()
        num_forward_evals = 0
        num_tokens_generated = 0
        total_verification_time = 0
        acceptance_counts = []
        
        
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        temperature = generation_config.temperature
        top_p = generation_config.top_p
        top_k = generation_config.top_k
        tokens_per_step = generation_config.tokens_per_step
        kv_window = generation_config.kv_window
        max_lookahead = generation_config.max_lookahead
        apd_mixture_weight = generation_config.apd_mixture_weight

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"


        # this allows user-defined token control of the intermediate steps
        generation_tokens_hook_func(None, x, None)
        
        indices = (x == mask_token_id).nonzero()
        masked_indices, _ = torch.unique(indices[:, 1], return_inverse=True)
        curr_idx = masked_indices[0].item()
        is_prefilling = True
        verifier_past_key_values = None
        
        prev_logits = None
        all_cache_positions = torch.arange(x.shape[-1]).long().to(self.device)
        diffusion_past_key_values = None
        
        while curr_idx < x.shape[-1]:
            
            if max_lookahead is not None:
                right_idx = min(curr_idx + max_lookahead, x.shape[-1])
            else:
                right_idx = x.shape[-1]
                
            if kv_window is not None: #and diffusion_past_key_values is not None:
                if diffusion_past_key_values is None:
                    left_idx = 0
                else:
                    left_idx = max(curr_idx - kv_window - num_accept + 1, 0) # Subtract num_accept to compute KV on newly sampled tokens
            else:
                left_idx = 0
            
            truncated_x = x[:, left_idx:right_idx]
            mask_index = (truncated_x == mask_token_id)

            
            cache_position = all_cache_positions[left_idx:right_idx] if diffusion_past_key_values is not None else None
            position_ids = cache_position.unsqueeze(0) if cache_position is not None else None
            
            if diffusion_past_key_values is not None:
                diffusion_past_key_values.crop(left_idx)
            
            outputs = self(truncated_x, 
                            attention_mask="full", 
                            past_key_values=diffusion_past_key_values,
                            cache_position=cache_position,
                            position_ids=position_ids)
            
            logits = outputs.logits
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
            
            diffusion_past_key_values = outputs.past_key_values if kv_window is not None else None
            
            num_forward_evals += 1

            # this allows user-defined logits control of the intermediate steps
            generation_logits_hook_func(curr_idx, x, None)
            mask_logits = logits[mask_index]
            
            gumbel_noise, x0 = sample_with_gumbel(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
            
            diffusion_logits = mask_logits[:, :151936].unsqueeze(0) #Restrict to vocab of qwen
            
            # Verification begins
            verification_time = time.time()
            
            if apd_mixture_weight is not None:
                if is_prefilling:
                    
                    draft = x.clone()[:, :right_idx]
                    draft[:, curr_idx:right_idx] = x0
                    cache_position = None
                    attention_mask = None
                    position_ids = None
                else:
                    draft = x0.clone().unsqueeze(0)
                    cache_position = all_cache_positions[curr_idx:right_idx]
                    position_ids = cache_position.unsqueeze(0)
                    
                
                verifier_outputs = verifier_model(draft, 
                                                  past_key_values=verifier_past_key_values,
                                                  cache_position=cache_position,
                                                  position_ids=position_ids) 
                
                verifier_past_key_values = verifier_outputs.past_key_values
                
                 
                if is_prefilling:
                    verifier_logits = verifier_outputs.logits[:, curr_idx-1:-1, :] #shift logits
                else:
                    verifier_logits = verifier_outputs.logits
                    verifier_logits = torch.cat([prev_logits, verifier_logits[:, :-1, :]], dim=1)
                
                if max_lookahead is not None:    
                    verifier_logits = verifier_logits[:, :max_lookahead, :]  
                is_prefilling = False

                accept = apd_accept_criterion(draft, diffusion_logits, verifier_logits, gumbel_noise, apd_mixture_weight)
                

                num_accept = accept.sum().item()
                num_accept = min(num_accept, x0.shape[-1])

                
                if num_accept < verifier_logits.shape[1]:
                    prev_logits = verifier_logits[:, num_accept, :].unsqueeze(1)
                else:
                    prev_logits = verifier_logits[:, -1, :].unsqueeze(1)
                    
                verifier_past_key_values.crop(curr_idx+num_accept)
            else:
                num_accept = tokens_per_step
            
            x[0, curr_idx:curr_idx+num_accept] = x0[:num_accept]
            curr_idx += num_accept

            no_mask = (x == mask_token_id).sum().item() == 0
            has_eos = (x == generation_config.eos_token_id).sum().item() > 0
            
            
            # this allows user-defined token control of the intermediate steps
            total_verification_time += time.time() - verification_time
            num_tokens_generated += num_accept
            
            acceptance_counts.append(num_accept)
            generation_tokens_hook_func(curr_idx, x, acceptance_counts)
            
            if no_mask or has_eos:
                break
            
        total_time = time.time() - total_time
        
        profile = ProfileOutput(
            num_forward_evals=num_forward_evals,
            num_tokens_generated=num_tokens_generated,
            verification_time=total_verification_time,
            total_time=total_time,
            acceptance_counts=acceptance_counts,
        )
        
        if return_dict_in_generate:
            return DreamModelOutputWithProfile(
                sequences=x,
                profile=profile
            )
        else:
            return x
        
        