from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    DataCollatorForLanguageModeling,
)
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn

from datasets import load_dataset, load_from_disk, concatenate_datasets
from args import TrainingArguments, ModelArguments
from LLM_Streamline.train_lightweightnetwork import lightweight_model_train


# In your main script (e.g., run.py)

def replace_lightweight_network(
    model,
    lightweight_network,
    pruned_model,
    best_layer,
    layer_intervals,
    model_name,
    num_layers,
):
    pruned_layers = [i for i in range(best_layer + 1, best_layer + layer_intervals)]

    lightweight_state_dict = lightweight_network.state_dict()
    pruned_weight = pruned_model.state_dict()
    original_weight = model.state_dict()

    if "llama" in model_name.lower():
        # === Handle Llama Model ===
        print("Configuring weights for a pruned Llama model.")
        pruned_weight["model.norm.weight"] = original_weight["model.norm.weight"]
        pruned_weight["model.embed_tokens.weight"] = original_weight["model.embed_tokens.weight"]
        pruned_weight["lm_head.weight"] = original_weight["lm_head.weight"]

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                continue

            pruned_prefix = f"model.layers.{j}"
            original_prefix = f"model.layers.{i}"

            if i == best_layer:
                # Copy Attention and Norms from original, but MLP from lightweight network
                for key_suffix in ["self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight"]:
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[f"{original_prefix}.{key_suffix}"]
                
                # Insert trained LlamaMLP weights
                for key_suffix in ["gate_proj.weight", "up_proj.weight", "down_proj.weight"]:
                    pruned_weight[f"{pruned_prefix}.mlp.{key_suffix}"] = lightweight_state_dict[key_suffix]
            else:
                # Copy the entire original layer
                for key_suffix in pruned_model.model.layers[j].state_dict().keys():
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[f"{original_prefix}.{key_suffix}"]
            j += 1

    elif "opt" in model_name.lower():
        # === Handle OPT Model ===
        print("Configuring weights for a pruned OPT model.")
        for key in ["model.decoder.embed_tokens.weight", "model.decoder.embed_positions.weight", "model.decoder.final_layer_norm.weight", "model.decoder.final_layer_norm.bias", "lm_head.weight"]:
            pruned_weight[key] = original_weight[key]

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                continue

            pruned_prefix = f"model.decoder.layers.{j}"
            original_prefix = f"model.decoder.layers.{i}"

            if i == best_layer:
                # Copy Attention and Norms from original, but FFN from lightweight network
                for key_suffix in ["self_attn.q_proj.weight", "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias", "self_attn_layer_norm.weight", "self_attn_layer_norm.bias", "final_layer_norm.weight", "final_layer_norm.bias"]:
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[f"{original_prefix}.{key_suffix}"]
                
                # Insert trained standard MLP weights (fc1, fc2)
                for key_suffix in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = lightweight_state_dict[key_suffix]
            else:
                # Copy the entire original layer
                for key_suffix in pruned_model.model.decoder.layers[j].state_dict().keys():
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[f"{original_prefix}.{key_suffix}"]
            j += 1
            
    else:
         raise NotImplementedError(f"Weight replacement not implemented for model type: {model_name}")

    pruned_model.load_state_dict(pruned_weight)
    return pruned_model


def parse_hf_args():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    args, training_args, _ = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )

    return args, training_args


def run():
    import os

    print(f"DEBUG: HF_HOME inside script = {os.environ.get('HF_HOME')}")
    print(f"DEBUG: HF_HUB_OFFLINE inside script = {os.environ.get('HF_HUB_OFFLINE')}")

    args, training_args = parse_hf_args()

    print(f"DEBUG: Attempting to load model directly from path: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    lightweight_network, best_layer = lightweight_model_train(
        model,
        tokenizer,
        "cuda",
        training_args.layer_intervals + 1,
        config.num_hidden_layers,
        training_args.cosine_num_data,
        training_args.train_num_data,
        training_args.batch_size,
        training_args.epoches,
        training_args.lr,
        training_args.min_lr,
        training_args.wd,
        config,
        args.model_name,
        training_args.gradient_accumulation_step,
    )

    config.num_hidden_layers -= training_args.layer_intervals
    pruned_model = AutoModelForCausalLM.from_config(config)
    pruned_model = replace_lightweight_network(
        model,
        lightweight_network,
        pruned_model,
        best_layer,
        training_args.layer_intervals + 1,
        args.model_name,
        config.num_hidden_layers + training_args.layer_intervals,
    )

    pruned_model.save_pretrained("{}-llm-streamline-mseloss".format(args.model_name))
