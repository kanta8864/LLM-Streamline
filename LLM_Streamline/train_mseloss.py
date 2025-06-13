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
import os

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
    # --- Start of Debugging Block ---
    print("\n[DEBUG] --- Entering replace_lightweight_network function ---")
    print(f"[DEBUG] Original model has {num_layers} layers.")
    print(f"[DEBUG] Best layer determined to be: {best_layer}")
    print(f"[DEBUG] Number of layers to prune: {layer_intervals - 1}")
    # --- End of Debugging Block ---

    pruned_layers = [i for i in range(best_layer + 1, best_layer + layer_intervals)]
    
    # --- Start of Debugging Block ---
    print(f"[DEBUG] Layers to be pruned (will be skipped): {pruned_layers}")
    # --- End of Debugging Block ---

    lightweight_state_dict = lightweight_network.state_dict()
    pruned_weight = pruned_model.state_dict()
    original_weight = model.state_dict()
    
    # --- Start of Debugging Block ---
    print(f"[DEBUG] Size of original weight dictionary: {len(original_weight)} keys")
    print(f"[DEBUG] Size of new (empty) pruned weight dictionary: {len(pruned_weight)} keys")
    print(f"[DEBUG] Size of lightweight network dictionary: {len(lightweight_state_dict)} keys")
    print(f"[DEBUG] Keys available in lightweight network: {list(lightweight_state_dict.keys())}")
    # --- End of Debugging Block ---


    if "llama" in model_name.lower():
        # === Handle Llama Model ===
        print("Configuring weights for a pruned Llama model.")
        pruned_weight["model.norm.weight"] = original_weight["model.norm.weight"]
        pruned_weight["model.embed_tokens.weight"] = original_weight["model.embed_tokens.weight"]
        pruned_weight["lm_head.weight"] = original_weight["lm_head.weight"]

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                print(f"[DEBUG] Pruning (skipping) original layer {i}")
                continue
            
            print(f"[DEBUG] Processing layer mapping: Original Layer {i} -> New Layer {j}")
            pruned_prefix = f"model.layers.{j}"
            original_prefix = f"model.layers.{i}"

            if i == best_layer:
                print(f"[DEBUG] ---> Inserting trained lightweight network at new layer {j}")
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
                print(f"[DEBUG] Pruning (skipping) original layer {i}")
                continue
            
            print(f"[DEBUG] Processing layer mapping: Original Layer {i} -> New Layer {j}")
            pruned_prefix = f"model.decoder.layers.{j}"
            original_prefix = f"model.decoder.layers.{i}"

            if i == best_layer:
                print(f"[DEBUG] ---> Inserting trained lightweight network at new layer {j}")
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

    # --- Start of Debugging Block ---
    # This is the most critical step. We will wrap it in a try...except block
    # to catch any errors if the weights don't match the new model's architecture.
    print(f"\n[DEBUG] Finished constructing weight dictionary. New model should have {j} layers.")
    print("[DEBUG] Attempting to load the constructed weights into the pruned model shell...")
    try:
        pruned_model.load_state_dict(pruned_weight)
        print("[DEBUG] ✅ SUCCESS: load_state_dict completed without errors.")
    except Exception as e:
        print(f"[DEBUG] ❌ FAILED: load_state_dict failed. This is why weights are missing.")
        print(f"[DEBUG]   - Specific Error: {e}")
    print("[DEBUG] --- Exiting replace_lightweight_network function ---\n")
    # --- End of Debugging Block ---

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

    import torch
    print("Torch sees GPU:", torch.cuda.is_available())
    print("Torch CUDA device count:", torch.cuda.device_count())
    print("Torch current device:", torch.cuda.current_device())
    print("Torch device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")


    args, training_args = parse_hf_args()

    print(f"DEBUG: Attempting to load model directly from path: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        device_map=device
    )

    print(f"DEBUG: Model loaded. Device of first parameter: {next(model.parameters()).device}")

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

    print("\n--- DEBUGGING MODEL STATE BEFORE SAVING ---")
    try:
        # Try to access a weight tensor from a layer in the pruned model
        # The exact name might vary based on the model architecture (OPT, Llama, etc.)
        example_tensor = pruned_model.model.decoder.layers[0].fc1.weight
        print("✅ SUCCESS: Found an example weight tensor.")
        print(f"   - Tensor shape: {example_tensor.shape}")
        print(f"   - A few example values: {example_tensor.view(-1)[:5]}")
    except Exception as e:
        print(f"❌ ERROR: Could not access an example weight tensor. The 'pruned_model' is likely empty.")
        print(f"   - Specific error: {e}")
    print("--- END DEBUGGING ---\n")

    base_model_name = os.path.basename(args.model_name.rstrip("/"))
    output_dir = f"{base_model_name}-llm-streamline-mseloss"

    # Save model, tokenizer, and config
    pruned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    config.save_pretrained(output_dir)

    # Save the lightweight network
    lightweight_path = os.path.join(output_dir, "lightweight_network.pt")
    torch.save(lightweight_network.state_dict(), lightweight_path)
    print(f"✅ Lightweight network saved to: {lightweight_path}")
