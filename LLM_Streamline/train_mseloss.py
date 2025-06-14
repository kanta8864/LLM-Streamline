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

from dataclasses import dataclass, field

# --- Argument Dataclasses ---


@dataclass
class ScriptArguments:
    """
    Arguments specific to this script's execution, like input/output paths.
    """

    output_dir: str = field(
        metadata={
            "help": "The local directory on the node to save final model artifacts before copying to network storage."
        }
    )


# --- Core Functions ---


def replace_lightweight_network(
    model,
    lightweight_network,
    pruned_model,
    best_layer,
    layer_intervals,
    model_name,
    num_layers,
):
    # This function's internal logic remains unchanged as it's performing the model surgery.
    print("\n[DEBUG] --- Entering replace_lightweight_network function ---")
    print(f"[DEBUG] Original model has {num_layers} layers.")
    print(f"[DEBUG] Best layer determined to be: {best_layer}")
    print(f"[DEBUG] Number of layers to prune: {layer_intervals - 1}")

    pruned_layers = [i for i in range(best_layer + 1, best_layer + layer_intervals)]
    print(f"[DEBUG] Layers to be pruned (will be skipped): {pruned_layers}")

    lightweight_state_dict = lightweight_network.state_dict()
    pruned_weight = pruned_model.state_dict()
    original_weight = model.state_dict()

    print(f"[DEBUG] Size of original weight dictionary: {len(original_weight)} keys")
    print(
        f"[DEBUG] Size of new (empty) pruned weight dictionary: {len(pruned_weight)} keys"
    )
    print(
        f"[DEBUG] Size of lightweight network dictionary: {len(lightweight_state_dict)} keys"
    )
    print(
        f"[DEBUG] Keys available in lightweight network: {list(lightweight_state_dict.keys())}"
    )

    if "llama" in model_name.lower():
        print("Configuring weights for a pruned Llama model.")
        pruned_weight["model.norm.weight"] = original_weight["model.norm.weight"]
        pruned_weight["model.embed_tokens.weight"] = original_weight[
            "model.embed_tokens.weight"
        ]
        pruned_weight["lm_head.weight"] = original_weight["lm_head.weight"]

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                continue

            pruned_prefix = f"model.layers.{j}"
            original_prefix = f"model.layers.{i}"

            if i == best_layer:
                for key_suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.k_proj.weight",
                    "self_attn.v_proj.weight",
                    "self_attn.o_proj.weight",
                    "input_layernorm.weight",
                    "post_attention_layernorm.weight",
                ]:
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[
                        f"{original_prefix}.{key_suffix}"
                    ]
                for key_suffix in [
                    "gate_proj.weight",
                    "up_proj.weight",
                    "down_proj.weight",
                ]:
                    pruned_weight[f"{pruned_prefix}.mlp.{key_suffix}"] = (
                        lightweight_state_dict[key_suffix]
                    )
            else:
                for key_suffix in pruned_model.model.layers[j].state_dict().keys():
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[
                        f"{original_prefix}.{key_suffix}"
                    ]
            j += 1

    elif "opt" in model_name.lower():
        print("Configuring weights for a pruned OPT model.")
        for key in [
            "model.decoder.embed_tokens.weight",
            "model.decoder.embed_positions.weight",
            "model.decoder.final_layer_norm.weight",
            "model.decoder.final_layer_norm.bias",
            "lm_head.weight",
        ]:
            pruned_weight[key] = original_weight[key]

        j = 0
        for i in range(num_layers):
            if i in pruned_layers:
                continue

            pruned_prefix = f"model.decoder.layers.{j}"
            original_prefix = f"model.decoder.layers.{i}"

            if i == best_layer:
                for key_suffix in [
                    "self_attn.q_proj.weight",
                    "self_attn.q_proj.bias",
                    "self_attn.k_proj.weight",
                    "self_attn.k_proj.bias",
                    "self_attn.v_proj.weight",
                    "self_attn.v_proj.bias",
                    "self_attn.out_proj.weight",
                    "self_attn.out_proj.bias",
                    "self_attn_layer_norm.weight",
                    "self_attn_layer_norm.bias",
                    "final_layer_norm.weight",
                    "final_layer_norm.bias",
                ]:
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[
                        f"{original_prefix}.{key_suffix}"
                    ]
                for key_suffix in ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]:
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = (
                        lightweight_state_dict[key_suffix]
                    )
            else:
                for key_suffix in (
                    pruned_model.model.decoder.layers[j].state_dict().keys()
                ):
                    pruned_weight[f"{pruned_prefix}.{key_suffix}"] = original_weight[
                        f"{original_prefix}.{key_suffix}"
                    ]
            j += 1
    else:
        raise NotImplementedError(
            f"Weight replacement not implemented for model type: {model_name}"
        )

    print(
        f"\n[DEBUG] Finished constructing weight dictionary. New model should have {j} layers."
    )
    print(
        "[DEBUG] Attempting to load the constructed weights into the pruned model shell..."
    )
    try:
        pruned_model.load_state_dict(pruned_weight)
        print("[DEBUG] ✅ SUCCESS: load_state_dict completed without errors.")
    except Exception as e:
        print(
            f"[DEBUG] ❌ FAILED: load_state_dict failed. This is why weights are missing."
        )
        print(f"[DEBUG]   - Specific Error: {e}")
    print("[DEBUG] --- Exiting replace_lightweight_network function ---\n")

    return pruned_model


def parse_hf_args():
    """
    Parses arguments for the script, including model, training, and custom script arguments.
    """
    # This parser will now look for arguments defined in all three dataclasses.
    parser = HfArgumentParser((ModelArguments, TrainingArguments, ScriptArguments))

    # It will unpack the parsed arguments into three corresponding objects.
    model_args, training_args, script_args = parser.parse_args_into_dataclasses()

    return model_args, training_args, script_args


def run():
    # --- Setup and Environment Fixes ---
    print(f"DEBUG: HF_HOME inside script = {os.environ.get('HF_HOME')}")
    invalid_cuda_home = os.environ.get("CUDA_HOME", "")
    if "insy" in invalid_cuda_home:
        print(f"--- Found and unsetting invalid CUDA_HOME: {invalid_cuda_home} ---")
        del os.environ["CUDA_HOME"]

    print("Torch sees GPU:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Argument Parsing ---
    # The function now returns three sets of arguments, including our script_args.
    args, training_args, script_args = parse_hf_args()

    # --- Model and Tokenizer Loading ---
    print(f"DEBUG: Attempting to load model directly from path: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, device_map=device
    )
    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    print(
        f"DEBUG: Model loaded. Device of first parameter: {next(model.parameters()).device}"
    )

    # --- Core Logic: Training and Pruning ---
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

    # --- Final Saving Stage ---
    # The output directory now comes from the command-line argument, not a hardcoded path.
    local_output_dir = script_args.output_dir
    os.makedirs(local_output_dir, exist_ok=True)
    print(f"\n--- Saving temporary artifacts to local disk: {local_output_dir} ---")

    # Save all the necessary artifacts.
    lightweight_path = os.path.join(local_output_dir, "lightweight_network.pt")
    torch.save(lightweight_network.state_dict(), lightweight_path)

    # Save the main pruned model, its configuration, and the tokenizer.
    pruned_model.save_pretrained(local_output_dir)
    tokenizer.save_pretrained(local_output_dir)

    print(f"✅ All artifacts successfully saved locally to {local_output_dir}")
