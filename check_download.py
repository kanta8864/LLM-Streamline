# from datasets import load_dataset
# import os

# dataset_name = "DKYoon/SlimPajama-6B"
# split_to_check = "train"

# print(f"HF_HOME is set to: {os.getenv('HF_HOME')}")
# print(f"HF_HUB_OFFLINE is set to: {os.getenv('HF_HUB_OFFLINE')}") # Check this

# try:
#     print(f"Attempting to load '{dataset_name}' (split: '{split_to_check}')...")
#     # REMOVED local_files_only=True from here:
#     dataset = load_dataset(dataset_name, split=split_to_check, trust_remote_code=True)
#     print("\nDataset loaded successfully!")
#     print("Dataset information:")
#     print(dataset)
#     if len(dataset) > 0:
#         print(f"\nFirst example: {dataset[0]}")
# except Exception as e:
#     print(f"\nCould not load dataset.")
#     print(f"Error: {e}")
#     print("Ensure HF_HUB_OFFLINE=1 is set in your environment if testing offline.")


import argparse
import os
import sys
from transformers import AutoModelForCausalLM, AutoConfig

def check_model_loading_from_local_cache(model_name: str, use_trust_remote_code: bool):
    """
    Attempts to load a Hugging Face model configuration and then the full model
    EXCLUSIVELY from the local Hugging Face cache.
    Uses 'local_files_only=True'.
    """
    print(f"--- Hugging Face Local Cache Model Loading Check ---")
    print(f"Requested model: {model_name}")
    print(f"Using trust_remote_code: {use_trust_remote_code}")

    # Display relevant Hugging Face environment variable status
    hf_home = os.getenv("HF_HOME")
    # Construct the actual or typical default cache directory path for display
    if hf_home:
        cache_path_to_check = hf_home
        print(f"HF_HOME environment variable is set to: {cache_path_to_check}")
    else:
        if sys.platform == "win32":
            # A more complex default, but for simplicity, we'll show the common .cache structure
            # Actual path can be C:\Users\<User>\.cache\huggingface or similar
            cache_path_to_check_parts = [os.path.expanduser("~"), ".cache", "huggingface"]
        else:
            cache_path_to_check_parts = [os.path.expanduser("~"), ".cache", "huggingface"]
        cache_path_to_check = os.path.join(*cache_path_to_check_parts)
        print(f"HF_HOME environment variable is NOT set. Will check default Hugging Face cache directory (typically under {cache_path_to_check}).")

    # HF_HUB_OFFLINE is less critical here as local_files_only=True overrides, but good for context.
    hf_hub_offline = os.getenv("HF_HUB_OFFLINE", "0")
    print(f"HF_HUB_OFFLINE environment variable is: {'1 (Offline mode)' if hf_hub_offline == '1' else '0 or not set (Online mode)'}")
    print("-" * 40)

    print("INFO: This script is now checking if the model can be loaded EXCLUSIVELY from the local cache.")
    print(f"      It uses 'local_files_only=True' for all Hugging Face loading operations.")
    print(f"      If the model (config or weights) is not fully present and valid in the cache at '{cache_path_to_check}', this check will fail.")

    try:
        print(f"\nStep 1: Attempting to load configuration for '{model_name}' from local cache...")
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=use_trust_remote_code,
            local_files_only=True,  # Crucial change for local-only check
        )
        print("SUCCESS: Model configuration loaded successfully FROM LOCAL CACHE.")
        print(f"  Config class: {config.__class__.__name__}")

        print(f"\nStep 2: Attempting to load full model weights for '{model_name}' from local cache...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=use_trust_remote_code,
            local_files_only=True,  # Crucial change for local-only check
        )
        print("\nSUCCESS: Full model loaded successfully FROM LOCAL CACHE!")
        print(f"  Model class: {model.__class__.__name__}")
        print("\nThis indicates all necessary files for the model were found and are valid in your local Hugging Face cache.")

    except ImportError as e:
        print(f"\nERROR: ImportError occurred.")
        print(f"  This often means a missing dependency required by the model's custom code (if trust_remote_code=True was used and relevant).")
        print(f"  These dependencies would also need to be available in your Python environment.")
        print(f"  Error details: {e}")
    except OSError as e:
        # OSError is the most common error when local_files_only=True and files are missing/corrupted.
        print(f"\nERROR: OSError occurred. When loading with local_files_only=True, this typically means:")
        print(f"  1. The model '{model_name}' (or its config/weights) is not found in your local Hugging Face cache.")
        print(f"     The script checked the cache path effectively determined by: '{cache_path_to_check}'.")
        print(f"  2. The cached files are incomplete, corrupted, or not in the expected Hugging Face structure for this model.")
        print(f"  3. Necessary configuration files (like config.json, model weights) are missing from the specific model's cache directory.")
        print(f"  Please ensure the model was fully and correctly downloaded to this cache location previously.")
        print(f"  Error details: {e}")
    except RuntimeError as e:
        print(f"\nERROR: RuntimeError occurred.")
        print(f"  This can be due to insufficient RAM to load the model from cache, or other runtime problems with the local files.")
        print(f"  Error details: {e}")
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred during the local cache loading attempt.")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error details: {e}")
    finally:
        print(f"\n--- Local Cache Model Loading Check Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Hugging Face model loading EXCLUSIVELY from local cache.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Name of the model (as it appears in your cache) to check from huggingface.co/models."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass this flag to set trust_remote_code=True when loading the model."
    )

    args = parser.parse_args()

    check_model_loading_from_local_cache(args.model_name, args.trust_remote_code)