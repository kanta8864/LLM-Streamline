import os
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
model_id = "facebook/opt-1.3b"
cache_dir = "/scratch/ktanahashi/huggingface_cache"  # Your HF_HOME


# Ensure the HF_HOME environment variable is set for this script session
# This tells transformers where to download and look for files.
os.environ["HF_HOME"] = cache_dir
# For newer versions of huggingface_hub, HF_HUB_CACHE might also be respected
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir


print(f"Using Hugging Face cache directory: {os.environ.get('HF_HOME')}")
print(f"Attempting to download/verify model: {model_id}")

try:
    # 1. Download/Load the configuration
    # Setting local_files_only=False (or omitting it) allows download.
    print("Downloading/Verifying config...")
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,  # If required by the model
        cache_dir=cache_dir,  # Explicitly pass cache_dir
    )
    print("Config loaded successfully.")
    print(f"Config saved in cache associated with: {config.name_or_path}")

    # 2. Download/Load the model weights
    # This is the largest part and will take time if not cached.
    print("Downloading/Verifying model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,  # If required by the model
        config=config,  # Can pass the loaded config
        cache_dir=cache_dir,  # Explicitly pass cache_dir
    )
    print("Model weights loaded successfully.")

    # 3. Download/Load the tokenizer (important for most use cases)
    print("Downloading/Verifying tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,  # If required by the model
        cache_dir=cache_dir,  # Explicitly pass cache_dir
    )
    print("Tokenizer loaded successfully.")

    print(
        f"\nAll components for '{model_id}' should now be downloaded and cached correctly in '{cache_dir}'."
    )
    print(
        "Please check the directory structure, especially within a 'hub' subdirectory."
    )
    print(
        f"Expected location for snapshots: {os.path.join(cache_dir, 'hub', 'models--' + model_id.replace('/', '--'), 'snapshots')}"
    )

except Exception as e:
    print(f"An error occurred: {e}")
    print(
        "Please check your internet connection, model ID, and cache directory permissions."
    )
