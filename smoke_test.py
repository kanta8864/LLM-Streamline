# File: smoke_test.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("--- SMOKE TEST STARTED ---")
print("This test will verify the environment, caching, and saving process.")

try:
    # --- 1. Verify Environment and CUDA ---
    print("\n--- Verifying Environment ---")
    hf_home = os.environ.get('HF_HOME')
    print(f"HF_HOME environment variable is set to: {hf_home}")
    assert hf_home and '/tmp/' in hf_home, "HF_HOME is not set or not pointing to /tmp."
    print("✅ HF_HOME is correctly pointing to a temporary directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    assert device == "cuda", "CUDA is not available to PyTorch."
    print("✅ PyTorch can see the CUDA device.")

    # --- 2. Verify Model Loading and Caching ---
    print("\n--- Verifying Model Loading (testing cache) ---")
    # Use a tiny model for a fast download and load test
    model_name = "distilgpt2" 
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"✅ Successfully loaded small model '{model_name}' to {next(model.parameters()).device}.")

    # --- 3. Verify Saving Process ---
    output_dir = os.path.join(os.environ.get('SLURM_JOB_ID', '.'), "smoke_test_output")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n--- Verifying Model Saving to: {output_dir} ---")
    # This is the critical step that was failing before.
    # It will trigger the deepspeed/nvcc check.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Successfully saved model and tokenizer.")

    print("\n-------------------------")
    print("✅✅✅ SMOKE TEST PASSED ✅✅✅")
    print("-------------------------")

except Exception as e:
    print("\n-------------------------")
    print(f"❌❌❌ SMOKE TEST FAILED: {e} ❌❌❌")
    print("-------------------------")
    # Re-raise the exception to make the job fail
    raise e