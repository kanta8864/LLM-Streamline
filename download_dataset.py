# Filename: download_dataset_script.py
from datasets import load_dataset
import os

# Verify HF_HOME (optional, but good for sanity check)
hf_home_path = os.environ.get("HF_HOME")
if hf_home_path:
    print(f"Using HF_HOME: {hf_home_path}")
    print(f"Datasets will be cached in: {os.path.join(hf_home_path, 'datasets')}")
else:
    print(
        "Warning: HF_HOME is not set. Datasets will be cached in the default location (usually ~/.cache/huggingface/datasets)."
    )
    print(
        "Ensure this matches the HF_HOME used by your compute job if you want it to find the cache."
    )

dataset_name = "DKYoon/SlimPajama-6B"

print(f"Attempting to download and cache dataset: {dataset_name}")
print("This may take a while as the dataset is around 15GB.")

# This command downloads the dataset and stores it in the cache.
dataset_dict = load_dataset(dataset_name)

print(f"Dataset '{dataset_name}' downloaded and cached successfully.")
print("Dataset structure:", dataset_dict)
# The actual cached data will be in a subdirectory like:
# /scratch/ktanahashi/huggingface_cache/datasets/DKYoon___slim_pajama-6_b/
