from transformers import AutoTokenizer

# The original model you started from
base_model_name = "facebook/opt-1.3b"

# The local path where your fine-tuned model is saved
local_model_path = "facebook/opt-1.3b-llm-streamline-mseloss/"

print(f"Loading tokenizer from '{base_model_name}'...")
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

print(f"Saving tokenizer to '{local_model_path}'...")
tokenizer.save_pretrained(local_model_path)

print("Done! Your model directory is now complete.")
