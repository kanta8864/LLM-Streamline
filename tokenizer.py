from transformers import GPT2TokenizerFast

# Load OPT-compatible tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("facebook/opt-1.3b")

# Optional: Add special tokens (if needed)
tokenizer.add_special_tokens({
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>"
})

# Save the tokenizer files to the model directory
tokenizer.save_pretrained("facebook/opt-1.3b-llm-streamline-mseloss")
