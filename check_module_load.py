from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "./facebook/opt-1.3b-llm-streamline-mseloss"

try:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True  # only needed if custom code is involved
    )
    print("Model loaded successfully.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    print("Tokenizer loaded successfully.")

except Exception as e:
    print("Error while loading model or tokenizer:")
    print(e)
