import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_dir = os.path.expanduser(
    "~/LLM-Streamline/facebook/opt-1.3b-llm-streamline-mseloss"
)


def main():
    print(f"Loading model and tokenizer from {model_dir} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )

    # Dummy input text
    input_text = "Hello, how are you?"

    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # Run model (no grad for speed)
    with torch.no_grad():
        outputs = model(**inputs)

    # Print output keys and shapes
    print("Output keys:", outputs.keys())
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

    # Example: print logits shape and a sample
    if "logits" in outputs:
        print("Sample logits (first token):", outputs.logits[0, 0, :5])


if __name__ == "__main__":
    main()
