from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_dir = (
    "/home/kantatanahashi/LLM-Streamline/facebook/opt-1.3b-llm-streamline-mseloss"
)
base_model_name = "facebook/opt-1.3b"  # or the base model you started with


def main():
    print(f"Loading tokenizer from base model {base_model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print(f"Loading model weights from {model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )

    input_text = "Hello, how are you?"
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    print("Output keys:", outputs.keys())
    if "logits" in outputs:
        print("Sample logits (first token):", outputs.logits[0, 0, :5])


if __name__ == "__main__":
    main()
