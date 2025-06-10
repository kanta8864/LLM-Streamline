import tensorflow as tf
import numpy as np
import os

model_path = os.path.expanduser(
    "~/LLM-Streamline/facebook/opt-1.3b-llm-streamline-mseloss"
)


def main():
    # Load the model
    print(f"Loading model from: {model_path}")
    model = tf.saved_model.load(model_path)

    # Create a dummy input matching typical input shape
    # This depends on your model's input signature
    # Let's assume the model takes input_ids tensor of shape (batch_size, sequence_length)
    # For example, batch_size=1, sequence_length=10
    dummy_input = tf.constant([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=tf.int32)

    # Depending on model, call might differ — here we try the simplest
    try:
        output = model(dummy_input)
    except Exception as e:
        print("Model call failed:", e)
        return

    # Print output info
    print("Model output type:", type(output))
    if isinstance(output, dict):
        for k, v in output.items():
            print(
                f"Output '{k}': type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}"
            )
    elif isinstance(output, (list, tuple)):
        for i, v in enumerate(output):
            print(
                f"Output[{i}]: type={type(v)}, shape={v.shape if hasattr(v, 'shape') else 'N/A'}"
            )
    else:
        print(f"Output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        print(
            f"Output value (sample): {output.numpy() if hasattr(output, 'numpy') else output}"
        )


if __name__ == "__main__":
    main()
