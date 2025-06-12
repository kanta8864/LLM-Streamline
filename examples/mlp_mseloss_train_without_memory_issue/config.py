# Configuration file for LLM-Streamline MLP-based Memory-Efficient Training

# Model Configuration
MODEL_NAME = "meta-llama/Llama-2-7b-hf"  # No approval needed
# Alternative options:
# MODEL_NAME = "meta-llama/Llama-2-13b-hf"  # Larger model, requires more memory
# MODEL_NAME = "huggyllama/llama-7b"       # Alternative Llama-7B

# Training Configuration
TRAINING_CONFIG = {
    # Dataset
    "subset_size": 50000,           # Number of training samples (start small!)
    "eval_size": 1000,              # Number of validation samples
    "block_size": 2048,             # Sequence length
    
    # Training parameters
    "batch_size": 4,                # Adjust based on your GPU memory
    "eval_batch_size": 8,           # Evaluation batch size
    "gradient_accumulation_steps": 2, # Effective batch size = batch_size * gradient_accumulation_steps
    "learning_rate": 2e-4,          # Learning rate (MLP may need higher LR than transformer)
    "min_learning_rate": 5e-6,      # Minimum learning rate for scheduler
    "weight_decay": 1e-3,           # Weight decay
    "epochs": 1,                    # Number of training epochs
    
    # Evaluation and saving
    "eval_steps": 500,              # Evaluate every N steps
    "save_model": True,             # Whether to save the best model
    "output_dir": "./best_llama2_mlp_pruned_model",  # Where to save the model
    
    # Technical settings
    "mixed_precision": "bf16",      # Mixed precision training
    "warmup_ratio": 0.03,          # Warmup steps as ratio of total steps
}

# MLP-specific configuration
MLP_CONFIG = {
    "hidden_size": 4096,           # Input/output size (Llama hidden size)
    "intermediate_size": 16384,    # Hidden layer size (4x hidden size)
    "activation": "relu",          # Activation function
    "parameters": 67125248,        # Approximate parameter count: 4096*16384 + 16384*4096 = ~134M
    "vs_transformer_layer": 110000000,  # Full transformer layer has ~110M parameters
    "efficiency_gain": "MLP is simpler but may need more training steps"
}

# Memory optimization settings
MEMORY_CONFIG = {
    "use_gradient_checkpointing": False,  # MLP is simple, checkpointing not needed
    "dataloader_num_workers": 0,          # Number of workers for data loading
    "pin_memory": False,                  # Pin memory for faster GPU transfer
}

# Dataset fallback configuration
DATASET_CONFIG = {
    "primary_dataset": "DKYoon/SlimPajama-6B",
    "fallback_dataset": "wikitext",
    "fallback_config": "wikitext-2-raw-v1",
    "fallback_train_size": 1000,
    "fallback_eval_size": 100,
}

# GPU memory recommendations (MLP version uses less memory than transformer)
GPU_MEMORY_GUIDE = {
    "8GB": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "subset_size": 15000,
        "note": "MLP version is more memory efficient than transformer"
    },
    "12GB": {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "subset_size": 35000,
        "note": "Good performance with MLP lightweight layer"
    },
    "16GB": {
        "batch_size": 6,
        "gradient_accumulation_steps": 2,
        "subset_size": 75000,
        "note": "High performance, MLP trains faster than transformer"
    },
    "24GB+": {
        "batch_size": 12,
        "gradient_accumulation_steps": 1,
        "subset_size": 150000,
        "note": "Maximum performance, can use large batches"
    }
}

def get_config_for_gpu_memory(memory_gb):
    """Get recommended configuration based on GPU memory"""
    if memory_gb <= 8:
        return GPU_MEMORY_GUIDE["8GB"]
    elif memory_gb <= 12:
        return GPU_MEMORY_GUIDE["12GB"]
    elif memory_gb <= 16:
        return GPU_MEMORY_GUIDE["16GB"]
    else:
        return GPU_MEMORY_GUIDE["24GB+"]

def print_config():
    """Print current configuration"""
    print("📋 MLP-based LLM-Streamline Configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {TRAINING_CONFIG['subset_size']:,}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Mixed precision: {TRAINING_CONFIG['mixed_precision']}")
    print(f"Output directory: {TRAINING_CONFIG['output_dir']}")
    
    print(f"\n🔧 MLP Architecture:")
    print(f"Input size: {MLP_CONFIG['hidden_size']}")
    print(f"Hidden size: {MLP_CONFIG['intermediate_size']}")
    print(f"Activation: {MLP_CONFIG['activation']}")
    print(f"Parameters: {MLP_CONFIG['parameters']:,} (~67M)")

def get_memory_recommendations():
    """Print memory usage recommendations"""
    print("\n💾 GPU Memory Recommendations (MLP Version):")
    for memory, config in GPU_MEMORY_GUIDE.items():
        print(f"{memory:>6}: batch_size={config['batch_size']}, grad_accum={config['gradient_accumulation_steps']}, subset={config['subset_size']:,} - {config['note']}")

def compare_approaches():
    """Compare MLP vs Transformer approaches"""
    print("\n⚖️  MLP vs Transformer Lightweight Layer:")
    print("┌─────────────────────┬──────────────────┬──────────────────┐")
    print("│ Aspect              │ MLP (This)       │ Transformer      │")
    print("├─────────────────────┼──────────────────┼──────────────────┤")
    print("│ Parameters          │ ~67M             │ ~110M            │")
    print("│ Training Speed      │ Faster           │ Slower           │")
    print("│ Memory Usage        │ Lower            │ Higher           │")
    print("│ Expressiveness      │ Simpler          │ More complex     │")
    print("│ Performance         │ Good             │ Potentially better│")
    print("│ Training Stability  │ More stable      │ May need tuning  │")
    print("└─────────────────────┴──────────────────┴──────────────────┘")

if __name__ == "__main__":
    print_config()
    get_memory_recommendations()
    compare_approaches() 