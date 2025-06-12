# Configuration file for LLM-Streamline Memory-Efficient Training

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
    "learning_rate": 2e-4,          # Learning rate
    "min_learning_rate": 5e-6,      # Minimum learning rate for scheduler
    "weight_decay": 1e-3,           # Weight decay
    "epochs": 1,                    # Number of training epochs
    
    # Evaluation and saving
    "eval_steps": 500,              # Evaluate every N steps
    "save_model": True,             # Whether to save the best model
    "output_dir": "./best_llama2_pruned_model",  # Where to save the model
    
    # Technical settings
    "mixed_precision": "bf16",      # Mixed precision training
    "warmup_ratio": 0.03,          # Warmup steps as ratio of total steps
}

# Memory optimization settings
MEMORY_CONFIG = {
    "use_gradient_checkpointing": True,   # Save memory at cost of speed
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

# GPU memory recommendations
GPU_MEMORY_GUIDE = {
    "8GB": {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "subset_size": 10000,
        "note": "Very limited, consider using smaller model"
    },
    "12GB": {
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "subset_size": 25000,
        "note": "Minimum recommended for Llama-2-7B"
    },
    "16GB": {
        "batch_size": 4,
        "gradient_accumulation_steps": 2,
        "subset_size": 50000,
        "note": "Good performance, default settings"
    },
    "24GB+": {
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
        "subset_size": 100000,
        "note": "High performance, can use larger subsets"
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
    print("📋 Current Configuration:")
    print(f"Model: {MODEL_NAME}")
    print(f"Training samples: {TRAINING_CONFIG['subset_size']:,}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Gradient accumulation: {TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Effective batch size: {TRAINING_CONFIG['batch_size'] * TRAINING_CONFIG['gradient_accumulation_steps']}")
    print(f"Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"Mixed precision: {TRAINING_CONFIG['mixed_precision']}")
    print(f"Output directory: {TRAINING_CONFIG['output_dir']}")

def get_memory_recommendations():
    """Print memory usage recommendations"""
    print("\n💾 GPU Memory Recommendations:")
    for memory, config in GPU_MEMORY_GUIDE.items():
        print(f"{memory:>6}: batch_size={config['batch_size']}, grad_accum={config['gradient_accumulation_steps']}, subset={config['subset_size']:,} - {config['note']}")

if __name__ == "__main__":
    print_config()
    get_memory_recommendations() 