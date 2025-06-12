#!/bin/bash

echo "🧠 Setting up LLM-Streamline MLP-based Memory-Efficient Training"
echo "================================================================"

# Check if we're in the right directory
if [ ! -f "modeling_llama.py" ]; then
    echo "❌ Error: Please run this script from the examples/mlp_mseloss_train_without_memory_issue directory"
    echo "Expected files: modeling_llama.py, scheduler.py, train_modified.py"
    exit 1
fi

echo "📦 Installing dependencies..."
# Install requirements from the main directory
pip install -r ../../requirements.txt

# Additional dependencies that might be needed
pip install accelerate datasets

echo "🔍 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'Current device: {torch.cuda.current_device() if torch.cuda.is_available() else \"None\"}')"

echo "📊 System memory check..."
python -c "import psutil; print(f'Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB')"

echo "🏗️ Setting up Hugging Face cache (if needed)..."
export HF_HOME="./hf_cache"
mkdir -p $HF_HOME

echo "🎯 MLP Configuration Summary:"
echo "- Model: Llama-2-7B (no approval needed)"
echo "- Dataset: SlimPajama-6B (auto-download) with fallback to WikiText-2"
echo "- Lightweight layer: Simple 2-layer MLP"
echo "- MLP architecture: 4096 -> 16384 -> 4096 (ReLU activation)"
echo "- Training subset: 50,000 samples"
echo "- Validation subset: 1,000 samples"
echo "- Batch size: 4 (adjust based on your GPU memory)"
echo "- Mixed precision: BF16"

echo ""
echo "🧠 MLP vs Transformer Comparison:"
echo "- MLP parameters: ~67M (vs ~110M for transformer)"
echo "- Training speed: Faster (simpler architecture)"
echo "- Memory usage: Lower than transformer approach"
echo "- Performance: Good, potentially slightly lower than transformer"

echo ""
echo "🚀 Starting training..."
echo "This will:"
echo "1. Download Llama-2-7B (~13GB)"
echo "2. Download SlimPajama dataset (~12GB) or fallback to WikiText-2"
echo "3. Start MLP-based memory-efficient training"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Run the training script
echo "Starting MLP training with train_modified.py..."
python train_modified.py

echo ""
echo "✅ MLP training completed!"
echo "Check the output above for training progress and final model location."
echo "The trained model uses a simple MLP to replace 10 transformer layers." 