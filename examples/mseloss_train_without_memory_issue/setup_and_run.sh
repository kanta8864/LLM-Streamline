#!/bin/bash

echo "🚀 Setting up LLM-Streamline Memory-Efficient Training"
echo "======================================================="

# Check if we're in the right directory
if [ ! -f "modeling_llama.py" ]; then
    echo "❌ Error: Please run this script from the examples/mseloss_train_without_memory_issue directory"
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

echo "🎯 Configuration Summary:"
echo "- Model: Llama-2-7B (no approval needed)"
echo "- Dataset: SlimPajama-6B (auto-download) with fallback to WikiText-2"
echo "- Training subset: 50,000 samples"
echo "- Validation subset: 1,000 samples"
echo "- Batch size: 4 (adjust based on your GPU memory)"
echo "- Mixed precision: BF16"

echo ""
echo "🚀 Starting training..."
echo "This will:"
echo "1. Download Llama-2-7B (~13GB)"
echo "2. Download SlimPajama dataset (~12GB) or fallback to WikiText-2"
echo "3. Start memory-efficient training"
echo ""

read -p "Continue? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Run the training script
echo "Starting training with train_modified.py..."
python train_modified.py

echo ""
echo "✅ Training completed!"
echo "Check the output above for training progress and final model location." 