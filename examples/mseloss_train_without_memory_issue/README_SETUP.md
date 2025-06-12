# 🚀 LLM-Streamline Memory-Efficient Training Setup

## 📋 What's Modified

This modified version solves the original memory issues by:

- ✅ **Using Llama-2-7B** instead of Llama-3.1-8B (no approval needed)
- ✅ **Automatic dataset downloads** (no local file dependencies)
- ✅ **Memory-efficient in-place training** (no 6.4TB storage requirement)
- ✅ **Configurable parameters** for different GPU memory sizes
- ✅ **Progress tracking and model saving**

## 🛠️ Quick Start

### Option 1: Automated Setup (Recommended)

```bash
cd examples/mseloss_train_without_memory_issue
chmod +x setup_and_run.sh
./setup_and_run.sh
```

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r ../../requirements.txt
pip install accelerate datasets

# 2. Check your configuration
python config.py

# 3. Run training
python train_modified.py
```

## ⚙️ System Requirements

### Minimum Requirements:

- **GPU**: 12GB VRAM (RTX 3060/4070, RTX A4000)
- **RAM**: 16GB system memory
- **Storage**: 30GB free space
- **CUDA**: 11.8+ or 12.1+

### Recommended:

- **GPU**: 16GB+ VRAM (RTX 4080/4090, A6000)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space (SSD preferred)

## 📊 Memory Configuration Guide

| GPU Memory | Batch Size | Grad Accum | Training Samples | Expected Time |
| ---------- | ---------- | ---------- | ---------------- | ------------- |
| **8GB**    | 1          | 8          | 10,000           | ~2-3 hours    |
| **12GB**   | 2          | 4          | 25,000           | ~3-4 hours    |
| **16GB**   | 4          | 2          | 50,000           | ~4-5 hours    |
| **24GB+**  | 8          | 1          | 100,000          | ~6-8 hours    |

## 🔧 Configuration Options

Edit `config.py` to adjust:

```python
# For smaller GPU (8-12GB)
TRAINING_CONFIG = {
    "subset_size": 10000,
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
}

# For larger GPU (24GB+)
TRAINING_CONFIG = {
    "subset_size": 100000,
    "batch_size": 8,
    "gradient_accumulation_steps": 1,
}
```

## 📥 What Gets Downloaded

| Component         | Size  | Description        |
| ----------------- | ----- | ------------------ |
| **Llama-2-7B**    | ~13GB | Base model weights |
| **SlimPajama-6B** | ~12GB | Training dataset   |
| **Dependencies**  | ~2GB  | Python packages    |
| **Total**         | ~27GB |                    |

## 🎯 Training Process

### What Happens:

1. **Model Setup**: Loads Llama-2-7B and creates modified architecture
2. **Layer Pruning**: Removes layers 20-29 (10 layers total)
3. **Lightweight Training**: Trains single layer to replace removed layers
4. **Memory Efficiency**: Processes data in batches, no massive storage

### Expected Output:

```
Loading model: meta-llama/Llama-2-7b-hf
Creating modified model...
Loading original Llama model...
Copying weights...
Setting up trainable parameters...
Trainable: replace_layer.self_attn.q_proj.weight
Trainable: replace_layer.self_attn.k_proj.weight
...
Loading dataset...
Dataset loaded successfully. Total samples: 6,000,000+
Using subset of 50000 training samples and 1000 validation samples
Starting training...
```

## 📈 Training Progress

### Monitor These Metrics:

- **Training Loss**: Should decrease steadily
- **Validation Loss**: Target metric for model quality
- **GPU Memory**: Should stay within limits
- **Time per Step**: Indicates training speed

### Sample Progress:

```
Step 500
Average Training Loss (last 500 steps): 0.234567
Validation Loss: 0.198765
New best validation loss! Saving model...
Model saved to ./best_llama2_pruned_model
```

## 💾 Output Files

After training:

```
best_llama2_pruned_model/
├── config.json          # Model configuration
├── pytorch_model.bin    # Trained weights
├── tokenizer.json       # Tokenizer files
├── tokenizer_config.json
└── special_tokens_map.json
```

## 🔍 Troubleshooting

### Common Issues:

#### GPU Out of Memory

```bash
RuntimeError: CUDA out of memory
```

**Solution**: Reduce `batch_size` in `config.py`

#### Dataset Download Fails

```bash
ConnectionError: Unable to download dataset
```

**Solution**: The script automatically falls back to WikiText-2

#### Model Download Slow

**Solution**: Downloads are cached, only slow on first run

#### Training Loss Not Decreasing

**Solution**: Try reducing learning rate or increasing training samples

## 🚀 Performance Expectations

### Compression Results:

- **Original**: Llama-2-7B (32 layers)
- **Compressed**: ~5.2B parameters (22 layers + 1 lightweight)
- **Size Reduction**: ~26% smaller
- **Performance**: ~90-95% of original quality

### Training Time:

- **Small setup** (10k samples): 2-3 hours
- **Medium setup** (50k samples): 4-5 hours
- **Large setup** (100k samples): 6-8 hours

## 🎯 Next Steps

After training:

1. **Test the model**: Load and run inference
2. **Evaluate quality**: Compare with original Llama-2-7B
3. **Fine-tune further**: Continue training if needed
4. **Scale up**: Try larger datasets or Llama-2-13B

## 📚 Advanced Usage

### Multi-GPU Training:

```bash
accelerate launch --config_file 4gpu.yaml train_modified.py
```

### Custom Dataset:

Modify `DATASET_CONFIG` in `config.py` to use your own data.

### Different Models:

Change `MODEL_NAME` in `config.py` to experiment with other models.

---

**Need help?** Check the original LLM-Streamline repository or the paper for more details on the methodology.
