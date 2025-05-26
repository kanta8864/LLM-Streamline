# 🚀 ICLR 2025 Spotlight: Streamlining Redundant Layers to Compress Large Language Models

I downloaded the models from the login node using the commands beforehand. As far as I understood, the compute nodes have no/limited internet access so we should pull the model beforehand. Maybe we can just opt to use opt-1.3b because then we have to wait less and requesting less resource will probably mean that we have to wait shorter in the queue.

```
# On the HPC LOGIN NODE, within your (myenv) environment
# Set the environment variable
export HF_HOME="/scratch/<your net id i believe>/huggingface_cache"
mkdir -p $HF_HOME # Ensure the directory exists

# Now run the download command (it will use the HF_HOME path)
huggingface-cli download meta-llama/Llama-3.1-8B --local-dir-use-symlinks False
```

I  tried created virtual env and then tried downloading the requirements from the login node. You may need to make sure python3 and pip are available and if not load them. I then tried to activate this virtual env in the script but somehow it doesnt work lol
```
python3 -m venv ~/myenv
source ~/myenv/bin/activate
pip install -r requirements.txt
```

You can submit the job using the command:

```
sbatch scripts/run_mseloss_training.sh
```


#### 📦 Open Source Models

We have released two compressed models on Hugging Face:
**[Llama-2-4.7B](https://huggingface.co/XiaodongChen/Llama-2-4.7B)** and
**[Llama-3.1-5.4B](https://huggingface.co/XiaodongChen/Llama-3.1-5.4B)**.

### 📊 Evaluation Results (lm-eval)

| Model                | arc_c | arc_e | boolq | hellaswag | openbookqa | rte  | winogrande | Avg  |
| -------------------- | ----- | ----- | ----- | --------- | ---------- | ---- | ---------- | ---- |
| Llama-3.1-8B         | 50.4  | 80.3  | 81.2  | 60.2      | 34.8       | 67.9 | 73.0       | 64.0 |
| ​**Llama-3.1-5.4B**​ | 42.1  | 72.2  | 78.0  | 54.3      | 27.2       | 62.8 | 71.0       | 58.2 |
| Llama-2-7B           | 43.3  | 76.4  | 77.7  | 57.2      | 31.4       | 62.8 | 69.1       | 59.7 |
| ​**Llama-2-4.7B**​   | 34.0  | 64.6  | 74.7  | 49.8      | 27.4       | 61.7 | 66.4       | 54.1 |

**Model Specifications**:

- `Llama-2-4.7B`: Using single Transformer Layer as lightweight network, trained on 0.06B tokens.
- `Llama-3.1-5.4B`: Using two Transformer Layers as lightweight network, trained on 1.3B tokens.
- Both models trained using `llm_loss`.

## 🤖 Supported LLM Architectures

- [Llama-3](https://huggingface.co/models?search=llama3)
- [Llama-2](https://huggingface.co/models?search=llama2)
- [OPT](https://huggingface.co/models?search=opt)

## ⚙️ Installation

The CUDA version we are using is 12.1.

```
pip install -r requirements.txt
```

## ✂️ Layer Pruning

Our code focuses on using Transformer layers as the lightweight network. The parameter weights of the first pruned layer are inherited for training, as this approach produces better results compared to using FFN or SwiGLU.

▶️ MSE Loss Training (Single GPU)

To train the lightweight network using MSE loss, execute:

```
python mseloss_entry.py
```

This training process will be executed on a single GPU. By default, Llama3.1-8B will be pruned and 8 layers will be removed from the model. All the pre-trained models and the dataset will be automatically downloaded, so you do not need to manually download the resource. When running it for the first time, it will require some time to download the model and the dataset. Please ensure that there is sufficient memory available, as all hidden states will be stored in memory. If memory is insufficient, you may modify the code to store the hidden states on the disk or utilize LLM loss for training.

▶️ LLM Loss Training (Multi-GPU)

To train the lightweight network using LLM loss under the Accelerate and DeepSpeed frameworks, execute:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --config_file 4gpu.yaml llmloss_entry.py
```

This training process will be executed on 4 GPUs. Compared to mse loss, this will require more GPU memory.

We recommend using MSE Loss for training when GPU resources are limited, but when GPU resources are sufficient, using LLM loss will yield better results.

### 🔧 Configuration Parameters

You can change all the Arguments in the LLM_Streamline/args.py file.

- `model_name`: Path to pretrained model or model identifier from huggingface.co/models
- `layer_intervals`: Number of layers to prune.
- `cosine_num_data`: Amount of data used to calculate cosine similarity.
- `train_num_data`: Amount of data used to train the lightweight model.
- `batch_size`: Batch size for training.
- `gradient_accumulation_step`: Number of gradient accumulation steps during training. The effective batch size is the product of gradient_accumulation_step and batch_size.
- `epoches`: Number of training epochs.
- `lr`: Learning rate for training.
- `min_lr`: Minimum learning rate during training.

## 📐 Stability Calculation

To calculate stability, execute:

```
python calculate_stability.py arg1 arg2
```

Here, arg1 refers to the model's evaluation predictions before pruning, and arg2 refers to the predictions after pruning. Both predictions are generated by OpenCompass.

For example:

arg1: "./opencompass/outputs/default/20241121_220629/predictions/llama-3-70b-hf"

arg2: "./opencompass/outputs/default/20241123_220629/predictions/llama-3-70b-hf"
