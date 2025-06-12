import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling
from accelerate import Accelerator
from tqdm import tqdm
from itertools import chain
import time
import gc
import os

# --- Local Imports ---
# Make sure these files are available in the specified path
from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup
from LLM_Streamline.get_cosine import get_cosine_similarity

# ========================================================================================
# --- CONFIGURATION ---
# All major parameters are here for easy modification.
# ========================================================================================
CONFIG = {
    # Model and Pruning Configuration
    "base_model_name": "facebook/opt-1.3b",
    "replace_model_name": "ffn",  # The lightweight model is an FFN
    "layer_intervals": 9,     # Gap between the input layer and target layer for the FFN
    "num_layers_to_check": 16,   # How many layers to check for cosine similarity

    # Dataset Configuration
    "dataset_name": "DKYoon/SlimPajama-6B",
    "train_num_data": 1000,     # Number of samples to create for the final training set
    "cosine_num_data": 50,     # Number of samples to use for the cosine similarity check

    # Training Hyperparameters
    "epochs": 1,
    "batch_size": 8,
    "learning_rate": 2e-4,
    "min_learning_rate": 5e-6,
    "weight_decay": 1e-3,
    "gradient_accumulation_steps": 2,
    "eval_every_n_steps": 10,

    # Output Directory
    "output_dir": "./opt-1.3b-ffn-trained",
}
# ========================================================================================


import torch
import torch.nn as nn
from transformers.activations import ACT2FN

class MLP(nn.Module):
    """
    A flexible Multi-Layer Perceptron that accepts intermediate size and
    an activation function string.
    """
    def __init__(self, hiddensize: int, intermediate_size: int, activation_fn_str: str = "relu"):
        """
        Initializes the MLP.

        Args:
            hiddensize (int): The input and output dimension.
            intermediate_size (int): The dimension of the hidden layer.
            activation_fn_str (str): The name of the activation function to use (e.g., 'relu', 'silu', 'gelu').
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hiddensize, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hiddensize)
        # Use ACT2FN to dynamically get the activation function from its name string
        self.activation = ACT2FN[activation_fn_str]

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

def process_datasets(dataset, train_num_data, tokenizer):
    """
    Creates a custom-mixed and uniformly-sized training dataset from the
    large and diverse SlimPajama dataset by re-weighting its sources.
    """
    proportions = {
        "RedPajamaC4": 0.492, "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3, "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031, "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200,
    }

    filtered_datasets = {name: dataset.filter(lambda x: x.get("meta") == {"redpajama_set_name": f"{name}"}) for name in proportions.keys()}

    test_datasets, train_datasets = [], []
    for name, proportion in proportions.items():
        if len(filtered_datasets[name]) < 2: continue
        
        # Initial split to get a small, separate test set
        split = filtered_datasets[name].train_test_split(test_size=0.01, seed=42)
        test_datasets.append(split["test"])
        
        # --- START: CORRECTED LOGIC ---
        
        # 1. Calculate the desired number of samples as an integer.
        num_samples_to_take = int(train_num_data * proportion)

        # 2. If the calculated number is less than 1, skip this source entirely.
        if num_samples_to_take < 1:
            print(f"Skipping source '{name}' because calculated samples ({num_samples_to_take}) is less than 1.")
            continue
        
        # 3. If we need more samples than are available, just take all available samples.
        if num_samples_to_take > len(split["train"]):
            num_samples_to_take = len(split["train"])

        # 4. Use .select() to directly grab the exact number of samples needed. This is safer than train_test_split.
        train_datasets.append(split["train"].select(range(num_samples_to_take)))

        # --- END: CORRECTED LOGIC ---

    dataset, test_dataset = concatenate_datasets(train_datasets), concatenate_datasets(test_datasets)
    
    # --- Data Cleaning Step ---
    def is_not_empty(example):
        return example.get("text") is not None and len(example.get("text", "")) > 0
    dataset = dataset.filter(is_not_empty)
    test_dataset = test_dataset.filter(is_not_empty)

    tokenizer.pad_token = tokenizer.eos_token
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])
    
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=column_names)
    test_dataset = test_dataset.map(tokenize_function, batched=True, remove_columns=column_names)

    block_size = 2048
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()}
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = dataset.map(group_texts, batched=True)
    test_dataset = test_dataset.map(group_texts, batched=True)
    return dataset, test_dataset

def init_ffn_layer(config, device):
    """Initializes a lightweight FFN layer based on the base model's config."""
    hidden_size = config.hidden_size
    if hasattr(config, 'intermediate_size'):
        ffn_intermediate_size = config.intermediate_size
    elif hasattr(config, 'ffn_dim'):
        ffn_intermediate_size = config.ffn_dim
    else:
        ffn_intermediate_size = config.hidden_size * 4

    # Get the activation function string from the config
    activation_str = getattr(config, 'hidden_act', getattr(config, 'activation_function', 'relu'))
    
    # Now this call will work correctly with the updated MLP class
    return MLP(hidden_size, ffn_intermediate_size, activation_fn_str=activation_str).to(device)


class StreamingHiddenStatesDataset(IterableDataset):
    """
    An iterable dataset that generates (input, target) hidden states on-the-fly
    to avoid storing them all in memory, thus preventing OOM errors.
    """
    def __init__(self, base_model, raw_dataset, tokenizer, config, best_layer, layer_intervals):
        super().__init__()
        self.base_model = base_model
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.config = config
        self.best_layer = best_layer
        self.layer_intervals = layer_intervals

    @torch.no_grad()
    def __iter__(self):
        # Setup model and dataloader within the iterator for multiprocessing compatibility
        device = next(self.base_model.parameters()).device
        self.base_model.eval()
        
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        dataloader = DataLoader(self.raw_dataset, batch_size=CONFIG['batch_size'], collate_fn=data_collator)

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = self.base_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                output_hidden_states=True
            )
            
            # hidden_states[i] is the output of layer i-1
            input_hidden_states = outputs.hidden_states[self.best_layer].cpu()
            target_hidden_states = outputs.hidden_states[self.best_layer + self.layer_intervals].cpu()
            
            # Unbind the batch and yield each sample individually
            for i in range(input_hidden_states.size(0)):
                yield input_hidden_states[i], target_hidden_states[i]
            
            del outputs, input_hidden_states, target_hidden_states, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def valid_model(model, test_dataloader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss, num_batches = 0.0, 0
    progress_bar = tqdm(test_dataloader, desc="Validating", total=len(test_dataloader), disable=not accelerator.is_local_main_process)
    with torch.no_grad():
        for input_data, output_data in progress_bar:
            input_data, output_data = input_data.to(device), output_data.to(device)
            pred = model(input_data)
            loss = loss_fn(pred, output_data)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    script_start_time = time.time()
    accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'])
    
    accelerator.print("="*80)
    accelerator.print(f"🚀 Starting Lightweight Layer Training 🚀")
    accelerator.print(f"Base Model: {CONFIG['base_model_name']}, Dataset: {CONFIG['dataset_name']}")
    accelerator.print("="*80)

    # --- Stage 1: Load Tokenizer and Models ---
    accelerator.print("\n--- Stage 1: Loading Models and Tokenizer ---")
    stage_start_time = time.time()
    
    config = AutoConfig.from_pretrained(CONFIG['base_model_name'])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['base_model_name'])
    base_model = AutoModelForCausalLM.from_pretrained(CONFIG['base_model_name']).to(accelerator.device)
    base_model.eval() # Set to eval mode as it's only used for generating hidden states
    for param in base_model.parameters():
        param.requires_grad = False
        
    accelerator.print(f"✅ Models and Tokenizer loaded in {time.time() - stage_start_time:.2f} seconds.")

    # --- Stage 2: Prepare Dataset ---
    accelerator.print("\n--- Stage 2: Preparing Dataset ---")
    stage_start_time = time.time()
    
    full_dataset = load_dataset(CONFIG['dataset_name'], split="train", streaming=False) # Load full metadata for splitting
    train_dataset_raw, test_dataset_raw = process_datasets(full_dataset, CONFIG['train_num_data'], tokenizer)
    
    accelerator.print(f"✅ Dataset processed in {time.time() - stage_start_time:.2f} seconds.")
    
    # --- Stage 3: Find Best Layer to Prune ---
    accelerator.print("\n--- Stage 3: Finding Best Layer via Cosine Similarity ---")
    stage_start_time = time.time()
    
    best_layer = get_cosine_similarity(
        base_model, train_dataset_raw, 
        CONFIG['cosine_num_data'], accelerator.device, 
        CONFIG['layer_intervals'], CONFIG['num_layers_to_check']
    )
    accelerator.print(f"✅ Best layer found: {best_layer}. Time: {time.time() - stage_start_time:.2f}s")
    
    # --- Stage 4: Initialize Lightweight Model and Dataloaders ---
    accelerator.print("\n--- Stage 4: Initializing Lightweight Model and Streaming Dataloaders ---")
    
    ffn_model = init_ffn_layer(config, accelerator.device)
    
    train_dataset = StreamingHiddenStatesDataset(base_model, train_dataset_raw, tokenizer, config, best_layer, CONFIG['layer_intervals'])
    test_dataset = StreamingHiddenStatesDataset(base_model, test_dataset_raw, tokenizer, config, best_layer, CONFIG['layer_intervals'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    optimizer = torch.optim.AdamW(ffn_model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    
    # Calculate training steps based on the raw dataset size before streaming
    num_training_steps = (len(train_dataset_raw) // CONFIG['batch_size']) * CONFIG['epochs']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * 0.03),
        num_training_steps=num_training_steps,
        max_learning_rate=CONFIG['learning_rate'],
        min_learning_rate=CONFIG['min_learning_rate']
    )

    ffn_model, optimizer, train_dataloader, test_dataloader, scheduler = accelerator.prepare(
        ffn_model, optimizer, train_dataloader, test_dataloader, scheduler
    )
    accelerator.print("✅ Setup complete.")

    # --- Stage 5: Training Loop ---
    accelerator.print("\n--- Stage 5: Starting Training Loop ---")
    
    # --- New Logging: Print training setup details ---
    effective_batch_size = CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']
    accelerator.print(f"Total training steps: {num_training_steps}")
    accelerator.print(f"Epochs: {CONFIG['epochs']}")
    accelerator.print(f"Per-device batch size: {CONFIG['batch_size']}")
    accelerator.print(f"Gradient accumulation steps: {CONFIG['gradient_accumulation_steps']}")
    accelerator.print(f"Effective global batch size: {effective_batch_size}")
    accelerator.print("-" * 40)
    
    training_start_time = time.time()
    criterion = nn.MSELoss()
    best_loss = float('inf')
    total_processed_samples = 0

    for epoch in range(CONFIG['epochs']):
        ffn_model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch} Training", total=num_training_steps, disable=not accelerator.is_local_main_process)
        
        for step, (input_data, output_data) in enumerate(progress_bar):
            # --- New Logging: Print details of the very first training batch ---
            if epoch == 0 and step == 0:
                accelerator.print(f"  [Training] First training batch shape: {input_data.shape}")
                
            with accelerator.accumulate(ffn_model):
                output = ffn_model(input_data)
                loss = criterion(output, output_data)
                
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Update the running total of samples processed
                # This accounts for the samples in the current batch across all devices
                total_processed_samples += input_data.size(0) * accelerator.num_processes
                
                # Update the TQDM progress bar with the current loss
                progress_bar.set_postfix(loss=loss.item())

            # --- Updated Periodic Evaluation and Logging Block ---
            if (step + 1) % CONFIG['eval_every_n_steps'] == 0:
                accelerator.print(f"\n--- Evaluating at Step {step + 1} ---")
                
                valid_loss = valid_model(ffn_model, test_dataloader, accelerator.device)
                current_lr = scheduler.get_last_lr()[0]

                # Use accelerator.print to ensure it only prints on the main process
                accelerator.print(
                    f"  Summary for Step {step + 1}/{num_training_steps}:\n"
                    f"    - Samples Processed: {total_processed_samples}\n"
                    f"    - Current Learning Rate: {current_lr:.2e}\n"
                    f"    - Step Training Loss: {loss.item():.6f}\n"
                    f"    - Validation Loss: {valid_loss:.6f}"
                )
                
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    accelerator.print(f"  ✨ New best validation loss: {best_loss:.6f}. Saving FFN model to {CONFIG['output_dir']}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(ffn_model)
                    if accelerator.is_main_process:
                        os.makedirs(CONFIG['output_dir'], exist_ok=True)
                        torch.save(unwrapped_model.state_dict(), os.path.join(CONFIG['output_dir'], "ffn_layer.pth"))
                
                accelerator.print("--- Finished Evaluation ---\n")
                ffn_model.train() # Set back to train mode

    accelerator.print(f"\n✅ Training finished in {time.time() - training_start_time:.2f} seconds.")
    accelerator.print("="*80)
    accelerator.print(f"Total script run time: {time.time() - script_start_time:.2f} seconds.")
    accelerator.print("🏁 Script finished successfully! 🏁")


if __name__ == "__main__":
    main()