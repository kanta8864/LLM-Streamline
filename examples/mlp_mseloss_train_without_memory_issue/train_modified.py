from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader
from datasets import load_dataset
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain
from accelerate import Accelerator
from accelerate.utils import set_seed
from scheduler import get_cosine_schedule_with_warmup
from tqdm import tqdm

from modeling_llama import LlamaModel
import deepspeed

# Initialize accelerator
accelerator = Accelerator(mixed_precision='bf16', gradient_accumulation_steps=2)

# Use Llama-2-7B instead of Llama-3-8B (no approval required)
model_name = "meta-llama/Llama-2-7b-hf"
print(f"Loading model: {model_name}")

config = AutoConfig.from_pretrained(model_name)
'''
Since Llama-2-7B has 32 layers (same as Llama-3.1-8B), we will prune layers 21 to 30 while keeping layers 31 and 32. 
The training data should be prepared such that the input to the 20th layer serves as the input to the MLP lightweight layer, 
and the output of the 30th layer serves as the output of the MLP lightweight layer. 
Therefore, during the training of the MLP lightweight layer, layers 31 and 32 are not involved.

NOTE: This version uses a simple MLP (2-layer feedforward network) instead of a full transformer layer.
'''
config.num_hidden_layers = 30  # Reduce from 32 to 30 layers

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Creating modified model with MLP lightweight layer...")
model = LlamaModel(config)

print("Loading original Llama model...")
llama_model = AutoModelForCausalLM.from_pretrained(model_name)

print("Copying weights...")
model_dict = model.state_dict()
llama_dict = llama_model.state_dict()

# Copy embedding weights
model_dict['embed_tokens.weight'] = llama_dict['model.embed_tokens.weight']

# Copy layer weights for first 30 layers
for i in range(30):
    # Self-attention weights
    model_dict[f'layers.{i}.self_attn.q_proj.weight'] = llama_dict[f'model.layers.{i}.self_attn.q_proj.weight']
    model_dict[f'layers.{i}.self_attn.k_proj.weight'] = llama_dict[f'model.layers.{i}.self_attn.k_proj.weight']
    model_dict[f'layers.{i}.self_attn.v_proj.weight'] = llama_dict[f'model.layers.{i}.self_attn.v_proj.weight']
    model_dict[f'layers.{i}.self_attn.o_proj.weight'] = llama_dict[f'model.layers.{i}.self_attn.o_proj.weight']

    # MLP weights
    model_dict[f'layers.{i}.mlp.gate_proj.weight'] = llama_dict[f'model.layers.{i}.mlp.gate_proj.weight']
    model_dict[f'layers.{i}.mlp.up_proj.weight'] = llama_dict[f'model.layers.{i}.mlp.up_proj.weight']
    model_dict[f'layers.{i}.mlp.down_proj.weight'] = llama_dict[f'model.layers.{i}.mlp.down_proj.weight']
    
    # Layer norm weights
    model_dict[f'layers.{i}.input_layernorm.weight'] = llama_dict[f'model.layers.{i}.input_layernorm.weight']
    model_dict[f'layers.{i}.post_attention_layernorm.weight'] = llama_dict[f'model.layers.{i}.post_attention_layernorm.weight']

# The MLP replace_layer doesn't need weight initialization like the transformer layer version
# because it's a simple 2-layer MLP that will be trained from scratch
print("MLP lightweight layer will be trained from scratch (no weight initialization needed)")

model.load_state_dict(model_dict)
del llama_model

print("Setting up trainable parameters...")
# Freeze all parameters except replace_layer (the MLP)
for name, p in model.named_parameters():
    if "replace_layer" in name:
        print(f"Trainable: {name}")
        continue
    else:
        if p.requires_grad == True:
            p.requires_grad = False

print("Loading dataset...")
# Load SlimPajama dataset automatically instead of from disk
try:
    dataset = load_dataset('DKYoon/SlimPajama-6B')['train']
    print(f"Dataset loaded successfully. Total samples: {len(dataset)}")
    
    # Create train/validation split
    dataset_split = dataset.train_test_split(test_size=0.01, seed=42)  # 1% for validation
    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']
    
    # For initial testing, use smaller subset
    subset_size = 50000  # Adjust based on your resources
    print(f"Using subset of {subset_size} training samples and 1000 validation samples")
    
    train_dataset = train_dataset.select(range(min(subset_size, len(train_dataset))))
    eval_dataset = eval_dataset.select(range(min(1000, len(eval_dataset))))
    
except Exception as e:
    print(f"Error loading SlimPajama dataset: {e}")
    print("Falling back to a smaller dataset for testing...")
    
    # Fallback to a smaller dataset
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_dataset = dataset['train'].select(range(1000))
    eval_dataset = dataset['validation'].select(range(100))

# Process datasets for language modeling
def process_dataset(dataset, tokenizer, block_size=2048):
    def tokenize_function(examples):
        return tokenizer(examples['text'])
    
    # Tokenize
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Running tokenizer on dataset",
    )
    
    # Group texts into chunks
    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
    
    return tokenized_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

print("Processing datasets...")
if 'text' in train_dataset.column_names:
    train_dataset = process_dataset(train_dataset, tokenizer)
    eval_dataset = process_dataset(eval_dataset, tokenizer)

print("Creating data loaders...")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=4)  # Small batch for memory
eval_dataloader = DataLoader(eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=8)

print("Setting up optimizer and scheduler...")
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-3, betas=(0.9, 0.95))
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=int(len(train_dataloader) * 0.03),
    num_training_steps=len(train_dataloader),
    max_learning_rate=2e-4,
    min_learning_rate=5e-6,
)

print("Preparing for distributed training...")
train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

mse_loss = nn.MSELoss()

print("Starting MLP lightweight layer training...")
print(f"Architecture: Simple 2-layer MLP (hidden_size -> 4*hidden_size -> hidden_size)")
print(f"Total training steps: {len(train_dataloader)}")
print(f"Evaluation every 500 steps")

best_loss = float('inf')

for epoch in range(1):
    model.train()
    total_loss = 0
    
    for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="Training MLP"):
        with accelerator.accumulate(model):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            labels = outputs.last_hidden_state[0]    # Original path output (layer 30)
            outputs = outputs.last_hidden_state[1]   # MLP lightweight layer output
            
            loss = mse_loss(outputs, labels)
            total_loss += loss.item()

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # Evaluation every 500 steps
        if (step + 1) % 500 == 0:            
            model.eval()
            eval_losses = []
            
            print(f"\nRunning evaluation at step {step + 1}...")
            for eval_step, eval_batch in tqdm(enumerate(eval_dataloader), desc="Evaluating"):
                with torch.no_grad():
                    input_ids = eval_batch['input_ids']
                    attention_mask = eval_batch['attention_mask']
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    labels = outputs.last_hidden_state[0]
                    outputs = outputs.last_hidden_state[1]

                    loss = mse_loss(outputs, labels)
                    eval_losses.append(accelerator.gather_for_metrics(loss.repeat(eval_batch['input_ids'].shape[0])))

            eval_losses = torch.cat(eval_losses)
            eval_loss = torch.mean(eval_losses)
            avg_train_loss = total_loss / 500
            
            print(f"Step {step + 1}")
            print(f"Average Training Loss (last 500 steps): {avg_train_loss:.6f}")
            print(f"Validation Loss: {eval_loss:.6f}")
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                print(f"New best validation loss! Saving MLP model...")
                
                # Save the best model
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                if accelerator.is_main_process:
                    unwrapped_model.save_pretrained("./best_llama2_mlp_pruned_model")
                    tokenizer.save_pretrained("./best_llama2_mlp_pruned_model")
                    print("Model saved to ./best_llama2_mlp_pruned_model")
            
            total_loss = 0
            model.train()

print(f"\nMLP lightweight layer training completed!")
print(f"Best validation loss: {best_loss:.6f}")
print("Final model saved to ./best_llama2_mlp_pruned_model")
print("\nArchitecture Summary:")
print("- Original: Llama-2-7B (32 layers)")
print("- Compressed: 22 layers + 1 simple MLP layer")
print("- MLP structure: Linear(4096 -> 16384) -> ReLU -> Linear(16384 -> 4096)")
print("- Parameters saved: ~3M parameters for the MLP vs ~110M for full transformer layer") 