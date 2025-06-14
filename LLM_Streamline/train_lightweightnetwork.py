import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader,IterableDataset
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from itertools import chain
import gc
import os
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, DataCollatorForLanguageModeling


from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer



from LLM_Streamline.scheduler import get_cosine_schedule_with_warmup
from LLM_Streamline.get_cosine import get_cosine_similarity
from LLM_Streamline.get_train_data import get_data
from LLM_Streamline.modeling_llama import LlamaMLP 
from LLM_Streamline.modeling_mlp import MLP



class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data

    def __getitem__(self, index):
        return (
            self.input_data[index].clone().detach(),
            self.output_data[index].clone().detach(),
        )

    def __len__(self):
        return len(self.input_data)
class StreamingHiddenStatesDataset(IterableDataset):
    """
    An iterable dataset that generates (input, target) hidden states on-the-fly,
    correctly managing the torch.no_grad() context to prevent it from
    leaking into the training loop.
    """
    def __init__(self, base_model, raw_text_dataset, tokenizer, device, layer_intervals, best_layer, batch_size):
        super().__init__()
        self.base_model = base_model
        self.raw_text_dataset = raw_text_dataset
        self.tokenizer = tokenizer
        self.device = device
        self.layer_intervals = layer_intervals
        self.best_layer = best_layer
        self.batch_size = batch_size

    def __iter__(self):
        data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        source_dataloader = DataLoader(self.raw_text_dataset, batch_size=self.batch_size, collate_fn=data_collator)

        self.base_model.eval()

        # Iterate over batches of raw text
        for text_batch in source_dataloader:
            # Prepare a list to hold the generated tensors for this batch
            generated_pairs = []

            # Use torch.no_grad() ONLY for the expensive base_model inference
            with torch.no_grad():
                text_batch = {k: v.to(self.device) for k, v in text_batch.items()}

                outputs = self.base_model(
                    input_ids=text_batch['input_ids'],
                    attention_mask=text_batch['attention_mask'],
                    output_hidden_states=True
                )

                input_hidden_states = outputs.hidden_states[self.best_layer]
                target_hidden_states = outputs.hidden_states[self.best_layer + self.layer_intervals]

                # Move generated tensors to CPU and add to a temporary list
                for i in range(input_hidden_states.size(0)):
                    inp = input_hidden_states[i].clone().detach().cpu()
                    out = target_hidden_states[i].clone().detach().cpu()
                    generated_pairs.append((inp, out))

                # Clean up GPU memory immediately after inference
                del outputs, text_batch, input_hidden_states, target_hidden_states
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # <<< THE CRITICAL FIX >>>
            # Now, OUTSIDE the no_grad() context, yield the data.
            for pair in generated_pairs:
                yield pair
                
def process_datasets(dataset, train_num_data, tokenizer):
    """
    We divided the proportions of RedPajamaCommonCrawl, RedPajamaArXiv,
    and RedPajamaBook by a normalization value because the data length
    in these domains is higher than in other domains.
    """
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200,
    }

    filtered_datasets = {
        name: dataset.filter(lambda x: x["meta"] == {"redpajama_set_name": f"{name}"})
        for name in proportions.keys()
    }

    test_datasets = []
    train_datasets = []

    for name, proportion in proportions.items():
        split = filtered_datasets[name].train_test_split(
            test_size=(3000 * proportion) / len(filtered_datasets[name])
        )
        test_datasets.append(split["test"])
        # Calculate how many training samples we *want* for this slice
        desired_train_samples = int(train_num_data * proportion)
        
        # Check how many are actually available in the current split
        available_train_samples = len(split["train"])

        # If we want more samples than are available, just take all of them.
        # Otherwise, split the data to get the number we want.
        if desired_train_samples >= available_train_samples:
            print(f"INFO: For '{name}', taking all {available_train_samples} available samples for training.")
            train_split = split["train"]
        else:
            # Calculate the test set size needed to leave the desired number of train samples
            test_size_ratio = 1.0 - (desired_train_samples / available_train_samples)
            train_split = split["train"].train_test_split(test_size=test_size_ratio)["train"]

        train_datasets.append(train_split)

    dataset, test_dataset = concatenate_datasets(train_datasets), concatenate_datasets(
        test_datasets
    )

    tokenizer.pad_token = tokenizer.eos_token

    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    test_dataset = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )

    block_size = 2048

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

    dataset = dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    test_dataset = test_dataset.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return dataset, test_dataset


def valid_model(model, test_dataloader, device, num_validation_steps):
    """
    Validates the model.
    Accepts num_validation_steps to work with IterableDatasets.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = []

    # Use the manually calculated total for the progress bar
    progress_bar = tqdm(test_dataloader, total=num_validation_steps, desc="Validating")

    with torch.no_grad():
        for input_data, output_data in progress_bar:
            input_data = input_data.to(device)  # Keep original dtype from quantized model
            output_data = output_data.to(device)  # Keep original dtype from quantized model
            pred = model(input_data.float())  # Convert input to float32 for the lightweight network
            if isinstance(pred, tuple):
                pred = pred[0]
            # Convert target to float32 for stable loss computation
            loss = loss_fn(pred, output_data.float())
            total_loss.append(loss.item())

    if not total_loss:
        return float('inf') # Return infinity if validation set was empty
        
    return sum(total_loss) / len(total_loss)

def init_layer(model_name, config, device):
    """
    Factory function to initialize the correct lightweight network (MLP)
    based on the model architecture.
    """
    if "llama" in model_name.lower():
        print("Initializing LlamaMLP (SwiGLU) for a Llama-style model.")
        return LlamaMLP(config).to(device)  # Keep in float32 for stable training
        
    elif "opt" in model_name.lower():
        print("Initializing standard MLP (fc1/fc2) for an OPT-style model.")
        return MLP(config.hidden_size).to(device)  # Keep in float32 for stable training
        
    else:
        raise NotImplementedError(f"No lightweight network implementation for model type: {model_name}")

def lightweight_model_train(
    model,
    tokenizer,
    device,
    layer_intervals,
    num_layer,
    cosine_num_data,
    train_num_data,
    batch_size,
    epochs,
    lr,
    min_lr,
    wd,
    config,
    model_name,
    gradient_accumulation_step,
    use_subset=False,
    subset_size=10000,
):
    dataset_name = "DKYoon/SlimPajama-6B"
    split_name = "train"

    # Stage 1: Process the raw text datasets
    print("Processing raw text datasets...")
    dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
    dataset, test_dataset = process_datasets(dataset, train_num_data, tokenizer)

    print("Raw train dataset size:", len(dataset))
    print("Raw test dataset size:", len(test_dataset))

    # Stage 2: Find the best layer to replace
    print("Finding best layer via cosine similarity...")
    best_layer = get_cosine_similarity(
        model, dataset, cosine_num_data, device, layer_intervals, num_layer
    )
    print(f"Best layer to start replacement: {best_layer}")

    print(f"Base model is already on the correct device (8-bit quantized models handle device placement automatically)")

    # Stage 3: Initialize the lightweight model
    replace_model = init_layer(model_name, config, device)

    # replace_model = replace_model.to(torch.bfloat16)

    # Stage 4: Setup streaming datasets and dataloaders
    print("Setting up streaming dataloaders...")
    train_dataset_streaming = StreamingHiddenStatesDataset(model, dataset, tokenizer, device, layer_intervals, best_layer, batch_size)
    test_dataset_streaming = StreamingHiddenStatesDataset(model, test_dataset, tokenizer, device, layer_intervals, best_layer, batch_size)

    train_dataloader = DataLoader(train_dataset_streaming, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset_streaming, batch_size=batch_size, shuffle=False)

    # --- FIX #1: Manually calculate step counts ---
    # This is required because an IterableDataset has no len().
    num_training_steps = len(dataset) // batch_size
    num_validation_steps = len(test_dataset) // batch_size
    if len(test_dataset) % batch_size != 0:
        num_validation_steps += 1 # Account for the last, smaller batch
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(replace_model.parameters(), lr=lr, weight_decay=wd)
    
    # --- FIX #2: Use the manually calculated num_training_steps for the scheduler ---
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * epochs * 0.01),
        num_training_steps=num_training_steps * epochs,
        max_learning_rate=lr,
        min_learning_rate=min_lr,
    )

    print("Verifying and enabling gradients for the lightweight model...")
    for name, param in replace_model.named_parameters():
        param.requires_grad = True
        if not param.requires_grad:
             print(f"Warning: Failed to enable gradient for {name}")
    print("✅ Gradients enabled.")

    # Stage 6: Training loop
    print("Starting training with streaming data...")
    best_loss = float('inf')
    best_state_dict = None
    
    # Create checkpoint directory
    checkpoint_dir = f"./checkpoints_{model_name.replace('/', '_')}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    for epoch in range(epochs):
        replace_model.train()
        progress_bar = tqdm(train_dataloader, total=num_training_steps, desc=f"Epoch {epoch}")
        
        for step, (input_data, output_data) in enumerate(progress_bar):
            input_data = input_data.to(device)  # Keep original dtype from quantized model
            output_data = output_data.to(device)  # Keep original dtype from quantized model
            
            # Convert input to float32 for the lightweight network
            output = replace_model(input_data.float())
            # Convert target to float32 for loss computation
            loss = criterion(output, output_data.float())

            # --- FIX #3 (Logic Correction): Normalize loss for gradient accumulation ---
            loss = loss / gradient_accumulation_step
            
            loss.backward()

            if (step + 1) % gradient_accumulation_step == 0 or (step + 1) == num_training_steps:
                torch.nn.utils.clip_grad_norm_(replace_model.parameters(), max_norm=5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix(loss=loss.item() * gradient_accumulation_step) # Display un-normalized loss

        print("Epoch finished. Running validation...")
        # --- FIX #4: Pass num_validation_steps to the validation function ---
        valid_loss = valid_model(replace_model, test_dataloader, device, num_validation_steps)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = replace_model.state_dict()
            print(f"New best validation loss: {valid_loss:.6f}. Best checkpoint saved.")

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"lightweight_network_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': replace_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'validation_loss': valid_loss,
                'best_layer': best_layer,
                'config': config
            }, checkpoint_path)
            print(f"✅ Checkpoint saved at epoch {epoch + 1}: {checkpoint_path}")

        print(f"Epoch: {epoch}, Validation Loss: {valid_loss:.6f}")

        torch.cuda.empty_cache()
        gc.collect()

    print("Training complete. Loading best model weights.")
    if best_state_dict is not None:
        replace_model.load_state_dict(best_state_dict)
    else:
        print("Warning: No best model was saved. Returning the model from the last epoch.")

    return replace_model, best_layer