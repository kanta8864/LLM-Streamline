import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets, load_from_disk
from tqdm import tqdm
from itertools import chain
import gc
import os

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
    
# This Dataset does not hold data in memory.
# Instead, it loads each sample from disk one at a time when requested.
class DiskBasedDataset(Dataset):
    def __init__(self, data_dir):
        """
        Initializes the dataset by pointing to a directory of saved tensor files.
        Args:
            data_dir (str): The directory where data files (.pt) are stored.
        """
        self.data_dir = data_dir
        # Get a sorted list of all .pt files in the directory
        self.file_paths = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pt')],
            key=lambda f: int(os.path.splitext(os.path.basename(f))[0]) # Sort numerically
        )
        self.num_samples = len(self.file_paths)
        if self.num_samples == 0:
            raise RuntimeError(f"No data files found in directory: {data_dir}")

    def __len__(self):
        """Returns the total number of samples (files) in the dataset."""
        return self.num_samples

    def __getitem__(self, idx):
        """
        Loads and returns a single data sample from disk.
        Args:
            idx (int): The index of the sample to retrieve.
        Returns:
            A tuple (input_tensor, output_tensor).
        """
        # Load the specific file corresponding to the index
        file_path = self.file_paths[idx]
        input_tensor, output_tensor = torch.load(file_path)
        return input_tensor, output_tensor

# --- NEW: Function to Pre-compute and Save Data ---
def precompute_and_save_data(
    save_dir,
    original_dataset,
    model,
    device,
    layer_intervals,
    best_layer,
    tokenizer,
    inference_batch_size,
):
    """
    Runs the large model over the dataset, saving the resulting hidden states
    to disk chunk by chunk to avoid using too much RAM.
    """
    print(f"Pre-computing and saving data to '{save_dir}'...")
    os.makedirs(save_dir, exist_ok=True)
    
    # Use the existing get_data function, but we will process its output differently.
    # NOTE: get_data must be a generator (`yield`) or process data in chunks
    # if the raw dataset itself is too large to fit in memory.
    # For this example, we assume get_data can return lists, but we will save them immediately.
    
    input_list, output_list = get_data(
        model, original_dataset, device, layer_intervals, best_layer, tokenizer, inference_batch_size
    )
    
    # Save each input/output pair as a separate file
    for i, (input_tensor, output_tensor) in enumerate(tqdm(zip(input_list, output_list), total=len(input_list), desc="Saving samples to disk")):
        # Move tensors to CPU before saving to avoid GPU memory issues
        save_path = os.path.join(save_dir, f"{i}.pt")
        torch.save((input_tensor.cpu(), output_tensor.cpu()), save_path)
        
    print(f"Finished saving {len(input_list)} samples.")

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
        train_split = split["train"].train_test_split(
            test_size=1 - (train_num_data * proportion) / len(split["train"])
        )["train"]
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


def valid_model(model, test_dataloader, device):
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = []

    with torch.no_grad():
        for input_data, output_data in tqdm(test_dataloader, desc="Validating"):
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            
            # The simple MLP models only take one argument
            pred = model(input_data)
            
            loss = loss_fn(pred, output_data)
            total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


def init_layer(model_name, config, device):
    """
    Factory function to initialize the correct lightweight network (MLP)
    based on the model architecture.
    """
    if "llama" in model_name.lower():
        print("Initializing LlamaMLP (SwiGLU) for a Llama-style model.")
        return LlamaMLP(config).to(device)
        
    elif "opt" in model_name.lower():
        print("Initializing standard MLP (fc1/fc2) for an OPT-style model.")
        return MLP(config.hidden_size).to(device)
        
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
):
    # --- STAGE 1: Dataset Processing ---
    dataset_name = "DKYoon/SlimPajama-6B"
    split_name = "train" 
    dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
    dataset, test_dataset = process_datasets(dataset, train_num_data, tokenizer)

    # --- STAGE 2: Find Best Layer ---
    best_layer = get_cosine_similarity(
        model, dataset, cosine_num_data, device, layer_intervals, num_layer
    )
    
    replace_model = init_layer(model_name, config, device)

    # --- STAGE 3: Pre-compute hidden states (if they don't exist) ---
    train_data_dir = "./precomputed_data/train"
    test_data_dir = "./precomputed_data/test"

    # Use a large batch size for efficient GPU utilization during pre-computation
    inference_batch_size = 1

    if not os.path.exists(train_data_dir) or not os.listdir(train_data_dir):
        precompute_and_save_data(
            train_data_dir, dataset, model, device, layer_intervals, 
            best_layer, tokenizer, inference_batch_size
        )
    else:
        print(f"Found existing pre-computed training data in '{train_data_dir}'. Skipping pre-computation.")

    if not os.path.exists(test_data_dir) or not os.listdir(test_data_dir):
        precompute_and_save_data(
            test_data_dir, test_dataset, model, device, layer_intervals, 
            best_layer, tokenizer, inference_batch_size
        )
    else:
        print(f"Found existing pre-computed test data in '{test_data_dir}'. Skipping pre-computation.")


    # --- STAGE 4: Load Disk-Based Datasets and Train ---
    print("Loading datasets from disk...")
    train_disk_dataset = DiskBasedDataset(train_data_dir)
    test_disk_dataset = DiskBasedDataset(test_data_dir)
    
    print(f"Successfully loaded {len(train_disk_dataset)} training samples and {len(test_disk_dataset)} test samples.")

    test_dataloader = DataLoader(
        test_disk_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=8, 
        pin_memory=True
    )
    train_dataloader = DataLoader(
        train_disk_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(replace_model.parameters(), lr=lr, weight_decay=wd)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) * epochs * 0.01 * 0.5,
        num_training_steps=len(train_dataloader) * epochs * 0.5,
        max_learning_rate=lr,
        min_learning_rate=min_lr,
    )

    best_loss = valid_model(replace_model, test_dataloader, device)
    print("Before training, Validation_Loss:", best_loss)
    print("Starting training...")
    best_state_dict = None

    for epoch in range(epochs):
        replace_model.train()
        step = 0
        optimizer.zero_grad()

        for input_data, output_data in tqdm(train_dataloader, desc=f"Epoch {epoch}"):
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            
            # Simple MLP models only need the input tensor
            output = replace_model(input_data)
            
            loss = criterion(output, output_data)
            loss /= gradient_accumulation_step
            loss.backward()

            if (step + 1) % gradient_accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(replace_model.parameters(), max_norm=5)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            del input_data, output_data, output, loss
            torch.cuda.empty_cache()

            step += 1

        valid_loss = valid_model(replace_model, test_dataloader, device)

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state_dict = replace_model.state_dict()

        print(f"Epoch: {epoch}, Validation Loss: {valid_loss:.6f}")

        torch.cuda.empty_cache()
        gc.collect()

    replace_model.load_state_dict(best_state_dict)

    return replace_model, best_layer