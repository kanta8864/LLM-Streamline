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
            print(
                f"INFO: For '{name}', taking all {available_train_samples} available samples for training."
            )
            train_split = split["train"]
        else:
            # Calculate the test set size needed to leave the desired number of train samples
            test_size_ratio = 1.0 - (desired_train_samples / available_train_samples)
            train_split = split["train"].train_test_split(test_size=test_size_ratio)[
                "train"
            ]

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
        for input_data, output_data in tqdm(test_dataloader):
            input_data = input_data.to(device)
            output_data = output_data.to(device)
            pred = model(input_data)
            if isinstance(pred, tuple):
                pred = pred[0]
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
        raise NotImplementedError(
            f"No lightweight network implementation for model type: {model_name}"
        )


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
    use_subset=True,  # Add this parameter
    subset_size=3000,  # Add this parameter
):
    # --- STAGE 1: Dataset Processing ---
    dataset_name = "DKYoon/SlimPajama-6B"
    split_name = "train"

    if use_subset:
        # Option 1: Load streaming dataset and take first N examples
        print(f"Loading subset of {subset_size} examples...")
        dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)
        subset_size = 10000
        dataset = dataset.select(range(subset_size))

        # Alternative Option 2: Load a percentage of the full dataset
        # dataset = load_dataset(dataset_name, split="train[:1%]", trust_remote_code=True)

    else:
        # Original full dataset loading
        dataset = load_dataset(dataset_name, split=split_name, trust_remote_code=True)

    # Get actual dataset size and adjust train_num_data
    actual_dataset_size = len(dataset)
    print(f"Actual dataset size: {actual_dataset_size}")

    # Ensure train_num_data doesn't exceed 80% of actual dataset size
    max_train_size = int(actual_dataset_size * 0.8)
    train_num_data = min(train_num_data, max_train_size)

    print(f"Using train_num_data: {train_num_data} (max possible: {max_train_size})")

    # Additional safety check
    if train_num_data >= actual_dataset_size:
        raise ValueError(
            f"train_num_data ({train_num_data}) must be less than dataset size ({actual_dataset_size})"
        )

    dataset, test_dataset = process_datasets(dataset, train_num_data, tokenizer)

    # --- STAGE 2: Find Best Layer ---
    best_layer = get_cosine_similarity(
        model, dataset, cosine_num_data, device, layer_intervals, num_layer
    )

    replace_model = init_layer(model_name, config, device)

    # --- STAGE 3: Pre-compute hidden states (if they don't exist) ---
    train_data_dir = "./precomputed_data/train"
    test_data_dir = "./precomputed_data/test"

    def prepare_dataset_for_training(dataset, model, device):
        input_list, output_list = get_data(
            model, dataset, device, layer_intervals, best_layer, tokenizer, batch_size
        )
        return CustomDataset(input_list, output_list)

    print(
        f"The test dataset has {len(test_dataset)} samples, so there will be {len(test_dataset)/batch_size} iterations."
    )

    test_dataset = prepare_dataset_for_training(test_dataset, model, device)

    # Add this line to see the size of the next dataset:
    print(
        f"The training dataset has {len(dataset)} samples, so there will be {len(dataset)/batch_size} iterations."
    )
    train_dataset = prepare_dataset_for_training(dataset, model, device)

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    print("test data loader completed")

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    print("train data loader completed")

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
