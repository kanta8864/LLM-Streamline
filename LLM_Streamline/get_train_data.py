import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from tqdm import tqdm
import gc
import os


@torch.no_grad()
def precompute_and_save_data(
    model,
    dataset,
    tokenizer,
    layer_intervals,
    best_layer,
    batch_size,
    save_dir,
    device,
):
    """
    Processes a dataset with a model and saves the specified hidden states
    to disk instead of holding them in memory.
    """
    print(f"Preparing to save pre-computed data to '{save_dir}'...")
    if os.path.exists(save_dir) and os.listdir(os.path.join(save_dir, "input")):
        print(f"Data already exists in '{save_dir}'. Skipping pre-computation.")
        return

    # --- Setup ---
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()

    # --- Create directories for saved tensors ---
    input_save_dir = os.path.join(save_dir, "input")
    output_save_dir = os.path.join(save_dir, "output")
    os.makedirs(input_save_dir, exist_ok=True)
    os.makedirs(output_save_dir, exist_ok=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        shuffle=False,  # Keep order for consistent file naming
        collate_fn=data_collator,
        batch_size=batch_size,
    )
    dataloader = accelerator.prepare(dataloader)

    print(f"Saving tensors to '{save_dir}'...")
    try:
        for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            with torch.no_grad():
                hidden_states = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                ).hidden_states

            input_tensor = hidden_states[best_layer].cpu()
            output_tensor = (
                hidden_states[best_layer + layer_intervals].cpu()
            )

            for i in range(input_tensor.size(0)):
                sample_idx = step * batch_size + i
                torch.save(
                    input_tensor[i],
                    os.path.join(input_save_dir, f"sample_{sample_idx}.pt"),
                )
                torch.save(
                    output_tensor[i],
                    os.path.join(output_save_dir, f"sample_{sample_idx}.pt"),
                )

            del hidden_states, input_tensor, output_tensor

    finally:
        accelerator.free_memory()
        torch.cuda.empty_cache()
        gc.collect()

    print("Finished saving data.")
