import argparse
import os
import yaml
import math
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, DataCollatorForLanguageModeling
from datasets import load_from_disk
from tqdm.auto import tqdm
import wandb

# Import the new, refactored components
from .config import ExperimentConfig
from .model import create_model
from .utils import find_project_root, set_seed, get_device


def _chunk_examples(batch, block_size: int):
    """Concatenates and chunks examples to a fixed block size."""
    concatenated_examples = {k: sum(batch[k], []) for k in batch.keys()}
    total_length = len(concatenated_examples[list(batch.keys())[0]])
    if total_length < block_size:
        return {k: [] for k in batch.keys()}
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def run_training(config: ExperimentConfig, base_dir: str):
    """Main function to orchestrate the model training process."""
    print(f"--- Starting Training Run for: {config.experiment_name} ---")

    # 1. Setup Environment
    set_seed(config.random_seed)
    device = get_device()

    # 2. Initialize Logging (W&B)
    if config.logging.use_wandb:
        wandb.init(
            project=config.logging.wandb_project,
            name=config.experiment_name,
            config=config.model_dump()
        )
        print(f"  - Weights & Biases logging enabled for project '{config.logging.wandb_project}'.")

    # 3. Load and Process Dataset
    tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", config.experiment_name)
    print(f"  - Loading tokenized dataset from: {tokenized_data_dir}")
    tokenized_dataset = load_from_disk(tokenized_data_dir)

    print("  - Chunking dataset into fixed-length blocks...")
    chunked_dataset = tokenized_dataset.map(
        _chunk_examples,
        batched=True,
        fn_kwargs={'block_size': config.data.max_sequence_length},
        remove_columns=tokenized_dataset.column_names,
    )
    print(f"  - Dataset chunked into {len(chunked_dataset):,} samples of size {config.data.max_sequence_length}.")

    data_collator = DataCollatorForLanguageModeling(tokenizer=None, mlm=False)
    dataloader = DataLoader(
        chunked_dataset,
        batch_size=config.data.batch_size,
        collate_fn=data_collator,
        shuffle=True
    )

    # 4. Create Model, Optimizer, and Scheduler
    model = create_model(config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        eps=config.training.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=config.training.warmup_steps,
        num_training_steps=config.training.train_steps
    )

    # 5. Training Loop
    output_dir = os.path.join(base_dir, config.training.output_dir)
    checkpoint_schedule = set(config.training.checkpoint_schedule)
    progress_bar = tqdm(range(config.training.train_steps), desc="Training Steps")
    global_step = 0

    model.train()
    for epoch in range(config.training.epochs):
        print(f"\n--- Starting Epoch {epoch + 1}/{config.training.epochs} ---")
        for batch in dataloader:
            if global_step >= config.training.train_steps: break

            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if config.logging.use_wandb:
                wandb.log({"loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, step=global_step)

            global_step += 1
            progress_bar.update(1)

            if global_step in checkpoint_schedule:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                print(f"\n  - Saved scheduled checkpoint at step {global_step} to '{checkpoint_dir}'")

        if global_step >= config.training.train_steps: break

    print("\n----- Training Complete -----")
    if config.logging.use_wandb: wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Run the main training loop for an experiment.")
    parser.add_argument("config_path", type=str, help="Path to the experiment's .yaml config file.")
    args = parser.parse_args()

    base_dir = find_project_root(__file__)
    abs_config_path = args.config_path if os.path.isabs(args.config_path) else os.path.join(base_dir, args.config_path)

    with open(abs_config_path, 'r') as f:
        config_data = yaml.safe_load(f)

    try:
        config = ExperimentConfig(**config_data)
    except Exception as e:
        print(f"FATAL: Error validating configuration file '{abs_config_path}':\n{e}")
        return

    run_training(config, base_dir)


if __name__ == '__main__':
    main()