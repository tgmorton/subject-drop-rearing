import argparse
import os
import yaml
import glob
import re
import logging
from pathlib import Path
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler, AutoTokenizer
from tqdm.auto import tqdm
import wandb
import random
import numpy as np
import subprocess

# Import the new, refactored components
from .config import ExperimentConfig
from .model import create_model
from .utils import find_project_root, set_seed, get_device, get_git_commit_hash
from .data import create_data_processor
from .logging_utils import setup_logging


# Removed _chunk_examples function - now handled by DataProcessor


class Trainer:
    def __init__(self, config: ExperimentConfig, base_dir: str):
        self.config = config
        self.base_dir = base_dir
        self.device = get_device()
        self.git_commit_hash = get_git_commit_hash()

        # State variables to be initialized
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.dataloader = None
        self.tokenizer = None
        self.global_step = 0
        self.epoch = 0
        
        # Initialize data processor
        self.data_processor = create_data_processor(config, base_dir)

    def _prepare_data(self):
        """Loads and prepares the dataset and dataloader."""
        # Preprocess data into fixed-length chunks
        if not self.data_processor.preprocess_data():
            raise RuntimeError("Data preprocessing failed")
        
        # Create dataloader
        self.dataloader = self.data_processor.create_dataloader(self.tokenizer)

    def _get_checkpoint_schedule(self) -> set:
        """
        Get the checkpoint schedule, either from config or by generating it dynamically.
        
        Returns:
            Set of training steps at which to save checkpoints
        """
        # If auto-generation is enabled and no schedule exists, generate one
        if (self.config.training.auto_generate_checkpoints and 
            not self.config.training.checkpoint_schedule):
            
            print("  - Auto-generating checkpoint schedule...")
            
            # Import the schedule generation function
            from scripts.generate_checkpoint_schedule import (
                generate_checkpoint_schedule, 
                CheckpointGenerationConfig
            )
            
            # Create generation config
            generation_config = CheckpointGenerationConfig(
                first_epoch_checkpoints=self.config.training.first_epoch_checkpoints,
                subsequent_epochs_spacing=self.config.training.subsequent_epochs_spacing,
                log_base=self.config.training.log_base,
                linear_interval=self.config.training.linear_interval,
                min_interval=self.config.training.min_checkpoint_interval
            )
            
            # Generate schedule
            schedule = generate_checkpoint_schedule(
                self.config, 
                self.base_dir, 
                generation_config
            )
            
            # Update the config with the generated schedule
            self.config.training.checkpoint_schedule = schedule
            
            print(f"  - Generated {len(schedule)} checkpoint steps")
        
        # Return as set for efficient lookup
        return set(self.config.training.checkpoint_schedule or [])

    def _save_checkpoint(self):
        """Saves the complete training state to a checkpoint directory."""
        checkpoint_dir = Path(self.base_dir) / self.config.training.output_dir / f"checkpoint-{self.global_step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model and tokenizer
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save training state
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'random_state': random.getstate(),
            'numpy_random_state': np.random.get_state(),
            'torch_random_state': torch.get_rng_state(),
            'torch_cuda_random_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'git_commit_hash': self.git_commit_hash,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")
        print(f"\n  - Saved checkpoint at step {self.global_step} to '{checkpoint_dir}'")

    def _load_checkpoint(self):
        """Loads training state from the latest checkpoint if resume is enabled."""
        output_dir = Path(self.base_dir) / self.config.training.output_dir
        if not self.config.training.resume_from_checkpoint or not output_dir.exists():
            return

        checkpoints = glob.glob(str(output_dir / "checkpoint-*"))
        if not checkpoints:
            print("  - `resume_from_checkpoint` is true, but no checkpoints found. Starting fresh.")
            return

        # Find the checkpoint with the highest step number
        latest_checkpoint = max(checkpoints, key=lambda p: int(re.search(r'checkpoint-(\d+)', p).group(1)))
        print(f"  - Resuming training from latest checkpoint: {latest_checkpoint}")

        # Load tokenizer first as it's needed for model setup
        self.tokenizer = AutoTokenizer.from_pretrained(latest_checkpoint)

        # Load model and move to device
        self.model = create_model(self.config).to(self.device)
        self.model.load_state_dict(torch.load(Path(latest_checkpoint) / "pytorch_model.bin", map_location=self.device))

        # Load training state
        state = torch.load(Path(latest_checkpoint) / "training_state.pt", map_location="cpu")
        self.global_step = state['global_step']
        self.epoch = state['epoch']

        # Restore optimizer and scheduler states
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

        # Restore RNG states
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_random_state'])
        torch.set_rng_state(state['torch_random_state'])
        if torch.cuda.is_available() and state['torch_cuda_random_state']:
            torch.cuda.set_rng_state_all(state['torch_cuda_random_state'])

        print(f"  - Resumed from step {self.global_step} at epoch {self.epoch}.")

    def train(self):
        """Main training loop."""
        # Set up unified logging
        logger = setup_logging("trainer", experiment=self.config.experiment_name, 
                              log_dir=self.config.logging.dir,
                              level=getattr(logging, self.config.logging.level))
        logger.info(f"--- Starting Training Run for: {self.config.experiment_name} ---")

        set_seed(self.config.random_seed)

        # Initialize components
        self.model = create_model(self.config).to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
        )
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=self.config.training.train_steps
        )

        self._load_checkpoint()

        # If not resuming, load tokenizer from its original path
        if self.tokenizer is None:
            tokenizer_path = os.path.join(self.base_dir, self.config.tokenizer.output_dir)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        self._prepare_data()

        # Initialize W&B logging
        if self.config.logging.use_wandb:
            wandb.init(
                project=self.config.logging.wandb_project,
                name=self.config.experiment_name,
                config=self.config.model_dump(),
                resume="allow",
                id=wandb.util.generate_id()
            )

        # Handle checkpoint scheduling
        checkpoint_schedule = self._get_checkpoint_schedule()
        progress_bar = tqdm(range(self.config.training.train_steps), initial=self.global_step, desc="Training Steps")

        self.model.train()
        while self.global_step < self.config.training.train_steps:
            for epoch in range(self.epoch, self.config.training.epochs):
                self.epoch = epoch
                for batch in self.dataloader:
                    if self.global_step >= self.config.training.train_steps:
                        break

                    inputs = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                    if self.config.logging.use_wandb:
                        wandb.log({"loss": loss.item(), "learning_rate": self.lr_scheduler.get_last_lr()[0]},
                                  step=self.global_step)

                    self.global_step += 1
                    progress_bar.update(1)

                    if self.global_step in checkpoint_schedule:
                        self._save_checkpoint()

                if self.global_step >= self.config.training.train_steps:
                    break

        print("\n----- Training Complete -----")
        if self.config.logging.use_wandb:
            wandb.finish()


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

    trainer = Trainer(config, base_dir)
    trainer.train()


if __name__ == '__main__':
    main()