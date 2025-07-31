"""
Data handling module for the Model Foundry framework.

This module provides utilities for preprocessing, chunking, and loading
tokenized datasets for language model training.
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generator
import numpy as np
from datasets import Dataset, load_from_disk, disable_progress_bar
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling
import torch

# Disable progress bars for cleaner output
disable_progress_bar()


class DataProcessor:
    """
    Handles data preprocessing, chunking, and loading for language model training.
    """
    
    def __init__(self, config, base_dir: str):
        self.config = config
        self.base_dir = base_dir
        self.tokenized_data_dir = os.path.join(base_dir, "data", "tokenized", config.experiment_name)
        self.chunked_data_dir = os.path.join(base_dir, "data", "chunked", config.experiment_name)
        
    def _validate_tokenized_dataset(self) -> bool:
        """Validate that the tokenized dataset exists and has the expected structure."""
        if not os.path.exists(self.tokenized_data_dir):
            # Try the training_corpus path directly
            training_corpus_path = os.path.join(self.base_dir, self.config.data.training_corpus)
            if os.path.exists(training_corpus_path):
                print(f"  ✓ Found tokenized dataset at: {training_corpus_path}")
                self.tokenized_data_dir = training_corpus_path
                return True
            else:
                print(f"  ✗ Tokenized dataset not found at: {self.tokenized_data_dir}")
                print(f"  ✗ Also not found at: {training_corpus_path}")
                return False
            
        try:
            dataset = load_from_disk(self.tokenized_data_dir)
            if 'input_ids' not in dataset.column_names:
                print(f"  ✗ Tokenized dataset missing 'input_ids' column")
                return False
            print(f"  ✓ Tokenized dataset loaded successfully")
            print(f"    - Dataset size: {len(dataset):,} examples")
            print(f"    - Columns: {dataset.column_names}")
            return True
        except Exception as e:
            print(f"  ✗ Error loading tokenized dataset: {e}")
            return False
    
    def _chunk_sequences(self, sequences: List[List[int]], chunk_size: int) -> List[List[int]]:
        """
        Chunk sequences into fixed-length blocks.
        
        Args:
            sequences: List of token sequences
            chunk_size: Target chunk size in tokens
            
        Returns:
            List of fixed-length chunks
        """
        chunks = []
        
        for sequence in sequences:
            # Skip sequences that are too short
            if len(sequence) < chunk_size:
                continue
                
            # Create non-overlapping chunks of exactly chunk_size tokens
            for i in range(0, len(sequence) - chunk_size + 1, chunk_size):
                chunk = sequence[i:i + chunk_size]
                chunks.append(chunk)
                
        return chunks
    
    def _create_chunked_dataset(self, tokenized_dataset: Dataset, chunk_size: int) -> Dataset:
        """
        Create a new dataset with fixed-length chunks.
        
        Args:
            tokenized_dataset: Original tokenized dataset
            chunk_size: Target chunk size in tokens
            
        Returns:
            Dataset with fixed-length chunks
        """
        print(f"  - Creating fixed-length chunks (size: {chunk_size})...")
        
        # Extract all sequences
        all_sequences = tokenized_dataset['input_ids']
        
        # Chunk the sequences
        chunked_sequences = self._chunk_sequences(all_sequences, chunk_size)
        
        print(f"    - Original sequences: {len(all_sequences):,}")
        print(f"    - Created chunks: {len(chunked_sequences):,}")
        
        # Create new dataset
        chunked_dataset = Dataset.from_dict({
            'input_ids': chunked_sequences
        })
        
        return chunked_dataset
    
    def _save_chunked_dataset(self, dataset: Dataset) -> None:
        """Save the chunked dataset to disk."""
        os.makedirs(self.chunked_data_dir, exist_ok=True)
        dataset.save_to_disk(self.chunked_data_dir)
        print(f"  ✓ Saved chunked dataset to: {self.chunked_data_dir}")
    
    def _load_chunked_dataset(self) -> Optional[Dataset]:
        """Load the chunked dataset from disk."""
        if not os.path.exists(self.chunked_data_dir):
            return None
            
        try:
            dataset = load_from_disk(self.chunked_data_dir)
            print(f"  ✓ Loaded chunked dataset from: {self.chunked_data_dir}")
            print(f"    - Dataset size: {len(dataset):,} chunks")
            return dataset
        except Exception as e:
            print(f"  ✗ Error loading chunked dataset: {e}")
            return None
    
    def _calculate_dataset_stats(self, dataset: Dataset) -> Dict[str, float]:
        """Calculate statistics about the dataset."""
        sequences = dataset['input_ids']
        
        # Calculate sequence lengths
        lengths = [len(seq) for seq in sequences]
        
        stats = {
            'num_sequences': len(sequences),
            'avg_length': np.mean(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'std_length': np.std(lengths),
            'total_tokens': sum(lengths)
        }
        
        return stats
    
    def preprocess_data(self, force_reprocess: bool = False) -> bool:
        """
        Preprocess the tokenized dataset into fixed-length chunks.
        
        Args:
            force_reprocess: If True, reprocess even if chunked data exists
            
        Returns:
            True if preprocessing was successful
        """
        print(f"--- Data Preprocessing: {self.config.experiment_name} ---")
        
        # Check if chunked data already exists
        if not force_reprocess and os.path.exists(self.chunked_data_dir):
            print(f"  - Chunked dataset already exists at: {self.chunked_data_dir}")
            return True
        
        # Validate tokenized dataset
        if not self._validate_tokenized_dataset():
            return False
        
        # Load tokenized dataset
        tokenized_dataset = load_from_disk(self.tokenized_data_dir)
        
        # Calculate and display original stats
        original_stats = self._calculate_dataset_stats(tokenized_dataset)
        print(f"  - Original dataset statistics:")
        print(f"    - Sequences: {original_stats['num_sequences']:,}")
        print(f"    - Total tokens: {original_stats['total_tokens']:,}")
        print(f"    - Avg length: {original_stats['avg_length']:.1f} tokens")
        
        # Create chunked dataset
        chunk_size = self.config.data.max_sequence_length
        chunked_dataset = self._create_chunked_dataset(tokenized_dataset, chunk_size)
        
        # Calculate and display chunked stats
        chunked_stats = self._calculate_dataset_stats(chunked_dataset)
        print(f"  - Chunked dataset statistics:")
        print(f"    - Chunks: {chunked_stats['num_sequences']:,}")
        print(f"    - Total tokens: {chunked_stats['total_tokens']:,}")
        print(f"    - Chunk size: {chunk_size} tokens (fixed)")
        
        # Save chunked dataset
        self._save_chunked_dataset(chunked_dataset)
        
        return True
    
    def create_dataloader(self, tokenizer) -> DataLoader:
        """
        Create a DataLoader for training.
        
        Args:
            tokenizer: The tokenizer to use for padding
            
        Returns:
            Configured DataLoader
        """
        print(f"  - Creating DataLoader...")
        
        # Load chunked dataset
        dataset = self._load_chunked_dataset()
        if dataset is None:
            raise RuntimeError("Chunked dataset not found. Run preprocessing first.")
        
        # Set up data collator
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8  # For efficiency on modern hardware
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            num_workers=4,  # Adjust based on your system
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        print(f"    - Batch size: {self.config.data.batch_size}")
        print(f"    - Sequence length: {self.config.data.max_sequence_length}")
        print(f"    - Batches per epoch: {len(dataloader)}")
        
        return dataloader
    
    def get_training_steps_per_epoch(self) -> int:
        """
        Calculate the number of training steps per epoch.
        
        Returns:
            Number of steps per epoch
        """
        dataset = self._load_chunked_dataset()
        if dataset is None:
            raise RuntimeError("Chunked dataset not found. Run preprocessing first.")
        
        num_chunks = len(dataset)
        steps_per_epoch = math.ceil(num_chunks / self.config.data.batch_size)
        
        return steps_per_epoch


def create_data_processor(config, base_dir: str) -> DataProcessor:
    """
    Factory function to create a DataProcessor instance.
    
    Args:
        config: Experiment configuration
        base_dir: Project base directory
        
    Returns:
        Configured DataProcessor instance
    """
    return DataProcessor(config, base_dir) 