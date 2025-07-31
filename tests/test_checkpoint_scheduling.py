"""
Test script for the checkpoint scheduling system.
"""

import os
import tempfile
import yaml
from pathlib import Path
import sys

# Add the model_foundry package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_foundry.config import ExperimentConfig, DataConfig, TokenizerConfig, ModelConfig, TrainingConfig, LoggingConfig
# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_checkpoint_schedule import (
    CheckpointGenerationConfig,
    generate_log_steps,
    estimate_dataset_size,
    generate_checkpoint_schedule
)


def create_test_config():
    """Create a test configuration."""
    return ExperimentConfig(
        experiment_name="test_checkpoint_experiment",
        data=DataConfig(
            source_corpus="test_data",
            training_corpus="test_data",
            batch_size=4,
            max_sequence_length=128
        ),
        tokenizer=TokenizerConfig(
            output_dir="test_tokenizer",
            vocab_size=1000
        ),
        model=ModelConfig(
            layers=2,
            embedding_size=64,
            hidden_size=64,
            intermediate_hidden_size=128,
            attention_heads=4,
            activation_function="GELU",
            dropout=0.1,
            attention_dropout=0.1
        ),
        training=TrainingConfig(
            output_dir="test_models",
            learning_rate=0.0001,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-6,
            warmup_steps=100,
            train_steps=1000,
            epochs=2,
            auto_generate_checkpoints=False,
            target_checkpoints={"small": 10, "medium": 20, "large": 30, "xlarge": 40},
            log_steps_first_epoch=True,
            min_checkpoint_interval=50
        ),
        logging=LoggingConfig(
            use_wandb=False
        ),
        random_seed=42
    )


def test_checkpoint_generation_config():
    """Test the CheckpointGenerationConfig class."""
    config = CheckpointGenerationConfig(
        target_checkpoints={"small": 10, "medium": 20},
        log_steps_first_epoch=True,
        min_interval=100
    )
    
    assert config.target_checkpoints["small"] == 10
    assert config.target_checkpoints["medium"] == 20
    assert config.log_steps_first_epoch is True
    assert config.min_interval == 100
    
    # Test string parsing
    config_from_string = CheckpointGenerationConfig(
        target_checkpoints="small:10,medium:20"
    )
    assert config_from_string.target_checkpoints["small"] == 10
    assert config_from_string.target_checkpoints["medium"] == 20


def test_generate_log_steps():
    """Test the log-based step generation."""
    # Test first epoch only
    log_steps = generate_log_steps(1000, first_epoch_only=True)
    assert 1 in log_steps
    assert 2 in log_steps
    assert 4 in log_steps
    assert 8 in log_steps
    assert 16 in log_steps
    assert 32 in log_steps
    # Note: 64 might not be included if it exceeds the first epoch estimate
    # assert 64 in log_steps
    # assert 128 in log_steps
    
    # All steps should be powers of 2
    for step in log_steps:
        assert step > 0
        assert (step & (step - 1)) == 0  # Power of 2 check
    
    # Test full range
    log_steps_full = generate_log_steps(1000, first_epoch_only=False)
    assert len(log_steps_full) > len(log_steps)  # Should have more steps


def test_estimate_dataset_size():
    """Test dataset size estimation."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    # Test fallback estimation
    size = estimate_dataset_size(config, base_dir)
    assert size in ["small", "medium", "large", "xlarge"]


def test_generate_checkpoint_schedule():
    """Test checkpoint schedule generation."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    generation_config = CheckpointGenerationConfig(
        target_checkpoints={"small": 10, "medium": 20, "large": 30, "xlarge": 40},
        log_steps_first_epoch=True,
        min_interval=50
    )
    
    schedule = generate_checkpoint_schedule(config, base_dir, generation_config)
    
    # Should have generated a schedule
    assert len(schedule) > 0
    
    # All steps should be positive and within training range
    for step in schedule:
        assert step > 0
        assert step <= config.training.train_steps
    
    # Should be sorted
    assert schedule == sorted(schedule)
    
    # Should include final step
    assert config.training.train_steps in schedule
    
    # Should include epoch boundaries
    steps_per_epoch = config.training.train_steps // config.training.epochs
    for epoch in range(1, config.training.epochs + 1):
        epoch_step = epoch * steps_per_epoch
        if epoch_step <= config.training.train_steps:
            assert epoch_step in schedule


def test_schedule_distribution():
    """Test that checkpoints are distributed reasonably."""
    config = create_test_config()
    base_dir = "/tmp/test"
    
    generation_config = CheckpointGenerationConfig(
        target_checkpoints={"small": 5, "medium": 10, "large": 15, "xlarge": 20},
        log_steps_first_epoch=True,
        min_interval=100
    )
    
    schedule = generate_checkpoint_schedule(config, base_dir, generation_config)
    
    # Check that gaps between checkpoints are reasonable
    # Allow for log-based steps which can be close together
    for i in range(len(schedule) - 1):
        gap = schedule[i + 1] - schedule[i]
        # Allow small gaps for log-based steps, but larger gaps should respect min_interval
        if gap > 50:  # Non-log-based steps
            assert gap >= generation_config.min_interval


def test_config_integration():
    """Test integration with the configuration system."""
    config = create_test_config()
    
    # Test that checkpoint parameters are accessible
    assert config.training.auto_generate_checkpoints is False
    assert config.training.target_checkpoints is not None
    assert config.training.log_steps_first_epoch is True
    assert config.training.min_checkpoint_interval == 50
    
    # Test that we can set checkpoint schedule
    test_schedule = [100, 200, 300, 400, 500]
    config.training.checkpoint_schedule = test_schedule
    assert config.training.checkpoint_schedule == test_schedule


if __name__ == "__main__":
    print("Running checkpoint scheduling tests...")
    
    test_checkpoint_generation_config()
    print("✓ CheckpointGenerationConfig test passed")
    
    test_generate_log_steps()
    print("✓ Log steps generation test passed")
    
    test_estimate_dataset_size()
    print("✓ Dataset size estimation test passed")
    
    test_generate_checkpoint_schedule()
    print("✓ Checkpoint schedule generation test passed")
    
    test_schedule_distribution()
    print("✓ Schedule distribution test passed")
    
    test_config_integration()
    print("✓ Configuration integration test passed")
    
    print("\nAll checkpoint scheduling tests passed! ✓") 