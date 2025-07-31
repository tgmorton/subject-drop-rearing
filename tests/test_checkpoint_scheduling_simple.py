"""
Simple test script for the checkpoint scheduling system.
"""

import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from generate_checkpoint_schedule import (
    CheckpointGenerationConfig,
    generate_first_epoch_checkpoints,
    generate_subsequent_epoch_checkpoints
)


def test_checkpoint_generation_config():
    """Test the new CheckpointGenerationConfig class."""
    config = CheckpointGenerationConfig(
        first_epoch_checkpoints=10,
        subsequent_epochs_spacing="log",
        log_base=2,
        linear_interval=None,
        min_interval=100
    )
    
    assert config.first_epoch_checkpoints == 10
    assert config.subsequent_epochs_spacing == "log"
    assert config.log_base == 2
    assert config.linear_interval is None
    assert config.min_interval == 100
    
    # Test validation
    try:
        CheckpointGenerationConfig(subsequent_epochs_spacing="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    print("✓ CheckpointGenerationConfig test passed")


def test_generate_first_epoch_checkpoints():
    """Test first epoch checkpoint generation."""
    # Test with 5 checkpoints in 100 steps
    checkpoints = generate_first_epoch_checkpoints(100, 5)
    
    assert len(checkpoints) == 5
    assert checkpoints[0] == 0  # First checkpoint
    assert checkpoints[-1] == 100  # Last checkpoint
    
    # Check that checkpoints are evenly distributed
    intervals = [checkpoints[i+1] - checkpoints[i] for i in range(len(checkpoints)-1)]
    assert all(interval > 0 for interval in intervals)
    
    # Test edge cases
    assert generate_first_epoch_checkpoints(100, 0) == []
    assert generate_first_epoch_checkpoints(100, 1) == [100]
    
    print("✓ First epoch checkpoint generation test passed")


def test_generate_subsequent_epoch_checkpoints():
    """Test subsequent epoch checkpoint generation."""
    epoch_start = 100
    epoch_end = 200
    
    # Test linear spacing
    linear_checkpoints = generate_subsequent_epoch_checkpoints(
        epoch_start, epoch_end, "linear", linear_interval=20
    )
    
    assert len(linear_checkpoints) > 0
    for checkpoint in linear_checkpoints:
        assert epoch_start < checkpoint < epoch_end
    
    # Test log spacing
    log_checkpoints = generate_subsequent_epoch_checkpoints(
        epoch_start, epoch_end, "log", log_base=2
    )
    
    assert len(log_checkpoints) > 0
    for checkpoint in log_checkpoints:
        assert epoch_start < checkpoint < epoch_end
    
    # Verify log spacing follows the pattern
    if len(log_checkpoints) > 1:
        for i in range(len(log_checkpoints) - 1):
            ratio = log_checkpoints[i+1] / log_checkpoints[i]
            assert ratio >= 2  # Should be at least base 2
    
    print("✓ Subsequent epoch checkpoint generation test passed")


if __name__ == "__main__":
    print("Running simple checkpoint scheduling tests...")
    
    test_checkpoint_generation_config()
    test_generate_first_epoch_checkpoints()
    test_generate_subsequent_epoch_checkpoints()
    
    print("\nAll simple checkpoint scheduling tests passed! ✓") 