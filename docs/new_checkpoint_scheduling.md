# New Checkpoint Scheduling System

This document describes the updated checkpoint scheduling system that provides precise control over checkpoint frequency and spacing.

## Overview

The new checkpoint scheduling system allows you to:
1. **Specify exact number of checkpoints** for the first epoch
2. **Choose spacing type** for subsequent epochs (linear or logarithmic)
3. **Control spacing parameters** (log base or linear interval)
4. **Ensure epoch boundary checkpoints** at the end of each epoch
5. **Calculate steps accurately** based on actual dataset size and batch configuration

## Key Features

### First Epoch Control
- **Exact Checkpoint Count**: Specify exactly how many checkpoints you want in the first epoch
- **Even Distribution**: Checkpoints are distributed evenly across the first epoch
- **Flexible Range**: Supports any number of checkpoints (0 to any positive integer)

### Subsequent Epochs Control
- **Linear Spacing**: Fixed interval between checkpoints (e.g., every 100 steps)
- **Logarithmic Spacing**: Exponential spacing with configurable base (default: 2)
- **Auto-calculation**: Linear interval can be auto-calculated based on epoch length

### Accurate Step Calculation
- **Dataset-based**: Uses actual dataset size to calculate steps per epoch
- **Batch-aware**: Considers batch size and gradient accumulation
- **Fallback Support**: Graceful fallback to estimation when actual data unavailable

## Configuration

### Training Configuration

```yaml
training:
  # ... existing parameters ...
  
  # Checkpoint generation parameters
  auto_generate_checkpoints: false  # Enable automatic generation
  
  # First epoch configuration
  first_epoch_checkpoints: 20       # Number of checkpoints in first epoch
  
  # Subsequent epochs configuration
  subsequent_epochs_spacing: "log"  # "linear" or "log"
  log_base: 2                       # Base for logarithmic spacing
  linear_interval: null             # Steps between checkpoints for linear spacing (null = auto)
  min_checkpoint_interval: 100      # Minimum steps between checkpoints
```

### Parameter Details

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `first_epoch_checkpoints` | int | 20 | Number of checkpoints in the first epoch |
| `subsequent_epochs_spacing` | str | "log" | "linear" or "log" |
| `log_base` | int | 2 | Base for logarithmic spacing |
| `linear_interval` | int/null | null | Steps between checkpoints for linear spacing |
| `min_checkpoint_interval` | int | 100 | Minimum steps between checkpoints |

## Usage Examples

### CLI Commands

1. **Basic usage with defaults**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml
   ```

2. **Custom first epoch checkpoints**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
     --first-epoch 10
   ```

3. **Linear spacing for subsequent epochs**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
     --spacing linear --linear-interval 200
   ```

4. **Custom logarithmic base**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
     --spacing log --log-base 3
   ```

5. **Complete custom configuration**:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml \
     --first-epoch 15 \
     --spacing linear \
     --linear-interval 150 \
     --min-interval 50
   ```

### Direct Script Usage

```bash
python scripts/generate_checkpoint_schedule.py configs/experiment.yaml \
  --first-epoch 20 \
  --spacing log \
  --log-base 2
```

## Algorithm Details

### First Epoch Checkpoints
- Distributes checkpoints evenly across the first epoch
- Formula: `step = i * (steps_per_epoch / (num_checkpoints - 1))`
- Always includes step 0 and the last step of the epoch

### Linear Spacing
- Fixed interval between checkpoints: `step = start + i * interval`
- Auto-calculation: `interval = epoch_length / 10` (if not specified)
- Continues until reaching epoch boundary

### Logarithmic Spacing
- Exponential spacing: `step = start + base^i`
- Default base is 2: 2, 4, 8, 16, 32, 64, ...
- Continues until reaching epoch boundary

### Epoch Boundaries
- Always includes checkpoint at the end of each epoch
- Ensures consistent evaluation points across experiments

## Example Schedules

### Example 1: 20 First Epoch Checkpoints, Log Spacing
```
First Epoch (0-333): [0, 17, 35, 52, 70, 87, 105, 122, 140, 157, 175, 192, 210, 227, 245, 262, 280, 297, 315, 333]
Second Epoch (334-666): [334, 336, 340, 348, 364, 396, 460, 588, 666]
Third Epoch (667-1000): [667, 669, 673, 681, 697, 729, 793, 921, 1000]
```

### Example 2: 5 First Epoch Checkpoints, Linear Spacing
```
First Epoch (0-333): [0, 83, 166, 249, 333]
Second Epoch (334-666): [334, 384, 434, 484, 534, 584, 634, 666]
Third Epoch (667-1000): [667, 717, 767, 817, 867, 917, 967, 1000]
```

## Integration with Training

The checkpoint schedule is automatically used during training:

```python
# In trainer.py
checkpoint_schedule = self._get_checkpoint_schedule()

for step in training_steps:
    # ... training logic ...
    
    if step in checkpoint_schedule:
        self._save_checkpoint()
```

## Auto-Generation

When `auto_generate_checkpoints: true` is set, the trainer automatically generates a schedule:

```yaml
training:
  auto_generate_checkpoints: true
  first_epoch_checkpoints: 20
  subsequent_epochs_spacing: "log"
  # No checkpoint_schedule needed - will be generated automatically
```

## Benefits

1. **Precise Control**: Exact control over checkpoint frequency and spacing
2. **Flexible Configuration**: Support for both linear and logarithmic spacing
3. **Accurate Calculation**: Based on actual dataset size and batch configuration
4. **Epoch Consistency**: Always checkpoints at epoch boundaries
5. **Storage Efficiency**: Configurable minimum intervals prevent excessive checkpointing
6. **Reproducibility**: Deterministic schedules for consistent experiments

## Testing

Run the test suite to validate the checkpoint scheduling:

```bash
python tests/test_checkpoint_scheduling_simple.py
```

This tests:
- Configuration parsing and validation
- First epoch checkpoint generation
- Subsequent epoch checkpoint generation (linear and log)
- Integration with the configuration system

## Migration from Old System

The new system is backward compatible. Old configurations will continue to work, but you can upgrade to the new system by:

1. **Adding new parameters** to your config:
   ```yaml
   training:
     first_epoch_checkpoints: 20
     subsequent_epochs_spacing: "log"
     log_base: 2
   ```

2. **Removing old parameters** (optional):
   ```yaml
   # Remove these old parameters
   # target_checkpoints: {...}
   # log_steps_first_epoch: true
   ```

3. **Regenerating schedules** with the new system:
   ```bash
   python -m model_foundry.cli generate-checkpoints configs/experiment.yaml
   ``` 