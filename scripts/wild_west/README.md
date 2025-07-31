# Wild-West GPU Management System

A flexible, responsible GPU management system for servers without job schedulers (SLURM, etc.). This system provides GPU monitoring, resource coordination, and distributed training capabilities.

## Overview

This system consists of three main components:

1. **GPU Monitor** (`gpu_monitor.sh`) - Real-time GPU status and resource coordination
2. **Experiment Runner** (`run_experiment.sh`) - Flexible experiment execution with GPU selection
3. **Distributed Training** (`run_distributed.sh`) - Multi-GPU distributed training coordination

## Quick Start

### 1. Check GPU Status
```bash
# Show current GPU status
./scripts/wild_west/gpu_monitor.sh

# Find available GPUs
./scripts/wild_west/gpu_monitor.sh available

# Watch GPU status in real-time
./scripts/wild_west/gpu_monitor.sh watch
```

### 2. Run a Simple Experiment
```bash
# Run on GPUs 1 and 2 (default)
./scripts/wild_west/run_experiment.sh experiment_1_remove_expletives

# Run on specific GPUs
./scripts/wild_west/run_experiment.sh -g 1,2 experiment_2_baseline

# Run with GPU locking and availability check
./scripts/wild_west/run_experiment.sh -g 1,2 -l -c experiment_3_remove_articles
```

### 3. Run Distributed Training
```bash
# Run distributed training on GPUs 1 and 2
./scripts/wild_west/run_distributed.sh -g 1,2 experiment_1_remove_expletives

# Run with custom batch size and epochs
./scripts/wild_west/run_distributed.sh -g 1,2 -b 32 -e 10 experiment_2_baseline
```

## GPU Monitoring

### Basic Commands
```bash
# Show GPU status and processes
./scripts/wild_west/gpu_monitor.sh status

# Find available GPUs (>10GB free)
./scripts/wild_west/gpu_monitor.sh available

# Lock a GPU for your use
./scripts/wild_west/gpu_monitor.sh lock 1

# Unlock a GPU
./scripts/wild_west/gpu_monitor.sh unlock 1

# Show all GPU locks
./scripts/wild_west/gpu_monitor.sh locks

# Watch GPU status in real-time
./scripts/wild_west/gpu_monitor.sh watch
```

### GPU Status Categories
- **AVAILABLE** (>20GB free) - Ideal for large experiments
- **LIMITED** (10-20GB free) - Good for smaller experiments
- **OCCUPIED** (<10GB free) - Avoid for new experiments

## Experiment Runner

### Basic Usage
```bash
./scripts/wild_west/run_experiment.sh [OPTIONS] <config_name>
```

### Options
- `-g, --gpus <gpu_ids>` - Comma-separated GPU IDs (default: 1,2)
- `-p, --phase <phase>` - Phase to run (default: full-pipeline)
- `-b, --batch-size <size>` - Batch size override
- `-e, --epochs <num>` - Number of epochs override
- `-d, --distributed` - Enable distributed training
- `-l, --lock-gpus` - Lock GPUs before running
- `-u, --unlock-gpus` - Unlock GPUs after running
- `-c, --check-gpus` - Check GPU availability before running
- `-v, --verbose` - Verbose output

### Phases
- `preprocess` - Dataset preprocessing and ablation
- `train-tokenizer` - SentencePiece tokenizer training
- `tokenize-dataset` - Dataset tokenization
- `run` - Model training
- `full-pipeline` - Complete pipeline (default)

### Examples
```bash
# Simple run on default GPUs
./scripts/wild_west/run_experiment.sh experiment_1_remove_expletives

# Run training only on GPU 1
./scripts/wild_west/run_experiment.sh -g 1 -p run experiment_2_baseline

# Run with custom parameters and GPU locking
./scripts/wild_west/run_experiment.sh -g 1,2 -b 64 -e 20 -l -c experiment_3_remove_articles

# Run distributed training
./scripts/wild_west/run_experiment.sh -g 1,2 -d experiment_4_lemmatize_verbs
```

## Distributed Training

### Basic Usage
```bash
./scripts/wild_west/run_distributed.sh [OPTIONS] <config_name>
```

### Options
- `-g, --gpus <gpu_ids>` - Comma-separated GPU IDs (default: 1,2)
- `-n, --nodes <num>` - Number of nodes (default: 1)
- `-p, --phase <phase>` - Phase to run (default: run)
- `-b, --batch-size <size>` - Batch size per GPU
- `-e, --epochs <num>` - Number of epochs
- `-m, --master-port <port>` - Master port (default: 12355)
- `-a, --master-addr <addr>` - Master address (default: localhost)
- `-l, --lock-gpus` - Lock GPUs before running
- `-c, --check-gpus` - Check GPU availability before running
- `-v, --verbose` - Verbose output

### Examples
```bash
# Distributed training on GPUs 1 and 2
./scripts/wild_west/run_distributed.sh -g 1,2 experiment_1_remove_expletives

# Distributed training on all 4 GPUs
./scripts/wild_west/run_distributed.sh -g 0,1,2,3 experiment_2_baseline

# Distributed training with custom parameters
./scripts/wild_west/run_distributed.sh -g 1,2 -b 32 -e 10 experiment_3_remove_articles
```

## Responsible GPU Usage

### Best Practices
1. **Check GPU availability** before starting experiments
2. **Lock GPUs** when running long experiments
3. **Use appropriate GPUs** based on memory requirements
4. **Monitor usage** during experiments
5. **Unlock GPUs** when done

### GPU Selection Guidelines
- **Large experiments** (>20GB): Use GPUs with >20GB available
- **Medium experiments** (10-20GB): Use GPUs with >15GB available
- **Small experiments** (<10GB): Use any GPU with >10GB available

### Memory Requirements
- **Preprocessing**: Usually <5GB
- **Tokenizer training**: Usually <2GB
- **Dataset tokenization**: Usually <5GB
- **Model training**: 15-40GB depending on model size and batch size

## System Tools

### Additional Monitoring Commands
```bash
# Quick GPU status
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# Monitor GPU processes
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# Check GPU temperature and power
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv,noheader,nounits
```

### Environment Variables
The scripts automatically set these environment variables:
- `CUDA_VISIBLE_DEVICES` - Controls which GPUs are visible
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - Better memory management
- `MASTER_ADDR`, `MASTER_PORT` - For distributed training
- `WORLD_SIZE`, `RANK` - For distributed training
- `NCCL_DEBUG=INFO` - For distributed training debugging

## Troubleshooting

### Common Issues

1. **GPU not found**
   ```bash
   # Check available GPUs
   nvidia-smi -L
   ```

2. **Insufficient memory**
   ```bash
   # Check memory usage
   ./scripts/wild_west/gpu_monitor.sh status
   ```

3. **GPU already locked**
   ```bash
   # Check locks
   ./scripts/wild_west/gpu_monitor.sh locks
   
   # Unlock if needed
   ./scripts/wild_west/gpu_monitor.sh unlock <gpu_id>
   ```

4. **Distributed training issues**
   ```bash
   # Check NCCL debug info
   export NCCL_DEBUG=INFO
   ```

### Log Files
- Experiment logs: `logs/` directory
- Distributed training logs: `logs/distributed_<config>_rank<rank>.log`
- GPU monitor logs: `/tmp/gpu_monitor_<user>.log`

## Configuration

### Default Settings
- Default GPUs: 1,2
- Default phase: full-pipeline
- Lock directory: `/tmp/gpu_locks`
- Master port: 12355
- Master address: localhost

### Customization
You can modify the default values in the script files:
- `DEFAULT_GPUS` in `run_experiment.sh`
- `DEFAULT_PHASE` in `run_experiment.sh`
- `MASTER_PORT` in `run_distributed.sh`

## Integration with Existing Workflow

### From SLURM Scripts
Replace SLURM-specific commands:
```bash
# Old SLURM way
sbatch scripts/p6000/run_ablation_experiment.sh experiment_1_remove_expletives preprocess

# New wild-west way
./scripts/wild_west/run_experiment.sh -g 1,2 -p preprocess experiment_1_remove_expletives
```

### Batch Processing
Create a batch script for multiple experiments:
```bash
#!/bin/bash
# batch_experiments.sh

experiments=(
    "experiment_1_remove_expletives"
    "experiment_2_baseline"
    "experiment_3_remove_articles"
)

for exp in "${experiments[@]}"; do
    echo "Running $exp..."
    ./scripts/wild_west/run_experiment.sh -g 1,2 -l -c "$exp"
    echo "Completed $exp"
done
```

This system provides a robust, flexible way to manage GPU resources in a wild-west server environment while being respectful of other users' needs. 