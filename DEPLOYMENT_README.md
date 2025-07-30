# Server Deployment Guide

## What's Ready for GPU Server

### ✅ Core Infrastructure
- **CLI Interface**: `model_foundry/cli.py` with all commands working
- **Configuration System**: Pydantic-based validation and experiment configs
- **SLURM Scripts**: `scripts/run_ablation_experiment.sh` and `scripts/submit_experiment.sh`
- **Fixed Preprocessing**: `remove_expletives.py` with all critical issues resolved

### ✅ Experiment Configurations
- `experiment_0_baseline.yaml`: Control experiment (no ablations)
- `experiment_1_remove_expletives.yaml`: Expletive removal experiment

### ✅ Container Support
- `singularity/ablation.sif`: For preprocessing and tokenization
- `singularity/training.sif`: For model training
- Both containers configured for GPU usage

## Server Setup Steps

### 1. Transfer Repository
```bash
# On your local machine
git push origin main

# On the server
git clone <repository-url>
cd subject-drop-rearing
```

### 2. Update Paths in SLURM Script
Edit `scripts/run_ablation_experiment.sh` and update:
```bash
HOST_PROJECT_DIR="/path/to/your/project/on/server"
HOST_ABLATION_SIF_PATH="/path/to/ablation.sif"
HOST_TRAINING_SIF_PATH="/path/to/training.sif"
```

### 3. Test the Setup
```bash
# Test CLI locally
python -m model_foundry.cli info

# Test config validation
python -m model_foundry.cli validate-config configs/experiment_1_remove_expletives.yaml

# Test preprocessing (small sample)
python preprocessing/remove_expletives.py \
  --input_dir data/raw/train_90M/ \
  --output_dir data/processed/test/ \
  --replacement_pool_dir data/raw/train_90M/ \
  --chunk_size 1000
```

### 4. Submit Jobs
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Submit preprocessing job
./scripts/submit_experiment.sh experiment_1_remove_expletives preprocess

# Submit full pipeline
./scripts/submit_experiment.sh experiment_1_remove_expletives full-pipeline
```

## Expected Performance on GPU Server

### Preprocessing (Expletive Removal)
- **Local (M1 Mac)**: ~290 lines/second (very slow)
- **GPU Server**: Should be 10-50x faster with proper GPU utilization
- **Expected time**: 2-6 hours for full 90M dataset

### Model Training
- **Baseline**: ~1M steps, ~7 days on A5000
- **Ablative experiments**: Similar timeframes
- **Checkpointing**: Every 1000 steps during first epoch

## Monitoring

### Job Status
```bash
squeue -u $USER
```

### Logs
```bash
# View preprocessing logs
tail -f logs/subject-drop-experiment_1_remove_expletives-preprocess-*.out

# View training logs  
tail -f logs/subject-drop-experiment_1_remove_expletives-run-*.out
```

### Weights & Biases
- Project: `just-drop-the-subject`
- Experiments will be logged automatically
- Monitor loss curves and learning rates

## Troubleshooting

### Common Issues
1. **Container not found**: Ensure `.sif` files are in correct paths
2. **GPU memory**: Reduce batch size if OOM errors
3. **Time limits**: Increase SLURM time limits for large datasets
4. **Dependencies**: Ensure all Python packages are installed in container

### Debugging
```bash
# Test container functionality
singularity exec --nv --bind .:/workspace singularity/ablation.sif python -m model_foundry.cli info

# Test preprocessing in container
singularity exec --nv --bind .:/workspace singularity/ablation.sif python preprocessing/remove_expletives.py --help
```

## Next Steps After Deployment

1. **Run baseline experiment** to establish control
2. **Run expletive removal experiment** to test ablation
3. **Implement evaluation framework** for measuring effects
4. **Add more ablation scripts** (determiner impoverishment, etc.)
5. **Scale up** to larger datasets and more experiments

## File Structure for Server
```
subject-drop-rearing/
├── configs/                    # Experiment configurations
├── data/
│   ├── raw/                   # Original datasets
│   ├── processed/             # Ablated datasets
│   └── tokenized/             # Tokenized datasets
├── model_foundry/             # Core framework
├── preprocessing/              # Ablation scripts
├── scripts/                   # SLURM scripts
├── singularity/               # Container definitions
├── tokenizers/               # Trained tokenizers
└── models/                   # Trained model checkpoints
``` 