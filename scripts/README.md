# SLURM Scripts for Subject-Drop Rearing Experiments

This directory contains SLURM scripts for running ablative experiments on the cluster using Singularity containers.

## Files

- `run_ablation_experiment.sh`: Main SLURM script for running experiments
- `submit_experiment.sh`: Helper script for easy job submission
- `README.md`: This documentation

## Usage

### Quick Start

```bash
# Submit a preprocessing job
./scripts/submit_experiment.sh experiment_1_remove_expletives preprocess

# Submit a full pipeline job
./scripts/submit_experiment.sh experiment_1_remove_expletives full-pipeline

# Submit a training-only job
./scripts/submit_experiment.sh experiment_0_baseline run
```

### Available Phases

1. **preprocess**: Run dataset manipulation pipeline (uses ablation.sif)
2. **train-tokenizer**: Train SentencePiece tokenizer (uses ablation.sif)
3. **tokenize-dataset**: Tokenize training data (uses ablation.sif)
4. **run**: Execute model training (uses training.sif)
5. **full-pipeline**: Run all phases in sequence

### Job Submission Examples

```bash
# Run preprocessing for expletive removal experiment
./scripts/submit_experiment.sh experiment_1_remove_expletives preprocess

# Run full pipeline for baseline experiment
./scripts/submit_experiment.sh experiment_0_baseline full-pipeline

# Run training only with custom job name
./scripts/submit_experiment.sh experiment_0_baseline run baseline_training
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job logs
tail -f logs/subject-drop-experiment_1_remove_expletives-preprocess-<job_id>.out

# Cancel a job
scancel <job_id>
```

## Configuration

### Required Paths

Update these paths in `run_ablation_experiment.sh` to match your cluster environment:

```bash
HOST_PROJECT_DIR="/home/AD/thmorton/subject-drop-rearing"
HOST_ABLATION_SIF_PATH="/home/AD/thmorton/subject-drop-rearing/singularity/ablation.sif"
HOST_TRAINING_SIF_PATH="/home/AD/thmorton/subject-drop-rearing/singularity/training.sif"
```

### SLURM Parameters

The script uses these SLURM parameters:
- Partition: `general_gpu_a5000`
- Memory: 64GB
- CPUs: 12 per task
- Time limit: 7 days
- GPU: 1 node with 1 task per node

Adjust these in `run_ablation_experiment.sh` if needed.

## Container Requirements

### Ablation Container (ablation.sif)
- Used for: preprocessing, tokenizer training, dataset tokenization
- Contains: spaCy, transformers, sentencepiece, and other preprocessing dependencies
- Includes: en_core_web_trf spaCy model for linguistic analysis

### Training Container (training.sif)
- Used for: model training
- Contains: PyTorch with CUDA support, transformers, and training dependencies
- Optimized for: GPU training with CUDA 11.8

## Experiment Workflow

1. **Create Experiment Config**: Add a new `.yaml` file to `configs/`
2. **Submit Preprocessing**: Run dataset manipulation pipeline
3. **Submit Tokenizer Training**: Train experiment-specific tokenizer
4. **Submit Dataset Tokenization**: Tokenize training data
5. **Submit Model Training**: Train the language model
6. **Evaluate Results**: Use evaluation scripts on trained models

## Troubleshooting

### Common Issues

1. **Container not found**: Ensure singularity images are in the correct path
2. **Config file not found**: Check that config files exist in `configs/`
3. **Permission denied**: Make scripts executable with `chmod +x scripts/*.sh`
4. **Out of memory**: Reduce batch size or increase memory allocation
5. **Time limit exceeded**: Increase time limit or split into smaller jobs

### Debugging

```bash
# Test script locally (without SLURM)
bash scripts/run_ablation_experiment.sh experiment_0_baseline validate-config

# Check container functionality
singularity exec --bind .:/workspace singularity/ablation.sif python -m model_foundry.cli info

# Validate config file
python -m model_foundry.cli validate-config configs/experiment_0_baseline.yaml
```

## Example Experiments

### Baseline Experiment
- **Config**: `experiment_0_baseline.yaml`
- **Goal**: Control experiment with no ablations
- **Command**: `./scripts/submit_experiment.sh experiment_0_baseline full-pipeline`

### Expletive Removal Experiment
- **Config**: `experiment_1_remove_expletives.yaml`
- **Goal**: Test effect of expletive presence on subject-drop acquisition
- **Command**: `./scripts/submit_experiment.sh experiment_1_remove_expletives full-pipeline`

## Log Files

Logs are saved to `logs/` directory with naming pattern:
- `subject-drop-<config>-<phase>-<job_id>.out` (stdout)
- `subject-drop-<config>-<phase>-<job_id>.err` (stderr)

Example: `logs/subject-drop-experiment_1_remove_expletives-preprocess-12345.out` 