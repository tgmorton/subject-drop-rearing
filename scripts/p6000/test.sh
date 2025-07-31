#!/bin/bash
#SBATCH --job-name=subject-drop-ablative # Descriptive job name
#SBATCH --gres=gpu:1 # P6000 GPU partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=10G
#SBATCH --time=0-1:00:00 # Time limit (D-HH:MM:SS)

module load matlab/R2022a
nvidia-smi
matlab -nodisplay -nosplash -r for_loop