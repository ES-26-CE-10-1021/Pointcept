#!/bin/bash
#SBATCH --job-name=ptv3_late_injection
#SBATCH --output=logs/dino_injection_ptv3_late_%j.log
#SBATCH --error=logs/dino_injection_ptv3_late_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=96G

# Activate conda env
source "$(dirname "$CONDA_EXE")/activate" pointcept

# WanDB
# put key in ~/.secrets/wandb_api_key ; chmod 600 ~/.secrets/wandb_api_key
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY=agco-student-job

# Run script
srun sh scripts/train.sh -d dino_injection -c ptv3_late_fusion -n dino-injection-ptv3-late -g 1
