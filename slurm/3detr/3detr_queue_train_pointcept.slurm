#!/bin/bash
#SBATCH --job-name=3detr_pointcept
#SBATCH --output=logs/3detr_train_pointcept_%j.log
#SBATCH --error=logs/3detr_train_pointcept_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G

# Activate conda env
source "$(dirname "$CONDA_EXE")/activate" pointcept

# WanDB
# put key in ~/.secrets/wandb_api_key ; chmod 600 ~/.secrets/wandb_api_key
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY=agco-student-job

# Run script
srun sh scripts/train.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr_debug -g 2
