#!/bin/bash
#SBATCH --job-name=3detr_pc_180
#SBATCH --output=logs/3detr_pointcept_180_%j.log
#SBATCH --error=logs/3detr_pointcept_180_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G

set -eo pipefail

# WandB
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY=agco-student-job

source "$(dirname "$CONDA_EXE")/activate" pointcept

cd /home/andreas/3D-Perception/Pointcept

srun sh scripts/train.sh \
    -d scannet \
    -c det-3detr-v0m1-0-scannet-180ep \
    -n 3detr_180ep_pointcept \
    -g 1
