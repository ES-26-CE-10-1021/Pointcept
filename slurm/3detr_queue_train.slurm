#!/bin/bash
#SBATCH --job-name=3detr_train
#SBATCH --output=logs/3detr_train_%j.log
#SBATCH --error=logs/3detr_train_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=7-00:00:00
#SBATCH --mem=64G

# Activate conda env
source ~/miniforge3/bin/activate pointcept

# Run script
export PYTHONUNBUFFERED=1
srun bash ~/3D-Perception/Pointcept/third_party/3detr/scripts/scannet_quick_modified.sh