#!/bin/bash
#SBATCH --job-name=real-only-ouster
#SBATCH --output=logs/real-only-ouster_%j.log
#SBATCH --error=logs/real-only-ouster_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:2
#SBATCH --time=7-00:00:00
#SBATCH --mem=128G

# Activate conda
source ~/anaconda3/bin/activate pointcept

# WanDB
# put your key in ~/.secrets/wandb_api_key ; chmod 600 ~/.secrets/wandb_api_key
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"

export WANDB_ENTITY=agco-student-job
# Run training

sh scripts/train.sh -d agco -c ptv3_intensity_predictor -n int-reg-ditr-projection -g 2
