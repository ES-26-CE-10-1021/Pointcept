#!/bin/bash
#SBATCH --job-name=ptv3_deep_no_dino_test
#SBATCH --output=logs/dino_injection_ptv3_deep_no_dino_test_%j.log
#SBATCH --error=logs/dino_injection_ptv3_deep_no_dino_test_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=96G

# Diagnostic: evaluate trained deep fusion checkpoint with DINO injection disabled.
# Measures how much the backbone learned from geometry alone vs relying on DINO shortcuts.

# Activate conda env
source "$(dirname "$CONDA_EXE")/activate" pointcept

# WanDB (disabled in config, but set key just in case)
export WANDB_API_KEY="$(cat ~/.secrets/wandb_api_key)"
export WANDB_ENTITY=agco-student-job

# Test using the no-dino config but loading weights from the deep fusion experiment
srun sh scripts/test.sh \
    -d dino_injection \
    -c ptv3_deep_fusion_no_dino_test \
    -n dino-injection-ptv3-deep \
    -w model_best \
    -g 1
