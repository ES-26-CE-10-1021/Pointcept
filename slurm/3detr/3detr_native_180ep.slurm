#!/bin/bash
#SBATCH --job-name=3detr_native_180
#SBATCH --output=logs/3detr_native_180_%j.log
#SBATCH --error=logs/3detr_native_180_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G

set -eo pipefail

DATA_ROOT="/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data"
META_ROOT="/home/andreas/3D-Perception/votenet/scannet/meta_data"
NATIVE_ROOT="/home/andreas/3D-Perception/Pointcept/third_party/3detr"

source "$(dirname "$CONDA_EXE")/activate" pointcept

cd "${NATIVE_ROOT}"

python main.py \
    --dataset_name scannet \
    --dataset_root_dir "${DATA_ROOT}" \
    --meta_data_dir "${META_ROOT}" \
    --max_epoch 180 \
    --batchsize_per_gpu 8 \
    --ngpus 1 \
    --checkpoint_dir "${NATIVE_ROOT}/outputs/scannet_180ep_comparison" \
    --eval_every_epoch 10 \
    --seed 0
