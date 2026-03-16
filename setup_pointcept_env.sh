#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="pointcept"
export TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)"
# https://developer.nvidia.com/cuda-gpus

conda env create --file=environment.yml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Point build system to conda's CUDA, not the system CUDA
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

pip install --no-build-isolation ./libs/pointops
pip install --no-build-isolation ./libs/pointgroup_ops
pip install --no-build-isolation --no-cache-dir flash-attn==2.7.4.post1

# 3DETR dependencies
pip install --no-build-isolation ./third_party/3detr/third_party/pointnet2
pip install trimesh opencv-python cython
(cd third_party/3detr/utils && python cython_compile.py build_ext --inplace)

echo "==> Done! Activate with: conda activate ${ENV_NAME}"
