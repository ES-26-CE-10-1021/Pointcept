#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="pointcept"
unset PIP_CONSTRAINT # Disregard /etc/pip/constraint.txt

if command -v nvidia-smi &> /dev/null && nvidia-smi -i 0 &> /dev/null; then
    export TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)"
    # https://developer.nvidia.com/cuda-gpus
else
    echo "Warning: No GPU detected. Defaulting TORCH_CUDA_ARCH_LIST to 8.0"
    export TORCH_CUDA_ARCH_LIST="8.0"
fi

conda env create --file=environment.yml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Build system to conda's CUDA, not the system CUDA
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

pip install --no-build-isolation ./libs/pointops
pip install --no-build-isolation ./libs/pointgroup_ops
MAX_JOBS=8 pip install --no-build-isolation --no-cache-dir flash-attn==2.7.4.post1

# 3DETR dependencies
echo "==> Building 3DETR dependencies"
pip install --no-build-isolation ./third_party/3detr/third_party/pointnet2
pip install trimesh opencv-python cython
(cd third_party/3detr/utils && python cython_compile.py build_ext --inplace)

# PointRoPE
echo "==> Building PointRoPE..."
# Using pip install . is cleaner than setup.py install
pip install --no-build-isolation ./third_party/LitePT/libs/pointrope

# Testing
pip install pytest

echo "==> Done! Activate with: conda activate ${ENV_NAME}"

