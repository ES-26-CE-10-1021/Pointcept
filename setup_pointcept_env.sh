#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="pointcept"

echo "==> Initializing Git submodules..."
git submodule update --init --recursive

unset PIP_CONSTRAINT # Disregard /etc/pip/constraint.txt

# Define cuda compute capability
echo "==> Getting GPU Cuda compute capability..."
if command -v nvidia-smi &> /dev/null && nvidia-smi -i 0 &> /dev/null; then
    export TORCH_CUDA_ARCH_LIST="$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader)"
    # https://developer.nvidia.com/cuda-gpus
else
    echo "Warning: No GPU detected. Defaulting TORCH_CUDA_ARCH_LIST to 8.0"
    export TORCH_CUDA_ARCH_LIST="8.0"
fi

# Create conda environment
echo "==> Creating conda env..."
conda env create --file=environment.yml
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# Build system to conda's CUDA, not the system CUDA
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"

# Pip installs
echo "==> Installing additional pip dependencies..."
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

# Pre-compiled wheels
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch_geometric spconv-cu124 peft

# Source builds requiring PyTorch (MUST use --no-build-isolation)
pip install --no-build-isolation git+https://github.com/octree-nn/ocnn-pytorch.git
pip install --no-build-isolation git+https://github.com/openai/CLIP.git
MAX_JOBS=8 pip install --no-build-isolation --no-cache-dir flash-attn==2.7.4.post1
pip install --no-build-isolation ./libs/pointops
pip install --no-build-isolation ./libs/pointgroup_ops

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
