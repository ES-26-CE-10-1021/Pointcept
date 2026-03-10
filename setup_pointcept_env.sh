#!/usr/bin/env bash
set -eo pipefail

ENV_NAME="pointcept"
export TORCH_CUDA_ARCH_LIST="8.0"
# https://developer.nvidia.com/cuda-gpus

conda env create --file=environment.yml

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

pip install --no-build-isolation ./libs/pointops
pip install --no-build-isolation ./libs/pointgroup_ops
pip install --no-build-isolation flash-attn==2.7.4.post1

echo "==> Done! Activate with: conda activate ${ENV_NAME}"
