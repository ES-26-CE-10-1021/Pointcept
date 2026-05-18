#!/usr/bin/env bash
# scripts/post_install.sh
# Run after `pixi install` to build the parts that don't fit into pixi.toml:
#   - PyG C extensions (torch-cluster, torch-scatter, torch-sparse) from the
#     PyG find-links wheel index — pixi/uv don't accept -f URLs.
#   - In-tree source builds with --no-build-isolation so they see the env's torch.
#   - 3DETR's cython utils.
#
# Run via: pixi run post-install
# This script assumes pixi has activated the env (CONDA_PREFIX, CUDA_HOME, etc.
# are set by scripts/activate.sh).

set -eo pipefail

PYTORCH_VERSION="2.7.0"
CUDA_VARIANT="cu128"
PYG_WHL_URL="https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+${CUDA_VARIANT}.html"

echo "==> Verifying activated env"
echo "    CONDA_PREFIX:          ${CONDA_PREFIX:-<unset>}"
echo "    CUDA_HOME:             ${CUDA_HOME:-<unset>}"
echo "    TORCH_CUDA_ARCH_LIST:  ${TORCH_CUDA_ARCH_LIST:-<unset>}"
echo "    nvcc:                  $(nvcc --version | tail -n1)"
echo "    gcc:                   $(gcc --version | head -n1)"
echo "    python:                $(python --version)"

# Sanity: torch must be importable. If this fails, `pixi install` didn't complete.
python -c "import torch; print(f'    torch:                 {torch.__version__}')"

echo "==> Initializing git submodules"
git submodule update --init --recursive

# ---------------------------------------------------------------------------
# PyG C extensions: not on regular PyPI, served from a find-links index.
# ---------------------------------------------------------------------------
echo "==> Installing PyG prebuilt wheels (torch ${PYTORCH_VERSION} + ${CUDA_VARIANT})"
pip install torch-cluster torch-scatter torch-sparse -f "${PYG_WHL_URL}"

# ---------------------------------------------------------------------------
# Git-hosted source builds. --no-build-isolation is mandatory: these setup.py
# files import torch at build time.
# ---------------------------------------------------------------------------
echo "==> Installing source dependencies (ocnn, CLIP)"
pip install --no-build-isolation git+https://github.com/octree-nn/ocnn-pytorch.git
pip install --no-build-isolation git+https://github.com/openai/CLIP.git

# ---------------------------------------------------------------------------
# In-tree CUDA extensions. Each setup.py reads TORCH_CUDA_ARCH_LIST to decide
# which sm_xx targets to compile for; activate.sh has set it to 10.0+PTX.
# ---------------------------------------------------------------------------
echo "==> Building pointops"
pip install --no-build-isolation ./libs/pointops

echo "==> Building pointgroup_ops"
pip install --no-build-isolation ./libs/pointgroup_ops

echo "==> Building 3DETR pointnet2"
pip install --no-build-isolation ./third_party/3detr/third_party/pointnet2

echo "==> Building 3DETR cython utils"
( cd third_party/3detr/utils && python cython_compile.py build_ext --inplace )

echo "==> Building PointRoPE (LitePT)"
pip install --no-build-isolation ./third_party/LitePT/libs/pointrope

# ---------------------------------------------------------------------------
# Optional: FlashAttention-4 (Blackwell-native, BETA, different API from FA2).
# PTv3 / Pointcept / 3DETR import from `flash_attn` (FA2 API); FA4 exposes
# `flash_attn.cute`. Installing alone won't help — needs code patching. The
# safer default is PyTorch's torch.nn.functional.scaled_dot_product_attention,
# which has Blackwell SDPA kernels in PyTorch >= 2.7. Uncomment once ready:
#
pip install flash-attn-4==4.0.0b13
# ---------------------------------------------------------------------------

echo ""
echo "==> post-install complete. Verify with: pixi run smoke-test"
