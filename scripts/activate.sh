#!/usr/bin/env bash
# scripts/activate.sh
# Sourced by pixi when entering the env via `pixi shell` or `pixi run`.
# Pixi sets CONDA_PREFIX before sourcing this; everything else derives from it.
 
# Pin downstream builds to the env's CUDA toolchain, not whatever's on the system.
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/gcc"
export CXX="${CONDA_PREFIX}/bin/g++"
export CPATH="${CONDA_PREFIX}/targets/x86_64-linux/include:${CPATH:-}"
 
# Source builds (pointops etc.) compile against this arch list. Append +PTX so
# the resulting cubins also carry forward-compatible PTX for newer GPUs.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"
 
# UCloud's base image ships /etc/pip/constraint.txt which forces older versions
# of common ML packages and conflicts with cu128 wheels. Disable for this session.
unset PIP_CONSTRAINT
