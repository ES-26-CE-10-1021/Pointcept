# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Pointcept is a point cloud perception research framework. It implements multiple 3D deep learning architectures (Point Transformer V1/V2/V3, SpUNet, SPVCNN, OctFormer, etc.) for tasks like semantic segmentation, instance segmentation, classification, and self-supervised pre-training (Concerto, Sonata, MSC).

## Common Commands

### Training
```bash
# Via shell script (recommended, handles experiment dirs and code snapshots)
sh scripts/train.sh -d <dataset> -c <config_name> -n <exp_name> -g <num_gpus>
# Example: sh scripts/train.sh -d scannet -c semseg-pt-v3m1-0-base -n my_exp -g 4

# Direct Python (simpler)
python tools/train.py --config-file configs/<dataset>/<config>.py --num-gpus <N> --options save_path=exp/<dataset>/<name>

# Resume training
sh scripts/train.sh -d <dataset> -c <config_name> -n <exp_name> -r true
```

### Testing
```bash
sh scripts/test.sh -d <dataset> -c <config_name> -n <exp_name> -w model_best -g <num_gpus>
# Or directly:
python tools/test.py --config-file <config>.py --num-gpus <N> --options save_path=<exp_dir> weight=<path>.pth
```

### Building Custom CUDA Ops
```bash
cd libs/pointops && python setup.py install
cd libs/pointgroup_ops && python setup.py install
# Similarly for pointops2, pointseg
```

### Code Formatting
```bash
black .
```
Black formatting is enforced via GitHub Actions CI on pushes to `main` and PRs.

## Architecture

### Registry Pattern
All major components use a Registry system (`pointcept/utils/registry.py`) for dynamic instantiation from config dicts. Key registries:

| Registry | Location | Purpose |
|----------|----------|---------|
| `MODELS` | `pointcept/models/builder.py` | Model architectures |
| `MODULES` | `pointcept/models/builder.py` | Reusable model modules |
| `DATASETS` | `pointcept/datasets/builder.py` | Dataset classes |
| `TRANSFORMS` | `pointcept/datasets/transform.py` | Data augmentations |
| `LOSSES` | `pointcept/models/losses/builder.py` | Loss functions |
| `HOOKS` | `pointcept/engines/hooks/builder.py` | Training hooks |
| `TRAINERS` | `pointcept/engines/train.py` | Trainer classes |
| `TESTERS` | `pointcept/engines/test.py` | Tester classes |
| `OPTIMIZERS` | `pointcept/utils/optimizer.py` | Optimizers |
| `SCHEDULERS` | `pointcept/utils/scheduler.py` | LR schedulers |

Register new components with `@REGISTRY.register_module()` decorator. Configs reference components by `type` string, e.g. `dict(type="DefaultSegmentor", backbone=dict(type="PT-v3m1", ...))`.

### Config System
Configs are Python files in `configs/` using inheritance via `_base_` lists. Base configs live in `configs/_base_/`. Config values can be overridden from CLI with `--options key=value`. The config system is in `pointcept/utils/config.py`.

### Engine (Training/Testing)
- **Entry points**: `tools/train.py`, `tools/test.py`
- **Training**: `pointcept/engines/train.py` — `Trainer` class with hook-based lifecycle (before_train, before_epoch, before_step, after_step, after_epoch, after_train)
- **Testing**: `pointcept/engines/test.py` — `SemSegTester`, `ClsTester`, `InsSegTester`
- **Launch**: `pointcept/engines/launch.py` — DDP multi-GPU via `torch.distributed`
- **Hooks**: `pointcept/engines/hooks/` — checkpoint saving/loading, evaluation, logging, timing
- Supports AMP, gradient accumulation, WandB/TensorBoard logging

### Dataset Format
Point clouds stored as `.npy` files in per-scene directories. Standard assets: `coord.npy`, `color.npy`, `normal.npy`, `segment.npy` (labels), `instance.npy`. The `DefaultDataset` auto-loads any available asset files.

### Custom CUDA Extensions
Located in `libs/`: `pointops` (core point operations), `pointops2`, `pointgroup_ops`, `pointseg`. Built with `torch.utils.cpp_extension`.

### Third-Party Submodules
- `third_party/3detr` — 3D Detection Transformer fork (branch `dev`)

## Key Conventions

- Models separate backbone from task head via Segmentor/Classifier wrappers
- The `Point` data structure (`pointcept/models/utils.py`) is the unified representation passed through models, carrying coords, features, offsets, and serialization info
- Experiment outputs go to `exp/<dataset>/<exp_name>/` with `model/` (checkpoints) and `code/` (snapshot) subdirs
- `scripts/train.sh` copies the full codebase into the experiment's `code/` dir for reproducibility
- Branch workflow: `dev` for development, `main` for releases/CI
