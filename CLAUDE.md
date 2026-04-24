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

# Re-run precise evaluation on a finished experiment without retraining:
sh scripts/test.sh -d scannet -c det-3detr-v1m1-0-scannet -n <exp_name> -w model_best -g 1
```

### Unit Tests

```bash
conda run -n pointcept python -m pytest tests/ -v -s
```

### SLURM

```bash
sbatch slurm/<job_script>.slurm
tail -n 40 logs/<job_name>_<job_id>.log   # stdout
tail -n 40 logs/<job_name>_<job_id>.err   # stderr (model output + errors)
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

| Registry     | Location                             | Purpose                |
| ------------ | ------------------------------------ | ---------------------- |
| `MODELS`     | `pointcept/models/builder.py`        | Model architectures    |
| `MODULES`    | `pointcept/models/builder.py`        | Reusable model modules |
| `DATASETS`   | `pointcept/datasets/builder.py`      | Dataset classes        |
| `TRANSFORMS` | `pointcept/datasets/transform.py`    | Data augmentations     |
| `LOSSES`     | `pointcept/models/losses/builder.py` | Loss functions         |
| `HOOKS`      | `pointcept/engines/hooks/builder.py` | Training hooks         |
| `TRAINERS`   | `pointcept/engines/train.py`         | Trainer classes        |
| `TESTERS`    | `pointcept/engines/test.py`          | Tester classes         |
| `OPTIMIZERS` | `pointcept/utils/optimizer.py`       | Optimizers             |
| `SCHEDULERS` | `pointcept/utils/scheduler.py`       | LR schedulers          |

Register new components with `@REGISTRY.register_module()` decorator. Configs reference components by `type` string, e.g. `dict(type="DefaultSegmentor", backbone=dict(type="PT-v3m1", ...))`.

### Config System

Configs are Python files in `configs/` using inheritance via `_base_` lists. Base configs live in `configs/_base_/`. Config values can be overridden from CLI with `--options key=value`. The config system is in `pointcept/utils/config.py`.

**`batch_size` and `num_worker` are totals across all GPUs.** Pointcept divides them by `world_size` at runtime (`defaults.py`) to get per-GPU values. E.g. `batch_size=16` with 4 GPUs → 4 per GPU.

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

### 3DETR Integration (`pointcept/models/detection_3detr/`)

3DETR integrated as modular Pointcept components with PTv3 encoder support. Key differences from segmentation models:

- **Input format**: internally converts between dense `(B, N, C)` tensors and the `Point` dataclass (concat+offset) at encoder boundaries
- **Dataset**: `ScanNetDetectionDataset` reads VoteNet-style data (`*_vert.npy`, `*_bbox.npy`) — separate from Pointcept's segmentation `.npy` format
- **Registered components**: `Model3DETRDetector` (MODELS), `PTv3PreEncoder`, `PTv3UNetPreEncoder`, `PTv3m3PreEncoder`, `PointnetSAPreEncoder`, `IdentityEncoder3DETR`, `VanillaTransformerEncoder3DETR`, `MaskedTransformerEncoder3DETR`, `TransformerDecoder3DETR`, `ScanNetDetectionConfig` (MODULES), `SetCriterion3DETR` (LOSSES), `ObjDetEvaluator` (HOOKS), `ObjDetTester` (TESTERS)
- **PTv3 encoder**: `PTv3PreEncoder` wraps PointTransformerV3 as a pre-encoder. Variable-length output from voxelization is handled by `point2dense()` which pads to max scene length and returns a `padding_mask` threaded through the decoder's cross-attention. FPS query sampling masks padded positions by pushing them to `1e6`.
- **Utonia pre-encoder**: `PTv3m3PreEncoder` wraps `PT-v3m3` (Utonia) as a pretrained backbone. Loads the HuggingFace checkpoint via `third_party.utonia.utonia.model.load(ckpt_only=True)` in a DDP-safe rank-0 + barrier pattern, zero-pads missing modalities (rgb / normal) on-device into the 9-ch input, and detaches + `requires_grad_(True)`'s the leaf at each freeze/trainable boundary so downstream layers can attach a fresh graph. Supports encoder-only (`enc_mode=True`, default) and full U-Net (`enc_mode=False`), plus optional FPS downsampling (`npoint=K`) that returns a fixed-length `(xyz, features, inds)` tuple matching `PointnetSAPreEncoder`. Freeze behaviour is a single enum `freeze_backbone`: `"enc"` (default — embedding + encoder frozen, decoder if present is trainable), `"enc_finetune"` (everything frozen except the **last encoder stage**, which trains end-to-end), or `"none"` (full gradient flow). **The distributed Utonia checkpoint is encoder-only** (`dec_*=None`); any `enc_mode=False` config builds and trains a **fresh, randomly-initialized** decoder from scratch — the config must supply `dec_*` kwargs explicitly, and `_load_pretrained_state` will log the expected dec-missing keys at INFO. Backbone hyperparameters (channels, depths, etc.) come from the checkpoint config; the pointcept config specifies `pretrained`, `grid_size`, `enc_mode`, `npoint`, `freeze_backbone`, and `dec_*` (when `enc_mode=False`).
- **`pre_encoder=None`**: skips PointNet++ SA; set `input_feature_dim` to match input channels beyond XYZ
- **Configs**: `det-3detr-v0m1-0-scannet.py` (native 3DETR baseline), `det-3detr-v1m1-0-scannet.py` (PointNet++ SA encoder), `det-3detr-v2m1-0-scannet.py` (PTv3 + Identity encoder), `det-3detr-v3m1-0-scannet.py` (PTv3 + Vanilla Transformer encoder), `det-3detr-v3m1-1-scannet.py` (PTv3 + FPS + Vanilla Transformer encoder), `det-3detr-v4m1-0-scannet.py` (PTv3 U-Net + Identity encoder), `det-3detr-utonia-v1m1-0-scannet.py` (Frozen Utonia PT-v3m3 + Vanilla Transformer encoder), `det-3detr-utonia-v2m1-0-scannet.py` (… + Utonia-canonical preprocessing), `det-3detr-utonia-v3m1-0-scannet.py` (… + RGB input), `det-3detr-utonia-v4m1-0-scannet.py` (Frozen Utonia encoder-only + FPS 2048), `det-3detr-utonia-v5m1-0-scannet.py` (Utonia `enc_finetune` — last encoder stage trainable), `det-3detr-utonia-v5m1-1-scannet.py` (… + FPS 2048), `det-3detr-utonia-v5m2-0-scannet.py` (Frozen Utonia encoder + fresh trainable PTv3 decoder), `det-3detr-utonia-v5m2-1-scannet.py` (… + FPS 2048)
- **Evaluation**: AP25/AP50 reported as percentages (0–100); uses `APCalculator.step_meter()` from `third_party/3detr/utils/ap_calculator.py`

## Key Conventions

- Models separate backbone from task head via Segmentor/Classifier wrappers
- The `Point` data structure (`pointcept/models/utils.py`) is the unified representation passed through models, carrying coords, features, offsets, and serialization info
- Experiment outputs go to `exp/<dataset>/<exp_name>/` with `model/` (checkpoints) and `code/` (snapshot) subdirs
- `scripts/train.sh` copies `scripts/`, `tools/`, `pointcept/`, and `third_party/` (via rsync, excluding `outputs/`, `runs/`, `logs/`) into the experiment's `code/` dir for reproducibility. The `PYTHONPATH` is set to `code/` so all imports resolve from the snapshot.
- Branch workflow: `dev` for development, `main` for releases/CI
