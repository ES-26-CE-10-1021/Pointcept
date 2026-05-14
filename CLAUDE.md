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

- `third_party/3detr` — original 3DETR (3D Detection Transformer) code, fork on branch `dev`. Used directly via `sys.path` patching by `pointcept/models/detection_3detr/`.
- `third_party/slurm_scripts` — SLURM job scripts for running training/testing on the cluster.

### 3DETR Integration (`pointcept/models/detection_3detr/`)

3DETR integrated as modular Pointcept components with PTv3 encoder support. Key differences from segmentation models:

- **Input format**: internally converts between dense `(B, N, C)` tensors and the `Point` dataclass (concat+offset) at encoder boundaries
- **Dataset**: `ScanNetDetectionDataset` reads VoteNet-style data (`*_vert.npy`, `*_bbox.npy`) — separate from Pointcept's segmentation `.npy` format. Augmentation is **config-driven via `transform=[...]`** (the same registry used by `AgcoBBoxV1`); the legacy `augment=True` / `random_cuboid_min_points` kwargs were removed. `use_color`, `use_height`, and `utonia_preprocess` remain as dataset kwargs (deterministic feature engineering / canonicalisation that runs before the transform pipeline). For PTv3-using configs (v2/v3/v4 + all utonia variants), `GridSampleDetection` must appear in every split's transform list with `grid_size` matching the pre-encoder. Multi-task semseg scaffold mirrors AGCO: `load_segment=False` kwarg loads `<scene>_sem_label.npy`, applies the existing NYU-40 → semseg-class mapping, and exposes the result under the `"segment"` key (auto-indexed by every point-subsampling transform via `_PER_POINT_KEYS`).
- **Registered components**: `Model3DETRDetector` (MODELS), `PTv3PreEncoder`, `PTv3UNetPreEncoder`, `PTv3m3PreEncoder`, `PointnetSAPreEncoder`, `IdentityEncoder3DETR`, `VanillaTransformerEncoder3DETR`, `MaskedTransformerEncoder3DETR`, `TransformerDecoder3DETR`, `ScanNetDetectionConfig`, `AgcoBBoxConfig` (MODULES), `SetCriterion3DETR` (LOSSES), `ObjDetEvaluator` (HOOKS), `ObjDetTester` (TESTERS)
- **AGCO dataset**: `AgcoBBoxV1` (DATASETS) consumes per-timestamp `(sensor, timestamp)` samples with raw-world annotations + `gravity_align.npz`. Pairs with `AgcoBBoxConfig` (`num_angle_bin=12`, SUN-RGBD-style yaw encoding). The class set is configurable via the `included_classes` kwarg on **both** `AgcoBBoxV1` and `AgcoBBoxConfig` — pass the same ordered tuple to both (default: all 5 — `tractor, harvester, trailer, car, hopper` → indices 0..4). Boxes for excluded classes are dropped at dataset load time so the model never sees them as targets and any prediction firing on them is penalised as background. `sensors=[...]` kwarg selects which LiDARs contribute (no cross-sensor fusion). Bbox loader filters parents by `is_visible` and `inliers >= min_inliers` (default 67) and expands their `children` (e.g. trailer hoppers) only when the parent passes. `R_level` is applied via explicit flags `apply_r_level_to_points` / `apply_r_level_to_boxes` (both default False; configs set them True). `T_rtk` (sensor → RTK) is wired behind `apply_t_rtk` / `require_calibration` flags using each root's `calibration.yml`. **Augmentation is config-driven** via a `transform=[...]` list of detection-aware transforms registered in `pointcept/datasets/det_transform.py` (`RandomFlipDetection`, `RandomRotateZDetection`, `RandomScaleDetection`, `RandomJitterDetection`, `RandomCuboidDetection`, `SphericalCropDetection`, `FovCropDetection`, `GridSampleDetection`, `PointSubsampleDetection`); these mutate the raw `(point_cloud, gt_box_centers_raw, gt_box_sizes_raw, gt_box_angles_raw, gt_box_labels_raw)` dict in step. The legacy `augment` and `random_cuboid_min_points` kwargs were removed. The dataset still enforces `num_points` as a final safety net. **`GridSampleDetection`** voxel-deduplicates points (one random per voxel, matching the seg `GridSample` train mode) and is required upstream of any PTv3 pre-encoder so the input contract — one feature per voxel — is honoured. Place it after geometric augmentations and before `PointSubsampleDetection`; pick `grid_size` to match the model's `PTv3PreEncoder.grid_size`. **Multi-task semseg scaffold**: `AgcoBBoxV1` accepts `load_segment=False` (default) / `segment_subdir="segment"` kwargs; when `load_segment=True`, it loads `<root>/<sensor>/<segment_subdir>/<ts>.npy` (int64, length must match `point_cloud`) and adds `"segment"` to the transform dict and the returned sample. Every point-subsampling/cropping transform indexes any key in the module-level `_PER_POINT_KEYS` registry in `det_transform.py` (currently `point_cloud`, `segment`, `pcl_color`) in lockstep, so optional per-point arrays survive the pipeline once the upcoming bbox+semseg multi-task head lands.
- **Multi-task model**: `MultiTask3DETRSegmentor` (`pointcept/models/detection_3detr/multi_task.py`) holds a full PT-v3m1 U-Net backbone and bolts both a semseg head (Linear over the unpooled decoder output, `CrossEntropy + Lovasz` losses) and a 3DETR detection branch (transformer encoder/decoder + `SetCriterion3DETR`) onto it. The detection branch taps the encoder bottleneck (after `self.backbone.enc`, before `self.backbone.dec` mutates the parent chain — the snapshot via `point2dense` happens inside `_backbone_forward`). The semseg branch is unchanged from a vanilla `DefaultSegmentorV2` setup. **Loss combination** uses **uncertainty weighting** by default (literal Cipolla form with `c_i=2`, see thesis Eq. uncertainty_weighting): `L_total = (1/(c_seg σ_seg²)) L_seg + log σ_seg + (1/(c_det σ_det²)) L_det + log σ_det`. The σ scalars are stored as learnable `log_sigma_sq_seg` / `log_sigma_sq_det` `nn.Parameter`s (init zero → σ=1). Both multi-task configs include `param_dicts=[dict(keyword="log_sigma_sq", weight_decay=0.0)]` so the σ params aren't decayed. Switch to `loss_weighting="fixed"` for ablation against the prior fixed-weight baseline. The model returns `dict(loss, loss_seg, loss_det, sigma_seg, sigma_det)` in train mode and `dict(seg_logits, outputs, aux_outputs)` in eval. Pairs with `CombinedSegDetEvaluator` (hook, also logs `val/sigma_*` once per eval) and `CombinedSegDetTester` (tester) which run AP25/AP50 and per-class mIoU/Acc in one pass over the val/test loader. Configs: `configs/scannet/multitask-3detr-ptv3-v0m1-0-scannet.py`, `configs/agco/multitask-3detr-ptv3-v0m1-0-agco.py`. AGCO segmentation labels (model space): 0=background, 1=tractor, 2=harvester, 3=trailer — matches detection classes. On-disk labels also include 4=car, remapped to background via `seg_label_map={4: 0}` at load time; hopper is detection-only and never appears in segment files. On-disk at `<root>/<sensor>/segment/<ts>.npy`. Standalone PTv3 semseg still uses `DefaultSegmentorV2` + `configs/scannet/semseg-pt-v3m1-0-base.py`; standalone 3DETR detection still uses `Model3DETRDetector`. The multi-task class is purpose-built for the joint-loss path.
- **PTv3 encoder**: `PTv3PreEncoder` wraps PointTransformerV3 as a pre-encoder. Variable-length output from voxelization is handled by `point2dense()` which pads to max scene length and returns a `padding_mask` threaded through the decoder's cross-attention. FPS query sampling masks padded positions by pushing them to `1e6`.
- **Utonia pre-encoder**: `PTv3m3PreEncoder` wraps `PT-v3m3` (Utonia) as a pretrained backbone. Loads the HuggingFace checkpoint via `third_party.utonia.utonia.model.load(ckpt_only=True)` in a DDP-safe rank-0 + barrier pattern, zero-pads missing modalities (rgb / normal) on-device into the 9-ch input, and detaches + `requires_grad_(True)`'s the leaf at each freeze/trainable boundary so downstream layers can attach a fresh graph. Supports encoder-only (`enc_mode=True`, default) and full U-Net (`enc_mode=False`), plus optional FPS downsampling (`npoint=K`) that returns a fixed-length `(xyz, features, inds)` tuple matching `PointnetSAPreEncoder`. Freeze behaviour is a single enum `freeze_backbone`: `"enc"` (default — embedding + encoder frozen, decoder if present is trainable), `"enc_finetune"` (everything frozen except the **last encoder stage**, which trains end-to-end), or `"none"` (full gradient flow). **The distributed Utonia checkpoint is encoder-only** (`dec_*=None`); any `enc_mode=False` config builds and trains a **fresh, randomly-initialized** decoder from scratch — the config must supply `dec_*` kwargs explicitly, and `_load_pretrained_state` will log the expected dec-missing keys at INFO. Backbone hyperparameters (channels, depths, etc.) come from the checkpoint config; the pointcept config specifies `pretrained`, `grid_size`, `enc_mode`, `npoint`, `freeze_backbone`, and `dec_*` (when `enc_mode=False`).
- **`pre_encoder=None`**: skips PointNet++ SA; set `input_feature_dim` to match input channels beyond XYZ
- **Configs**: `det-3detr-v0m1-0-scannet.py` (native 3DETR baseline), `det-3detr-v1m1-0-scannet.py` (PointNet++ SA encoder), `det-3detr-v2m1-0-scannet.py` (PTv3 + Identity encoder), `det-3detr-v3m1-0-scannet.py` (PTv3 + Vanilla Transformer encoder), `det-3detr-v3m1-1-scannet.py` (PTv3 + FPS + Vanilla Transformer encoder), `det-3detr-v4m1-0-scannet.py` (PTv3 U-Net + Identity encoder), `configs/agco/det-3detr-v0m1-0-agco.py` (AGCO baseline using `AgcoBBoxV1` + `AgcoBBoxConfig`, oriented boxes), `det-3detr-utonia-v1m1-0-scannet.py` (Frozen Utonia PT-v3m3 + Vanilla Transformer encoder), `det-3detr-utonia-v2m1-0-scannet.py` (… + Utonia-canonical preprocessing), `det-3detr-utonia-v3m1-0-scannet.py` (… + RGB input), `det-3detr-utonia-v4m1-0-scannet.py` (Frozen Utonia encoder-only + FPS 2048), `det-3detr-utonia-v5m1-0-scannet.py` (Utonia `enc_finetune` — last encoder stage trainable), `det-3detr-utonia-v5m1-1-scannet.py` (… + FPS 2048), `det-3detr-utonia-v5m2-0-scannet.py` (Frozen Utonia encoder + fresh trainable PTv3 decoder), `det-3detr-utonia-v5m2-1-scannet.py` (… + FPS 2048)
- **Evaluation**: AP25/AP50 reported as percentages (0–100); uses `APCalculator.step_meter()` from `third_party/3detr/utils/ap_calculator.py`

## Key Conventions

- Models separate backbone from task head via Segmentor/Classifier wrappers
- The `Point` data structure (`pointcept/models/utils.py`) is the unified representation passed through models, carrying coords, features, offsets, and serialization info
- Experiment outputs go to `exp/<dataset>/<exp_name>/` with `model/` (checkpoints) and `code/` (snapshot) subdirs
- `scripts/train.sh` copies `scripts/`, `tools/`, `pointcept/`, and `third_party/` (via rsync, excluding `outputs/`, `runs/`, `logs/`) into the experiment's `code/` dir for reproducibility. The `PYTHONPATH` is set to `code/` so all imports resolve from the snapshot.
- Branch workflow: `dev` for development, `main` for releases/CI
