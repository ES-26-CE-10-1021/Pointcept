# Thesis Experiments

This note tracks the experiments selected to produce results for the thesis.
Configs live in `configs/<dataset>/...` and are launched via
`scripts/train.sh -d <dataset> -c <config> -n <exp_name> -g <num_gpus>`.

## ScanNet (baseline benchmark)

ScanNet runs serve as a **relatable baseline** on a public dataset so the thesis
has comparable numbers outside of AGCO. They are intentionally minimal: three
runs that vary **only the feature extractor**, with everything else (token
budget, query count, encoder/decoder shape, schedule, criterion) held fixed.
Any AP delta between them is therefore attributable to the backbone choice.

### Shared settings (held constant across all three ScanNet runs)

| Knob | Value |
|---|---|
| Pre-encoder output budget | 2048 tokens (FPS) |
| Encoder | `VanillaTransformerEncoder3DETR`, 3 layers |
| Decoder | `TransformerDecoder3DETR`, 8 layers, dim=256, nhead=4, ffn=256 |
| Queries | 256 (FPS-sampled from encoder output) |
| Box parameterization | Axis-aligned (ScanNet convention), `num_angle_bin=1` |
| Optimizer | AdamW, lr=5e-4, weight_decay=0.1 |
| Scheduler | OneCycleLR, cosine annealing |
| Epochs | 720 |
| Batch size | 8 (total) |
| AMP | disabled |
| Criterion | Native 3DETR defaults (matcher: `cost_class=1, cost_giou=2, cost_objectness=0, cost_center=0`; loss: `giou=1, no_object=0.25, center=5, size=1, sem_cls=1`) |
| Dataset | `ScanNetDetectionDataset` with `augment=True`, `random_cuboid_min_points=30000` |

### Runs

| # | Slot | Config | Backbone stack |
|---|---|---|---|
| 1 | **Base 3DETR** | `configs/scannet/det-3detr-v0m1-0-scannet.py` | PointNet++ SA (2048 pts) → Vanilla encoder (3L) — faithful native 3DETR replication |
| 2 | **PTv3 from scratch** | `configs/scannet/det-3detr-v3m1-1-scannet.py` | PTv3 (enc→256) + FPS 2048 → Vanilla encoder (3L) — swaps PointNet++ for PTv3, same token budget |
| 3 | **Utonia fine-tuning** | `configs/scannet/det-3detr-utonia-v5m1-1-scannet.py` | Pretrained Utonia (PT-v3m3), `freeze_backbone="enc_finetune"` (last enc stage trainable) + FPS 2048 → Vanilla encoder (3L) — partial fine-tune of the pretrained VFM |

The triple isolates the **backbone-effect axis**:
PointNet++ → PTv3-from-scratch → Utonia-pretrained-with-partial-finetune.

### Launch commands

```bash
# 1. Base 3DETR
sh scripts/train.sh -d scannet -c det-3detr-v0m1-0-scannet      -n thesis_scannet_base_3detr_720ep      -g 2

# 2. PTv3 from scratch
sh scripts/train.sh -d scannet -c det-3detr-v3m1-1-scannet      -n thesis_scannet_ptv3_fps_720ep        -g 2

# 3. Utonia fine-tuning
sh scripts/train.sh -d scannet -c det-3detr-utonia-v5m1-1-scannet -n thesis_scannet_utonia_encft_fps_720ep -g 2
```

### Notes on validity

- ScanNet 3DETR configs still use `ScanNetDetectionDataset`'s legacy inline
  augmentation (`augment=True` + random-cuboid/flip/rotate); the transform-list
  refactor was AGCO-only. The legacy path is intact and reproduces prior
  numbers — see `notes/3detr_config_overview.md` for the full config table.
- Reporting metric: AP25 / AP50 from `ObjDetEvaluator` (uses 3DETR's
  `APCalculator.step_meter`), values reported as percentages (0–100).

## AGCO (primary contribution)

AGCO runs are the thesis's main contribution. The plan mirrors the ScanNet
triple — base / PTv3 / Utonia — at the same FPS-2048 token budget, so any
delta between the two datasets is attributable to the data, not the
architecture. The SUN-like matcher/loss (m3 family) is used throughout: it
outperformed the default 3DETR loss in our pilot runs, and AGCO's oriented
boxes are closer to SUN-RGBD than to ScanNet's axis-aligned setting.

Each backbone is run on both AGCO sensors (`ouster` 40k pts, `lslidar` 100k
pts), giving 6 headline runs.

### Shared settings (held constant across all six AGCO runs)

| Knob | Value |
|---|---|
| Classes | 3 — `("tractor", "harvester", "trailer")` |
| Box parameterization | Oriented, `num_angle_bin=12` (SUN-RGBD-style yaw cls + residual) |
| `max_num_obj` | 16 |
| Pre-encoder output budget | 2048 tokens (PointNet++ SA built-in for v1m3-0; FPS post-PTv3/Utonia for v3m4-0 and v2m4-0) |
| Encoder | `VanillaTransformerEncoder3DETR`, 3 layers |
| Decoder | `TransformerDecoder3DETR`, 8 layers, dim=256, nhead=4, ffn=256 |
| Queries | 32 (FPS-sampled from encoder output) |
| Center prediction | `center_offset_normalized=True` |
| Optimizer | AdamW, lr=5e-4, weight_decay=0.1 |
| Scheduler | OneCycleLR, `pct_start=0.10`, `div_factor=500`, `final_div_factor=1.0` |
| Epochs | 720 |
| Criterion | SUN-like: matcher `(class=1, objectness=5, giou=3, center=5)`; loss `(giou=0, sem_cls=1, no_object=0.1, angle_cls=0.1, angle_reg=0.5, center=5, size=1)`; `giou_on_aux_outputs=False` |
| Detection crops | `FovCropDetection` ±60° + `SphericalCropDetection` per-sensor (60m lslidar / 40m ouster) |
| Train augmentation | y-flip (p=0.5), RandomRotateZ ±5°, `PointSubsampleDetection` |
| Dataset | `AgcoBBoxV1` with `apply_t_rtk`, `apply_r_global`, `apply_r_level_to_{points,boxes}`, `require_gravity_align`, `fixed_pc_dims` per sensor, `min_inliers=dict(lslidar=350, ouster=200, rslidar=80)` |

### Sensor differences

| Sensor | `num_points` | Spherical max_dist | `fixed_pc_dims` (x/y/z) |
|---|---|---|---|
| ouster | 40,000 | 40 m | min [0, -36, -10] / max [40, 36, 8] |
| lslidar | 100,000 | 60 m | min [0, -55, -11] / max [60, 55, 10] |

### Runs

| # | Backbone slot | Sensor | Config |
|---|---|---|---|
| 1 | Base 3DETR (PointNet++ SA + Vanilla) | ouster | `configs/agco/det-3detr-v1m3-0-3cls-agco-ouster.py` |
| 2 | Base 3DETR (PointNet++ SA + Vanilla) | lslidar | `configs/agco/det-3detr-v1m3-0-3cls-agco.py` |
| 3 | PTv3 + FPS 2048 + Vanilla | ouster | `configs/agco/det-3detr-v3m4-0-3cls-agco-ouster.py` |
| 4 | PTv3 + FPS 2048 + Vanilla | lslidar | `configs/agco/det-3detr-v3m4-0-3cls-agco.py` |
| 5 | Utonia `enc_finetune` + FPS 2048 + Vanilla | ouster | `configs/agco/det-3detr-v2m4-0-3cls-agco-ouster.py` |
| 6 | Utonia `enc_finetune` + FPS 2048 + Vanilla | lslidar | `configs/agco/det-3detr-v2m4-0-3cls-agco.py` |

### Launch commands

```bash
# Base 3DETR
sh scripts/train.sh -d agco -c det-3detr-v1m3-0-3cls-agco-ouster -n 3detr_agco_v1m3_sunloss_3cls_ouster_720ep         -g 1
sh scripts/train.sh -d agco -c det-3detr-v1m3-0-3cls-agco        -n 3detr_agco_v1m3_sunloss_3cls_lslidar_720ep        -g 1

# PTv3 from scratch + FPS
sh scripts/train.sh -d agco -c det-3detr-v3m4-0-3cls-agco-ouster -n 3detr_agco_v3m4_ptv3_fps_sunloss_3cls_ouster_720ep   -g 1
sh scripts/train.sh -d agco -c det-3detr-v3m4-0-3cls-agco        -n 3detr_agco_v3m4_ptv3_fps_sunloss_3cls_lslidar_720ep  -g 1

# Utonia enc_finetune + FPS
sh scripts/train.sh -d agco -c det-3detr-v2m4-0-3cls-agco-ouster -n 3detr_agco_v2m4_utonia_fps_sunloss_3cls_ouster_720ep  -g 1
sh scripts/train.sh -d agco -c det-3detr-v2m4-0-3cls-agco        -n 3detr_agco_v2m4_utonia_fps_sunloss_3cls_lslidar_720ep -g 1
```

SLURM scripts for all six runs are under
`third_party/slurm_scripts/agco-cluster/pointcept/3detr/agco_bbox/{ouster,lslidar}/`
(`v1m3_3cls.slurm`, `v3m4_3cls.slurm`, `v2m4_3cls.slurm`).

### Cross-dataset architecture mapping

| Backbone slot | ScanNet config | AGCO config |
|---|---|---|
| Base 3DETR | `det-3detr-v0m1-0-scannet.py` | `det-3detr-v1m3-0-3cls-agco{,-ouster}.py` |
| PTv3 from scratch | `det-3detr-v3m1-1-scannet.py` | `det-3detr-v3m4-0-3cls-agco{,-ouster}.py` |
| Utonia `enc_finetune` | `det-3detr-utonia-v5m1-1-scannet.py` | `det-3detr-v2m4-0-3cls-agco{,-ouster}.py` |

Within each row the pre-encoder type, FPS budget (2048), encoder
(VanillaTransformer 3L), decoder (8L, 256d), and schedule (720ep) are
identical; only dataset-specific knobs differ (oriented vs axis-aligned boxes,
`grid_size=0.05` vs 0.02, `num_queries=32` vs 256, AGCO sensor crops, SUN-like
loss for AGCO).

### Hyperparameter deviations from ScanNet (and why)

- `grid_size=0.05` (vs ScanNet's 0.02): outdoor LiDAR is much sparser.
- `num_queries=32` (vs 256): AGCO scenes contain far fewer objects per frame.
- `num_angle_bin=12` (vs 1): boxes are oriented around z, not axis-aligned.
- `fixed_pc_dims` per sensor: stabilizes center/size normalization across
  scenes with very different spatial scales.
- FOV ±60° + spherical crops: matches the effective sensor frustum and bounds
  per-sensor effective range.
- SUN-like matcher/loss: empirically stronger than default 3DETR loss on AGCO,
  defensible given oriented-box similarity to SUN-RGBD.
- `batch_size=4`, `gradient_accumulation_steps=2`: heavier backbones (PTv3,
  Utonia) on outdoor point counts (40k–100k) require more memory than the
  ScanNet defaults; v1m3-0 keeps `batch_size=8`.

## Color ablation (added 2026-05-15)

A second ScanNet/AGCO triple was added to ablate **per-point RGB** as an
input modality, holding everything else identical to the headline triples.
The ScanNet headline Utonia run (`det-3detr-utonia-v5m1-1-scannet.py`)
silently uses `use_color=True` because the pretrained PT-v3m3 checkpoint
expects 9 input channels; to make the ScanNet row strictly comparable, a
new `-nocolor` variant of that config was created as the fair baseline.

**Utonia 9-channel guarantee.** Across all configurations the PT-v3m3
embedding sees exactly 9 channels. When color is off, the missing RGB
slot is zero-filled (model-side via
`PTv3m3PreEncoder._build_padded_feat`, dataset-side on AGCO via the
existing `utonia_preprocess` zero-pad).

### Runs

| # | Backbone slot | Dataset / sensor | Config |
|---|---|---|---|
| C1 | Base 3DETR + color | ScanNet | `configs/scannet/det-3detr-v0m1-0-scannet-color.py` |
| C2 | PTv3 + FPS + color | ScanNet | `configs/scannet/det-3detr-v3m1-1-scannet-color.py` |
| C3 | Utonia `enc_finetune` + FPS, **no color** | ScanNet | `configs/scannet/det-3detr-utonia-v5m1-1-scannet-nocolor.py` |
| C4 | Base 3DETR + color | AGCO / ouster | `configs/agco/det-3detr-v1m3-0-3cls-agco-color-ouster.py` |
| C5 | PTv3 + FPS + color | AGCO / ouster | `configs/agco/det-3detr-v3m4-0-3cls-agco-color-ouster.py` |
| C6 | Utonia `enc_finetune` + FPS + color | AGCO / ouster | `configs/agco/det-3detr-v2m4-0-3cls-agco-color-ouster.py` |

### Diffs vs the headline triples

| Backbone | Color-config delta |
|---|---|
| Base 3DETR (PointNet++) | `use_color=True` in train/val/test; `PointnetSAPreEncoder.mlp_dims[0]: 0 → 3` |
| PTv3 from scratch | `use_color=True`; `PTv3PreEncoder.in_channels: 3 → 6` |
| Utonia (PT-v3m3) | `use_color=True/False` only — `_build_padded_feat` (or AGCO `utonia_preprocess`) keeps the 9-channel layout |

### Dataset support

- **ScanNet**: `ScanNetDetectionDataset` already had `use_color=True`
  plumbed (`pointcept/datasets/scannet_detection.py:209`); no change.
- **AGCO**: `AgcoBBoxV1` gained a `use_color: bool = False` kwarg. When
  enabled, `_load_scan()` reads
  `<root>/<sensor>/color/<timestamp>.npy` (float32, `(N, 3)`, values in
  `[0, 1]`) and concatenates RGB after XYZ — canonical layout
  `[xyz, rgb, intensity]`.

### Launch commands

```bash
# ScanNet color triple
sh scripts/train.sh -d scannet -c det-3detr-v0m1-0-scannet-color           -n thesis_scannet_base_3detr_color_720ep         -g 2
sh scripts/train.sh -d scannet -c det-3detr-v3m1-1-scannet-color           -n thesis_scannet_ptv3_fps_color_720ep           -g 2
sh scripts/train.sh -d scannet -c det-3detr-utonia-v5m1-1-scannet-nocolor  -n thesis_scannet_utonia_encft_fps_nocolor_720ep -g 4

# AGCO color triple (ouster)
sh scripts/train.sh -d agco -c det-3detr-v1m3-0-3cls-agco-color-ouster -n 3detr_agco_v1m3_color_sunloss_3cls_ouster_720ep         -g 1
sh scripts/train.sh -d agco -c det-3detr-v3m4-0-3cls-agco-color-ouster -n 3detr_agco_v3m4_ptv3_fps_color_sunloss_3cls_ouster_720ep -g 1
sh scripts/train.sh -d agco -c det-3detr-v2m4-0-3cls-agco-color-ouster -n 3detr_agco_v2m4_utonia_fps_color_sunloss_3cls_ouster_720ep -g 1
```

SLURM scripts (both clusters, matching file names under each tree):
- `third_party/slurm_scripts/{ai-lab,agco-cluster}/pointcept/3detr/scannet/v0m1-color.slurm`
- `third_party/slurm_scripts/{ai-lab,agco-cluster}/pointcept/3detr/scannet/v3m1-1-color.slurm`
- `third_party/slurm_scripts/{ai-lab,agco-cluster}/pointcept/3detr/scannet/utonia-v5m1-1-nocolor.slurm`
- `third_party/slurm_scripts/{ai-lab,agco-cluster}/pointcept/3detr/agco_bbox/ouster/v1m3_3cls_color.slurm`
- `third_party/slurm_scripts/{ai-lab,agco-cluster}/pointcept/3detr/agco_bbox/ouster/v3m4_3cls_color.slurm`
- `third_party/slurm_scripts/{ai-lab,agco-cluster}/pointcept/3detr/agco_bbox/ouster/v2m4_3cls_color.slurm`

### Ablations preserved

The following configs are not part of the thesis-headline triple, but are kept
in the repo for follow-up ablations if reviewers ask "did FPS help?" or
"does PTv3 need a Vanilla encoder on top?":

- `det-3detr-v3m3-0-3cls-agco-ouster.py` — PTv3 + Identity encoder, no FPS.
- `det-3detr-v2m3-0-3cls-agco-ouster.py` — Utonia + Vanilla encoder, no FPS.
- Overfit `-1` variants (`v1m3-1`, `v2m3-1`, `v3m3-1`) for sanity checks.

See `notes/3detr_config_overview.md` § AGCO for the full inventory.
