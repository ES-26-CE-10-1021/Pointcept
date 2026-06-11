# Thesis Experiment Context

> **Purpose.** This note pins the **final, frozen experiment matrix** for the
> thesis (as of 2026-05-19). For each run it gives the config path, the SLURM
> launcher, the architectural delta vs sibling configs, and a results
> placeholder. It is the single source of truth for the thesis design /
> implementation chapters and doubles as a run-tracker.
>
> **Companion docs (read-only, more detail).**
> - [`3detr_config_overview.md`](3detr_config_overview.md) — full per-config
>   field tables for every 3DETR config in the repo (more than just the
>   thesis subset).
> - [`multitask_ptv3_3detr.md`](multitask_ptv3_3detr.md) — design + code-path
>   walkthrough for the joint semseg + detection model used in RQ3.
> - `CLAUDE.md` — code-level integration notes (registry types, dataset
>   plumbing, `MultiTask3DETRSegmentor`, `PTv3m3PreEncoder` freeze enum).
>
> **Scope.** All AGCO runs are **ouster, 3-class** (`tractor, harvester,
> trailer`) — lslidar is kept out of the thesis scope. All ScanNet runs are
> standard 18-class detection on the VoteNet-preprocessed splits. Every run
> uses `epoch=720`, AdamW (lr=5e-4, wd=0.1), OneCycleLR cosine.

## Status legend

| Symbol | Meaning |
|---|---|
| ✅ | Run finished, metrics ready to fill in |
| ⏳ | Currently running |
| 👀 | Queued (config + SLURM ready, not yet submitted / waiting in queue) |
| ❌ | Not queued / out of scope / config missing |

## Backbone slots (vocabulary)

Five backbone slots appear across RQ1/RQ2/RQ3. They differ **only** in the
pre-encoder + freeze policy + the encoder block sitting between pre-encoder
and the 3DETR decoder. Decoder is always `TransformerDecoder3DETR`, 8 layers,
`d_model=256`, `nhead=4`, `ffn_dim=256`.

| Slot | Pre-encoder | Encoder | enc_dim | Freeze | FPS | What trains |
|---|---|---|---|---|---|---|
| **Base 3DETR** | PointNet++ SA (→2048 pts) | Vanilla (3L) | 256 | — | (built-in 2048) | everything |
| **PTv3 from scratch** | `PTv3PreEncoder` + FPS 2048 | Vanilla (3L) | 256 (ScanNet) / 256 (AGCO v3m4) | — | 2048 | everything |
| **Utonia ft** | `PTv3m3PreEncoder` + FPS 2048 | Vanilla (3L) | 576 | `enc_finetune` (last enc stage only) | 2048 | last enc stage + head |
| **Utonia frozen** | `PTv3m3PreEncoder` + FPS 2048 | Vanilla (3L) | 576 | `enc` (full encoder frozen) | 2048 | encoder block + 3DETR head |
| **Utonia frozen, minimal head** | `PTv3m3PreEncoder` + FPS 2048 | **Identity** | 576 | `enc` | 2048 | 3DETR decoder + MLP heads only |

"Minimal head" ≠ linear probing — the 8-layer 3DETR decoder is still there. It
is the **minimum architecture that admits DETR-style set prediction on top of
frozen pretrained features** (no extra transformer encoder between the frozen
backbone and the decoder). Cf. `IdentityEncoder3DETR` in
`pointcept/models/detection_3detr/encoder.py`.

For the per-modality input contract: PointNet++ ingests `[xyz (+ rgb?)]`,
PTv3-from-scratch ingests `[xyz (+ rgb?)]`, Utonia **always** sees 9 channels
`[xyz, rgb, normal]` — missing modalities are zero-filled (`_build_padded_feat`
in the model, or AGCO's `utonia_preprocess=True` dataset-side).

---

## RQ1 — Backbone choice & pretraining strategy

> **Held fixed across all 10 RQ1 runs (within a dataset):** token budget
> (2048 via built-in FPS or post-encoder FPS), encoder block shape (Vanilla
> 3L, or Identity for the minimal-head slot), 8-layer decoder, query count
> (256 ScanNet / 32 AGCO), schedule (720 ep), criterion (3DETR-default for
> ScanNet, SUN-like for AGCO), augmentation pipeline. The **only** axis that
> varies between cells in a column is the backbone slot.

### RQ1 run matrix

| Slot | Dataset | Config | SLURM | Exp name | Status | AP25 | AP50 | Notes |
|---|---|---|---|---|---|---|---|---|
| Base 3DETR | ScanNet | `configs/scannet/det-3detr-v0m1-0-scannet.py` | `…/scannet/v0m1.slurm` | `thesis_scannet_base_3detr_720ep` | 👀 | TBD | TBD | |
| Base 3DETR | AGCO ouster | `configs/agco/det-3detr-v1m3-0-3cls-agco-ouster.py` | `…/agco_bbox/ouster/v1m3_3cls.slurm` | `3detr_agco_v1m3_sunloss_3cls_ouster_720ep` | ✅ | TBD | TBD | |
| PTv3 from scratch | ScanNet | `configs/scannet/det-3detr-v3m1-1-scannet.py` | `…/scannet/v3m1-1.slurm` | `thesis_scannet_ptv3_fps_720ep` | 👀 | TBD | TBD | |
| PTv3 from scratch | AGCO ouster | `configs/agco/det-3detr-v3m4-0-3cls-agco-ouster.py` | `…/agco_bbox/ouster/v3m4_3cls.slurm` | `3detr_agco_v3m4_ptv3_fps_sunloss_3cls_ouster_720ep` | ⏳ | TBD | TBD | job 8827442 |
| Utonia ft | ScanNet | `configs/scannet/det-3detr-utonia-v5m1-1-scannet-nocolor.py` | `…/scannet/utonia-v5m1-1-nocolor.slurm` | `thesis_scannet_utonia_encft_fps_nocolor_720ep` | 👀 | TBD | TBD | nocolor = RQ1 baseline; the color-on variant lives in RQ2 |
| Utonia ft | AGCO ouster | `configs/agco/det-3detr-v2m4-0-3cls-agco-ouster.py` | `…/agco_bbox/ouster/v2m4_3cls.slurm` | `3detr_agco_v2m4_utonia_fps_sunloss_3cls_ouster_720ep` | ⏳ | TBD | TBD | job 8841984 |
| Utonia frozen | ScanNet | `configs/scannet/det-3detr-utonia-v4m1-0-scannet.py` | `…/scannet/utonia-v4m1-0.slurm` | `thesis_scannet_utonia_enc_fps_720ep` | 👀 | TBD | TBD | |
| Utonia frozen | AGCO ouster | `configs/agco/det-3detr-v5m2-0-3cls-agco-ouster.py` | `…/agco_bbox/ouster/v5m2_3cls.slurm` | `3detr_agco_v5m2_utonia_frozen_vanillaenc_3cls_ouster_720ep` | ✅ | TBD | TBD | |
| Utonia frozen, minimal head | ScanNet | `configs/scannet/det-3detr-utonia-v4m2-0-scannet.py` | `…/scannet/utonia-v4m2-0.slurm` | `thesis_scannet_utonia_minhead_720ep` | 👀 | TBD | TBD | Identity encoder (no Vanilla 3L) |
| Utonia frozen, minimal head | AGCO ouster | `configs/agco/det-3detr-v5m1-0-3cls-agco-ouster.py` | `…/agco_bbox/ouster/v5m1_3cls.slurm` | `3detr_agco_v5m1_utonia_frozen_minhead_3cls_ouster_720ep` | ❌ | TBD | TBD | not queued yet |

All SLURM paths are rooted at `third_party/slurm_scripts/agco-cluster/pointcept/3detr/`.

### What each slot answers

- **Base 3DETR (`v0m1` ScanNet, `v1m3` AGCO).** Native-style 3DETR with a
  PointNet++ SA pre-encoder. Reference point — the field-standard detector
  without any modern point-transformer plumbing. AGCO variant uses the
  SUN-RGBD-style criterion (`v1m3` = `v1m2` + SUN-like loss; oriented boxes
  match SUN-RGBD better than ScanNet's axis-aligned setting).

- **PTv3 from scratch (`v3m1-1` ScanNet, `v3m4` AGCO).** Same FPS-2048 token
  budget as base, but feature extractor is **PointTransformerV3 trained from
  random init**. Isolates "does PTv3 beat PointNet++ as a feature extractor
  when both models train end-to-end on the same data?" The transformer
  encoder (3L Vanilla) is preserved so the only delta vs Base 3DETR is the
  pre-encoder family. AGCO `v3m4` configures `PTv3PreEncoder` with FPS
  `npoint=2048` to match the ScanNet token budget exactly.

- **Utonia ft (`utonia-v5m1-1-scannet-nocolor` ScanNet, `v2m4` AGCO).**
  Pretrained PT-v3m3 (Utonia VFM, HuggingFace checkpoint) with
  `freeze_backbone="enc_finetune"`: embedding + early encoder stages stay
  frozen and in `eval()`, **last encoder stage** unfreezes. A leaf-bridge in
  `PTv3m3PreEncoder` starts a fresh autograd graph so gradients reach the
  last stage + 3DETR head but stop at the frozen prefix. Isolates "does
  pretraining help when we still allow a small amount of backbone
  adaptation?"

- **Utonia frozen (`utonia-v4m1-0` ScanNet, `v5m2` AGCO).** Same Utonia
  checkpoint, but `freeze_backbone="enc"` — the **entire encoder** is frozen
  in `torch.no_grad()` + `eval()`. Same Vanilla(3L) transformer encoder + 8L
  decoder + MLP heads as the ft variant — those are the only trainable
  parameters. Isolates "do Utonia features transfer to detection
  out-of-the-box, with no backbone updates whatsoever?"

- **Utonia frozen, minimal head (`utonia-v4m2-0` ScanNet, `v5m1` AGCO).**
  Identical to the frozen slot **except** the Vanilla(3L) transformer
  encoder is replaced by `IdentityEncoder3DETR` (pass-through). Only the
  3DETR decoder + MLP heads are learnable. Isolates "are Utonia features
  rich enough that we don't even need an extra trainable transformer
  encoder on top of them?" — the floor for how minimal the trainable head
  can get while keeping DETR set prediction.

### Cross-dataset architecture mapping

| Slot | ScanNet config | AGCO config |
|---|---|---|
| Base 3DETR | `det-3detr-v0m1-0-scannet.py` | `det-3detr-v1m3-0-3cls-agco-ouster.py` |
| PTv3 from scratch | `det-3detr-v3m1-1-scannet.py` | `det-3detr-v3m4-0-3cls-agco-ouster.py` |
| Utonia ft | `det-3detr-utonia-v5m1-1-scannet-nocolor.py` | `det-3detr-v2m4-0-3cls-agco-ouster.py` |
| Utonia frozen | `det-3detr-utonia-v4m1-0-scannet.py` | `det-3detr-v5m2-0-3cls-agco-ouster.py` |
| Utonia frozen, minimal head | `det-3detr-utonia-v4m2-0-scannet.py` | `det-3detr-v5m1-0-3cls-agco-ouster.py` |

Within each row the pre-encoder family, FPS budget (2048), encoder kind
(Vanilla 3L, or Identity for the minimal-head row), decoder (8L, 256d), and
schedule (720ep) are identical; only dataset-specific knobs differ (oriented
vs axis-aligned boxes, `grid_size=0.05` vs `0.01`, `num_queries=32` vs `256`,
AGCO sensor crops, SUN-like loss for AGCO).

---

## RQ2 — Per-point RGB as an input modality

> **Held fixed across all 6 RQ2 runs:** everything from RQ1 (same backbones,
> same token budget, same schedule) — only the input feature contract
> changes. For PointNet++ and PTv3-from-scratch we **add** an RGB channel;
> for Utonia we **flip the zero-pad** on the RGB slot of its 9-channel input.

### RQ2 run matrix (color-on variants)

| Slot | Dataset | Config | SLURM | Exp name | Status | AP25 | AP50 | Notes |
|---|---|---|---|---|---|---|---|---|
| Base 3DETR + color | ScanNet | `configs/scannet/det-3detr-v0m1-0-scannet-color.py` | `…/scannet/v0m1-color.slurm` | `thesis_scannet_base_3detr_color_720ep` | ⏳ | TBD | TBD | |
| Base 3DETR + color | AGCO ouster | `configs/agco/det-3detr-v1m3-0-3cls-agco-color-ouster.py` | `…/agco_bbox/ouster/v1m3_3cls_color.slurm` | `3detr_agco_v1m3_color_sunloss_3cls_ouster_720ep` | ✅ | TBD | TBD | |
| PTv3 from scratch + color | ScanNet | `configs/scannet/det-3detr-v3m1-1-scannet-color.py` | `…/scannet/v3m1-1-color.slurm` | `thesis_scannet_ptv3_fps_color_720ep` | ⏳ | TBD | TBD | |
| PTv3 from scratch + color | AGCO ouster | `configs/agco/det-3detr-v3m4-0-3cls-agco-color-ouster.py` | `…/agco_bbox/ouster/v3m4_3cls_color.slurm` | `3detr_agco_v3m4_ptv3_fps_color_sunloss_3cls_ouster_720ep` | ❌ | TBD | TBD | not queued |
| Utonia ft + color | ScanNet | `configs/scannet/det-3detr-utonia-v5m1-1-scannet.py` | `…/scannet/utonia-v5m1-1.slurm` | `thesis_scannet_utonia_encft_fps_720ep` | 👀 | TBD | TBD | The headline Utonia ScanNet config (`use_color=True`) is the **color arm** of the ablation; the **baseline** for the ablation is the `-nocolor` variant in RQ1. |
| Utonia ft + color | AGCO ouster | `configs/agco/det-3detr-v2m4-0-3cls-agco-color-ouster.py` | `…/agco_bbox/ouster/v2m4_3cls_color.slurm` | `3detr_agco_v2m4_utonia_fps_color_sunloss_3cls_ouster_720ep` | ⏳ | TBD | TBD | job 8854333 |

### Implementation deltas vs the RQ1 baseline

| Backbone | Color-config delta |
|---|---|
| Base 3DETR (PointNet++ SA) | `use_color=True` in train/val/test dataset blocks; `PointnetSAPreEncoder.mlp_dims[0]: 0 → 3` so the SA MLP accepts an extra RGB channel. |
| PTv3 from scratch | `use_color=True`; `PTv3PreEncoder.in_channels: 3 → 6` (xyz + rgb). |
| Utonia (PT-v3m3) | `use_color=True/False` only. The 9-channel `[xyz, rgb, normal]` layout is constant — `_build_padded_feat` (model) or `utonia_preprocess` (AGCO dataset) zero-pads any slot that's not provided. **No `in_channels` change** because the input contract is fixed by the pretrained backbone. |

### Why the Utonia row inverts the ablation direction

The thesis-headline Utonia ScanNet run **already** has `use_color=True`
because the PT-v3m3 checkpoint was pretrained on a 9-channel `[xyz, rgb,
normal]` input. To get a clean RGB ablation pair on ScanNet, we **add** a
no-color variant (`utonia-v5m1-1-scannet-nocolor`) which zero-fills the RGB
slot. The result:

- **RQ1 row** (Utonia ScanNet) = `utonia-v5m1-1-scannet-nocolor.py` → no color.
- **RQ2 row** (Utonia + color, ScanNet) = `utonia-v5m1-1-scannet.py` → with color.

AGCO mirrors this with two configs (`v2m4_3cls.slurm` vs
`v2m4_3cls_color.slurm`).

### Dino slot — pending

A planned 4th backbone slot — Dino-pretrained 3D features — is **not yet
available** at the time of this freeze. If it lands before submission it
joins this section as a 4th row × 2 datasets pair; otherwise it stays out of
scope. **Do not invent results for it.**

---

## RQ3 — Task axis: detection-only vs semseg-only vs multi-task (AGCO only)

> **Why AGCO-only.** (a) The multi-task model + AGCO segmentation labels are
> the thesis's primary contribution. (b) ScanNet semseg + multi-task configs
> for the PTv3/Utonia stack do not exist (would require additional setup
> outside the time budget). The detection-only column reuses the RQ1 cell
> for AGCO directly.

### RQ3 matrix (3 backbones × 3 tasks, AGCO ouster)

| Backbone | Semseg only | Detection only (RQ1) | Multi-task |
|---|---|---|---|
| **PTv3 from scratch** | `semseg-ptv3-v3m4-0-3cls-agco-ouster.py` (`semseg_v3m4_3cls.slurm`) ✅ | `det-3detr-v3m4-0-3cls-agco-ouster.py` (see RQ1) ⏳ | `multitask-3detr-ptv3-v3m4-0-3cls-agco-ouster.py` (`multitask_v3m4_3cls.slurm`) ✅ |
| **Utonia ft** | `semseg-ptv3-v2m4-0-3cls-agco-ouster.py` (`semseg_v2m4_3cls.slurm`) ✅ | `det-3detr-v2m4-0-3cls-agco-ouster.py` (see RQ1) ⏳ | `multitask-3detr-ptv3-v2m4-0-3cls-agco-ouster.py` (`multitask_v2m4_3cls.slurm`) ✅ |
| **Utonia frozen** | `semseg-ptv3-v5m4-0-3cls-agco-ouster.py` (`semseg_v5m4_3cls.slurm`) ✅ | `det-3detr-v5m2-0-3cls-agco-ouster.py` (see RQ1) ✅ | `multitask-3detr-ptv3-v5m4-0-3cls-agco-ouster.py` (`multitask_v5m4_3cls.slurm`) ✅ |

### Per-run tracker (RQ3 new rows; det-only column delegates to RQ1)

| Row | Config | SLURM | Exp name | Status | AP25 | AP50 | mIoU |
|---|---|---|---|---|---|---|---|
| Semseg, PTv3 from scratch | `semseg-ptv3-v3m4-0-3cls-agco-ouster.py` | `semseg_v3m4_3cls.slurm` | `seg_ptv3_agco_v3m4_3cls_ouster_720ep` | ✅ | — | — | TBD |
| Semseg, Utonia ft | `semseg-ptv3-v2m4-0-3cls-agco-ouster.py` | `semseg_v2m4_3cls.slurm` | `seg_ptv3_agco_v2m4_utonia_3cls_ouster_720ep` | ✅ | — | — | TBD |
| Semseg, Utonia frozen | `semseg-ptv3-v5m4-0-3cls-agco-ouster.py` | `semseg_v5m4_3cls.slurm` | `seg_ptv3_agco_v5m4_utonia_frozen_3cls_ouster_720ep` | ✅ | — | — | TBD |
| Multi-task, PTv3 from scratch | `multitask-3detr-ptv3-v3m4-0-3cls-agco-ouster.py` | `multitask_v3m4_3cls.slurm` | `mt_3detr_agco_v3m4_ptv3_fps_sunloss_3cls_ouster_720ep` | ✅ | TBD | TBD | TBD |
| Multi-task, Utonia ft | `multitask-3detr-ptv3-v2m4-0-3cls-agco-ouster.py` | `multitask_v2m4_3cls.slurm` | `mt_3detr_agco_v2m4_utonia_fps_sunloss_3cls_ouster_720ep` | ✅ | TBD | TBD | TBD |
| Multi-task, Utonia frozen | `multitask-3detr-ptv3-v5m4-0-3cls-agco-ouster.py` | `multitask_v5m4_3cls.slurm` | `mt_3detr_agco_v5m4_utonia_frozen_fps_sunloss_3cls_ouster_720ep` | ✅ | TBD | TBD | TBD |

### Architectural notes

**Semseg-only configs** (`semseg-ptv3-{v3m4,v2m4,v5m4}-0-3cls-agco-ouster.py`).
- Model: `Dense3DETRSegmentor` (a `DefaultSegmentorV2`-shaped wrapper around the same backbone family used for detection so the comparison stays clean).
- Backbone: PT-v3m1 (`v3m4` = from scratch) or `PTv3m3PreEncoder` (`v2m4`/`v5m4` = pretrained Utonia, `enc_finetune` / `enc` respectively).
- Loss: `CrossEntropyLoss` (weight 1.0) + `LovaszLoss` (weight 1.0).
- Output: **4 classes** — `0=background, 1=tractor, 2=harvester, 3=trailer`. On-disk labels also contain `4=car`; remapped to background via `seg_label_map={4: 0}` at load time. Hopper has no segment label (detection-only class).
- Transforms: same FOV ±60° + per-sensor spherical crop + GridSample(0.05) + PointSubsample(40k) as the detection configs.

**Multi-task configs** (`multitask-3detr-ptv3-{v3m4,v2m4,v5m4}-0-3cls-agco-ouster.py`).
- Model: `MultiTask3DETRSegmentor` (`pointcept/models/detection_3detr/multi_task.py`).
- Backbone: same PT-v3m1 / PTv3m3 stacks as the semseg-only configs, sharing the full U-Net for the seg branch; the **encoder bottleneck** is tapped for the det branch (snapshot via `point2dense` happens inside `_backbone_forward` before `self.backbone.dec` mutates the parent chain).
- Det head: FPS(2048) → Vanilla transformer encoder (3L; `encoder_dim=576` for Utonia rows, `512` for the PTv3-from-scratch row in some variants) → 8-layer 3DETR decoder, `num_queries=32`, SUN-like criterion (`class=1, objectness=5, giou=3, center=5` matcher; loss `giou=0, sem_cls=1, no_object=0.1, angle_cls=0.1, angle_reg=0.5, center=5, size=1`).
- Seg head: linear over the unpooled decoder output, same CE + Lovász as the semseg-only configs.
- **Loss combination: uncertainty weighting** (Cipolla form, `c_seg=c_det=2`): `L_total = (1/(c_seg σ_seg²)) L_seg + log σ_seg + (1/(c_det σ_det²)) L_det + log σ_det`. The σ scalars are learnable `nn.Parameter`s `log_sigma_sq_seg` / `log_sigma_sq_det` (init zero → σ=1). Both multi-task configs include `param_dicts=[dict(keyword="log_sigma_sq", weight_decay=0.0)]` so the σ params aren't decayed. `loss_weighting="fixed"` is available for ablation but is not the default for the RQ3 runs.
- Evaluation: `CombinedSegDetEvaluator` (hook) + `CombinedSegDetTester` (tester) — runs AP25 / AP50 and per-class mIoU / Acc in one pass over the val/test loader; also logs `val/sigma_seg` and `val/sigma_det` once per eval.

See [`multitask_ptv3_3detr.md`](multitask_ptv3_3detr.md) for the full multi-task model walkthrough — the joint loss derivation, the encoder-bottleneck snapshot, and the eval-hook interaction.

---

## Shared settings appendix

### ScanNet (held constant across RQ1 + RQ2)

| Knob | Value |
|---|---|
| Box parameterization | Axis-aligned, `num_angle_bin=1` |
| Queries | 256 (FPS-sampled from encoder output) |
| Pre-encoder output budget | 2048 tokens (built-in FPS or post-encoder FPS) |
| Encoder | `VanillaTransformerEncoder3DETR` 3L (`IdentityEncoder3DETR` for the Utonia-minimal-head slot) |
| Decoder | `TransformerDecoder3DETR` 8L, `d_model=256`, `nhead=4`, `ffn_dim=256` |
| Criterion | 3DETR defaults — matcher `(class=1, giou=2, objectness=0, center=0)`; loss `(giou=1, no_object=0.25, center=5, size=1, sem_cls=1)` |
| Optimizer / scheduler | AdamW `lr=5e-4`, `wd=0.1`; OneCycleLR cosine |
| Epochs | 720 |
| Batch size | 8 total |
| AMP | disabled |
| Dataset | `ScanNetDetectionDataset` (VoteNet-style format); transform list = ScanNet detection augs (incl. `GridSampleDetection(grid_size=0.02)` for all PTv3-using configs) |
| `grid_size` (PTv3 / Utonia) | 0.02 (ScanNet indoor) — Utonia ScanNet configs use `grid_size=0.01` to match the Utonia pretraining grid |

### AGCO (held constant across RQ1 + RQ2 + RQ3)

| Knob | Value |
|---|---|
| Classes | 3: `(tractor, harvester, trailer)`; seg labels also drop `car → bg`, no `hopper` label |
| Box parameterization | Oriented, `num_angle_bin=12` (SUN-RGBD-style yaw cls + residual) |
| Queries | 32 (FPS-sampled from encoder output) |
| `max_num_obj` | 16 |
| Pre-encoder output budget | 2048 tokens (PointNet++ SA built-in for `v1m3`; FPS post-PTv3/Utonia for `v3m4`/`v2m4`/`v5m1`/`v5m2`) |
| Encoder | Vanilla 3L (`IdentityEncoder3DETR` for the Utonia-minimal-head slot `v5m1`) |
| Decoder | 8L, `d_model=256`, `nhead=4`, `ffn_dim=256` |
| Center prediction | `center_offset_normalized=True` |
| Criterion (SUN-like) | matcher `(class=1, objectness=5, giou=3, center=5)`; loss `(giou=0, sem_cls=1, no_object=0.1, angle_cls=0.1, angle_reg=0.5, center=5, size=1)`; `giou_on_aux_outputs=False` |
| Optimizer / scheduler | AdamW `lr=5e-4`, `wd=0.1`; OneCycleLR (`pct_start=0.10`, `div_factor=500`) |
| Epochs | 720 |
| Batch size | 8 total (PTv3/Utonia heavier configs may set `gradient_accumulation_steps`) |
| AMP | disabled |
| Dataset | `AgcoBBoxV1` with `apply_t_rtk`, `apply_r_global`, `apply_r_level_to_{points,boxes}`, `require_gravity_align`, `fixed_pc_dims` per sensor, `min_inliers=dict(lslidar=350, ouster=200, rslidar=80)`; RQ3 rows additionally set `load_segment=True`, `segment_subdir="segment"`, `seg_label_map={4: 0}` |
| Detection crops | `FovCropDetection(azimuth_deg=(-60, 60))` + `SphericalCropDetection` per-sensor (40 m ouster) |
| Train-only augs | `RandomFlipDetection(p_x=0, p_y=0.5)`, `RandomRotateZDetection(±5°)` |
| `grid_size` | 0.05 (outdoor LiDAR is sparser than ScanNet) |
| `num_points` | 40 000 (ouster) |

### Sensor (ouster only for the thesis)

| Sensor | `num_points` | Spherical max_dist | `fixed_pc_dims` (x / y / z) |
|---|---|---|---|
| ouster | 40 000 | 40 m | min `[0, -36, -10]` / max `[40, 36, 8]` |

`lslidar` (100 k pts, 60 m) and `rslidar` (30 k pts, 20 m) configs exist in the
repo but are **out of thesis scope** — kept for potential follow-up work.

---

## Open items / known stale

- **Dino backbone for RQ2.** Listed in the user's run plan as "still up in
  the air whether we have time for this." No config yet. Add if it lands.
- **ScanNet RQ3 (semseg + multi-task).** Intentionally skipped — multi-task
  is the AGCO-side contribution; building a comparable ScanNet pipeline is
  out of scope for this thesis.
- **AGCO color × PTv3-from-scratch (RQ2).** SLURM exists (`v3m4_3cls_color.slurm`)
  but is `❌` — not queued. Mention as "completed config inventory, not run."
- **AGCO Utonia-frozen-minimal-head (RQ1, AGCO).** SLURM exists (`v5m1_3cls.slurm`)
  but is `❌` — not queued.
- **Stale companion file.** `notes/thesis_experiments.md` predates the
  RQ1 expansion to 5 slots and the RQ3 axis; it has been superseded by this
  file. Banner added at top of that file; body kept for pre-freeze history.

## Run command reference (verbatim from SLURMs)

```bash
# RQ1 — Base 3DETR
sh scripts/train.sh -d scannet -c det-3detr-v0m1-0-scannet                  -n thesis_scannet_base_3detr_720ep                              -g 1
sh scripts/train.sh -d agco    -c det-3detr-v1m3-0-3cls-agco-ouster         -n 3detr_agco_v1m3_sunloss_3cls_ouster_720ep                    -g 1

# RQ1 — PTv3 from scratch
sh scripts/train.sh -d scannet -c det-3detr-v3m1-1-scannet                  -n thesis_scannet_ptv3_fps_720ep                                -g 1
sh scripts/train.sh -d agco    -c det-3detr-v3m4-0-3cls-agco-ouster         -n 3detr_agco_v3m4_ptv3_fps_sunloss_3cls_ouster_720ep           -g 1

# RQ1 — Utonia ft
sh scripts/train.sh -d scannet -c det-3detr-utonia-v5m1-1-scannet-nocolor   -n thesis_scannet_utonia_encft_fps_nocolor_720ep                -g 1
sh scripts/train.sh -d agco    -c det-3detr-v2m4-0-3cls-agco-ouster         -n 3detr_agco_v2m4_utonia_fps_sunloss_3cls_ouster_720ep         -g 1

# RQ1 — Utonia frozen
sh scripts/train.sh -d scannet -c det-3detr-utonia-v4m1-0-scannet           -n thesis_scannet_utonia_enc_fps_720ep                          -g 1
sh scripts/train.sh -d agco    -c det-3detr-v5m2-0-3cls-agco-ouster         -n 3detr_agco_v5m2_utonia_frozen_vanillaenc_3cls_ouster_720ep   -g 1

# RQ1 — Utonia frozen, minimal head
sh scripts/train.sh -d scannet -c det-3detr-utonia-v4m2-0-scannet           -n thesis_scannet_utonia_minhead_720ep                          -g 1
sh scripts/train.sh -d agco    -c det-3detr-v5m1-0-3cls-agco-ouster         -n 3detr_agco_v5m1_utonia_frozen_minhead_3cls_ouster_720ep      -g 1

# RQ2 — color variants
sh scripts/train.sh -d scannet -c det-3detr-v0m1-0-scannet-color            -n thesis_scannet_base_3detr_color_720ep                         -g 1
sh scripts/train.sh -d agco    -c det-3detr-v1m3-0-3cls-agco-color-ouster   -n 3detr_agco_v1m3_color_sunloss_3cls_ouster_720ep               -g 1
sh scripts/train.sh -d scannet -c det-3detr-v3m1-1-scannet-color            -n thesis_scannet_ptv3_fps_color_720ep                           -g 1
sh scripts/train.sh -d agco    -c det-3detr-v3m4-0-3cls-agco-color-ouster   -n 3detr_agco_v3m4_ptv3_fps_color_sunloss_3cls_ouster_720ep      -g 1
sh scripts/train.sh -d scannet -c det-3detr-utonia-v5m1-1-scannet           -n thesis_scannet_utonia_encft_fps_720ep                         -g 1
sh scripts/train.sh -d agco    -c det-3detr-v2m4-0-3cls-agco-color-ouster   -n 3detr_agco_v2m4_utonia_fps_color_sunloss_3cls_ouster_720ep    -g 1

# RQ3 — semseg only (AGCO)
sh scripts/train.sh -d agco -c semseg-ptv3-v3m4-0-3cls-agco-ouster          -n seg_ptv3_agco_v3m4_3cls_ouster_720ep                           -g 1
sh scripts/train.sh -d agco -c semseg-ptv3-v2m4-0-3cls-agco-ouster          -n seg_ptv3_agco_v2m4_utonia_3cls_ouster_720ep                    -g 1
sh scripts/train.sh -d agco -c semseg-ptv3-v5m4-0-3cls-agco-ouster          -n seg_ptv3_agco_v5m4_utonia_frozen_3cls_ouster_720ep             -g 1

# RQ3 — multi-task (AGCO)
sh scripts/train.sh -d agco -c multitask-3detr-ptv3-v3m4-0-3cls-agco-ouster -n mt_3detr_agco_v3m4_ptv3_fps_sunloss_3cls_ouster_720ep          -g 1
sh scripts/train.sh -d agco -c multitask-3detr-ptv3-v2m4-0-3cls-agco-ouster -n mt_3detr_agco_v2m4_utonia_fps_sunloss_3cls_ouster_720ep        -g 1
sh scripts/train.sh -d agco -c multitask-3detr-ptv3-v5m4-0-3cls-agco-ouster -n mt_3detr_agco_v5m4_utonia_frozen_fps_sunloss_3cls_ouster_720ep -g 1
```
