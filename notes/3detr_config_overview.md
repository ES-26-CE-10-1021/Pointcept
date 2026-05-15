# 3DETR Config Overview

This note is split by dataset:

- **ScanNet configs**: `configs/scannet/det-3detr-*.py`
- **AGCO configs**: `configs/agco/det-3detr-*.py`

> **Looking for the multi-task model?** The joint semseg + 3DETR detection
> path (`MultiTask3DETRSegmentor`, `CombinedSegDetEvaluator`,
> `CombinedSegDetTester`, the `multitask-3detr-ptv3-*` configs, the
> uncertainty-weighted loss combination, and the AGCO segmentation label
> layout) is documented separately in
> [`multitask_ptv3_3detr.md`](multitask_ptv3_3detr.md). This file covers
> the detection-only configs.

## ScanNet dataset configs

ScanNet 3DETR configs are 18-class detection with axis-aligned boxes and use
the `ScanNetDetectionDataset` dataloader (`pointcept/datasets/scannet_detection.py`,
VoteNet-style preprocessed layout).

**Augmentation** is config-driven via a `transform=[...]` list (Pointcept-style).
The same pipeline now drives **both** ScanNet (`ScanNetDetectionDataset`) and
AGCO (`AgcoBBoxV1`) — ScanNet's legacy `augment=True` /
`random_cuboid_min_points` kwargs were removed and replaced by the same
detection-aware transform list. ScanNet keeps `use_color`, `use_height`, and
`utonia_preprocess` as deterministic dataset kwargs that run before the
transform pipeline. PTv3-using ScanNet configs (v2/v3/v4 + all utonia
variants) include `GridSampleDetection` in every split's transform list with
`grid_size` matching the pre-encoder. Both datasets accept an optional
`load_segment=False` kwarg for the upcoming multi-task semseg branch.

Detection-aware transforms registered in `pointcept/datasets/det_transform.py`:

| Transform                  | Knobs                                                             |
|----------------------------|-------------------------------------------------------------------|
| `RandomFlipDetection`      | `p_x`, `p_y` — independent X/Y mirror; updates centers + yaws.     |
| `RandomRotateZDetection`   | `angle_deg=(lo, hi)` in degrees; rotates points + centers + yaws.  |
| `RandomScaleDetection`     | `scale=(lo, hi)`, `apply_to_sizes=True`.                           |
| `RandomJitterDetection`    | `sigma`, `clip` — Gaussian jitter on point XYZ only.               |
| `RandomCuboidDetection`    | `min_points`, `aspect`, `min_crop`, `max_crop`; filters boxes.     |
| `GridSampleDetection`      | `grid_size`, `hash_type`; one random point per voxel (mirrors seg `GridSample` train mode). Place after geometric augs, before `PointSubsampleDetection`. Required for PTv3 pre-encoders so the input contract (one feature per voxel) is honoured. |
| `PointSubsampleDetection`  | `num_points` — fixed-size subsample; dataset enforces as fallback. |

All point-subsampling / point-cropping transforms (`RandomCuboidDetection`,
`SphericalCropDetection`, `FovCropDetection`, `GridSampleDetection`,
`PointSubsampleDetection`) index every key in `det_transform._PER_POINT_KEYS`
(currently `point_cloud` and `segment`) in lockstep, so optional per-point
labels for the upcoming multi-task semseg branch survive the pipeline.

Legacy `augment` and `random_cuboid_min_points` kwargs were removed.

## Config summary

| Config            | Pre-Encoder                         | pre_enc out dim | Encoder                 | encoder_dim | decoder_dim | Epochs | Purpose                                                |
| ----------------- | ----------------------------------- | --------------- | ----------------------- | ----------- | ----------- | ------ | ------------------------------------------------------ |
| **v0m1-0**        | PointNet++ SA (2048 pts)            | 256             | VanillaTransformer (3L) | 256         | 256         | 720    | Native 3DETR baseline                                  |
| **v0m1-0-180ep**  | _(inherits v0)_                     | 256             | _(inherits v0)_         | 256         | 256         | 180    | Shortened v0 for quick comparison                      |
| **v1m1-0**        | PointNet++ SA (2048 pts)            | 256             | VanillaTransformer (3L) | 256         | 256         | 90     | Pointcept integration sanity check                     |
| **v2m1-0**        | PTv3 (enc→512)                      | 512             | IdentityEncoder         | 512         | 256         | 720    | PTv3; no transformer encoder                           |
| **v2m1-0-180ep**  | PTv3 (enc→512)                      | 512             | IdentityEncoder         | 512         | 256         | 180    | Shortened v2 for quick comparison                      |
| **v3m1-0**        | PTv3 (enc→512)                      | 512             | VanillaTransformer (3L) | 512         | 256         | 720    | PTv3 + transformer encoder                             |
| **v3m1-1**        | PTv3 (enc→256) + FPS (2048 pts)     | 256             | VanillaTransformer (3L) | 256         | 256         | 720    | PTv3 + FPS to fixed-length output                      |
| **v4m1-0**        | PTv3UNet (dec→64)                   | 64              | IdentityEncoder         | 64          | 256         | 720    | PTv3 U-Net; no transformer encoder                     |
| **v4m1-1**        | PTv3UNet (dec→64)                   | 64              | VanillaTransformer (3L) | 64          | 256         | 720    | PTv3 U-Net + transformer encoder                       |
| **utonia-v1m1-0** | Frozen Utonia PT-v3m3 (enc→576)     | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | Frozen pretrained Utonia VFM                           |
| **utonia-v2m1-0** | Frozen Utonia PT-v3m3 (enc→576)     | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | v1 + Utonia-canonical preprocessing                    |
| **utonia-v3m1-0** | Frozen Utonia PT-v3m3 (enc→576)     | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | v2 + RGB input                                         |
| **utonia-v4m1-0** | Frozen Utonia enc-only + FPS 2048   | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | Utonia enc-only + fixed-length FPS                     |
| **utonia-v5m1-0** | Utonia, last enc stage trainable    | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | Fine-tune Utonia's last encoder stage                  |
| **utonia-v5m1-1** | Utonia, last enc stage + FPS 2048   | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | v5m1-0 + fixed-length FPS                              |
| **utonia-v5m2-0** | Frozen Utonia enc + fresh PTv3 dec  | 54              | VanillaTransformer (3L) | 54          | 256         | 720    | Learn a fresh decoder on frozen Utonia features        |
| **utonia-v5m2-1** | Frozen Utonia enc + fresh dec + FPS | 54              | VanillaTransformer (3L) | 54          | 256         | 720    | v5m2-0 + fixed-length FPS                              |

**Encoder layers**: "(3L)" means 3 transformer layers (`nlayers=3`). The decoder always uses 8 layers.

**Dimension flow**: pre_enc out dim → encoder (preserves dim) → `encoder_to_decoder_projection` → decoder_dim → decoder → MLP heads.
The projection maps `encoder_dim` → `decoder_dim` (always 256). The decoder and MLP heads both operate at `decoder_dim`.

**Projection norm**: All PTv3-based configs use `projection_norm="ln"` (LayerNorm, padding-safe). The PointNet++ configs use `bn1d` (default).

## What each version tests

### v0 — Native 3DETR baseline

Replicates the original 3DETR paper setup within Pointcept's framework. PointNet++ SA
downsamples to 2048 points, a 3-layer vanilla transformer encoder refines features,
and the 8-layer decoder predicts boxes. All criterion/matcher settings match native defaults.

### v1 — Pointcept integration sanity check

Same architecture as v0 but with a shorter training schedule (90 epochs) and
different LR warmup/decay (div_factor=10, final_div_factor=1000). Used to verify
the Pointcept training loop produces comparable results to native 3DETR.

### v2 — PTv3 as feature encoder (no transformer encoder)

Replaces PointNet++ SA with PTv3 (5-stage, channels 32→512). The transformer
encoder is replaced by `IdentityEncoder` (passthrough) — the hypothesis is that
PTv3's serialized attention already provides sufficient feature mixing.
Variable-length voxel output requires padding + LayerNorm projection.

### v3m1-0 — PTv3 + transformer encoder

Same PTv3 pre-encoder as v2, but adds a 3-layer vanilla transformer encoder
between PTv3 and the decoder. Tests whether additional transformer refinement
on top of PTv3 features improves detection.

### v3m1-1 — PTv3 + FPS + transformer encoder

PTv3 pre-encoder with FPS downsampling to a fixed 2048 points, then a 3-layer
transformer encoder. FPS produces fixed-length output, eliminating the need for
padding masks. Encoder channels reduced to 256 to match decoder_dim directly.

### v4m1-0 — PTv3 U-Net (no transformer encoder)

Full PTv3 U-Net (encoder + decoder with skip connections) producing high-resolution
features at the original voxel grid (~40K voxels at 0.02m). Uses `IdentityEncoder`
since the U-Net's multi-scale skip connections already provide cross-scale context.
Output dim is 64 (dec_channels[0]), projected to 256 for the decoder.

### v4m1-1 — PTv3 U-Net + transformer encoder

Same PTv3 U-Net as v4m1-0, but adds a 3-layer vanilla transformer encoder (64d)
to refine the U-Net features before the decoder. Tests whether the additional
transformer refinement improves over the U-Net's skip connections alone.
Comparison with v4m1-0 isolates the effect of the transformer encoder on U-Net features.

### utonia-v1m1-0 — Frozen pretrained Utonia VFM

Swaps the PT-v3m1 pre-encoder for a **frozen, pretrained Utonia (PT-v3m3)
backbone**, pulled from HuggingFace on first use. Only the 3DETR transformer
encoder, decoder, and MLP heads are trained — the Utonia weights are never
updated. Utonia's 9-dim `[xyz, rgb, normal]` input contract is honoured by
zero-padding the missing modalities on-device (Causal Modality Blinding makes
this tolerable). The frozen backbone runs under `torch.no_grad()` for VRAM
savings and forced eval mode for feature determinism (controlled by the single
`freeze_backbone="enc"` enum on `PTv3m3PreEncoder`). Same 3-layer vanilla
transformer encoder as v3m1-0, but widened to `encoder_dim=576` to match
Utonia's deepest-stage output.

### utonia-v2m1-0 — v1 + Utonia-canonical preprocessing

Identical model to v1 but threads Utonia's pretraining-time preprocessing
(`RandomScale=0.5` + `CenterShift(apply_z=True)`, `grid_size=0.01`) into the
detection dataset via `utonia_preprocess=True`. Tests whether matching Utonia's
training-time point distribution improves its frozen features' utility for
detection. Still XYZ-only — RGB/normal slots stay zero-padded.

### utonia-v3m1-0 — v2 + RGB input

Adds `use_color=True` with Utonia-style `rgb/255` normalization so RGB flows
into Utonia's RGB input slots. Normals are still absent and zero-padded.
Tests whether restoring one of Utonia's three pretraining modalities narrows
the gap to the fully-populated VFM setting.

### utonia-v4m1-0 — Frozen Utonia encoder-only + FPS(2048)

v3m1-0 with FPS downsampling of the variable-length Utonia encoder output to
a fixed 2048-token budget. Eliminates the padding mask on the transformer
encoder / decoder side and matches `det-3detr-v3m1-1-scannet.py`'s token
count. Terminates the strictly-frozen-VFM line of experiments (v1–v4); the
v5 family below explores partial fine-tuning.

### utonia-v5 family — partial fine-tuning

The Utonia checkpoint is encoder-only (`dec_*=None` in the saved config),
so v5 explores two orthogonal ways of *relaxing* the freeze on top of
Utonia's pretrained encoder, each at two token budgets:

|         | no FPS (variable-length) | FPS 2048 (fixed-length) |
|---------|---------------------------|--------------------------|
| last enc stage trainable | **v5m1-0** | **v5m1-1** |
| whole enc frozen + fresh PTv3 decoder | **v5m2-0** | **v5m2-1** |

#### utonia-v5m1-0 / -1 — last encoder stage fine-tuned

`freeze_backbone="enc_finetune"`: embedding + encoder stages 0..N-2 stay
frozen and in `eval()` (forward runs under `torch.no_grad()` for VRAM);
the last encoder stage is `train()` mode and `requires_grad=True`. A
leaf-bridge on `point.feat` between the frozen prefix and the trainable
last stage starts a fresh autograd graph so backprop reaches the last
stage (and the 3DETR head) but stops before the frozen parameters.
v5m1-1 adds FPS(2048) to get a fixed-length token budget.

#### utonia-v5m2-0 / -1 — frozen encoder + fresh PTv3 decoder

`enc_mode=False` with `freeze_backbone="enc"`: the Utonia encoder runs
frozen under `torch.no_grad()`, then a **randomly-initialized** PTv3
decoder (built from the explicit `dec_*` kwargs in the config —
`dec_channels=(54, 108, 216, 432)` with head_dim=18 to satisfy 3D-RoPE)
trains from scratch on top. `_load_pretrained_state` logs every `dec.*`
key as expected-missing at INFO — that is intentional, not an error.
v5m2-1 adds FPS(2048).

## Shared settings

All configs share these settings (matching native 3DETR where applicable):

- **Optimizer**: AdamW, lr=5e-4, weight_decay=0.1
- **Scheduler**: OneCycleLR, cosine annealing
- **Decoder**: TransformerDecoder3DETR, decoder_dim=256, nhead=4, nlayers=8, ffn_dim=256
- **Queries**: 256
- **Criterion**: Native 3DETR defaults (see `third_party/3detr/scripts/scannet_ep1080.sh`)
  - Matcher: cost_class=1, cost_giou=2, cost_objectness=0, cost_center=0
  - Loss: loss_giou=1, loss_no_object=0.25, loss_center=5, loss_size=1, loss_sem_cls=1
- **batch_size**: 8 (total across GPUs)
- **AMP**: disabled
- **Dataset**: ScanNetDetectionDataset (VoteNet-style format)

## Key architectural differences

```
 PointNet++ SA      ─┐
                     │
 PTv3 enc-only       ─┤
    (+ FPS?)         │
                     ├──▶ [optional encoder: Identity | Vanilla] ──▶ projection ──▶ 3DETR decoder ──▶ MLP heads
 PTv3 U-Net         ─┤         encoder_dim                              → 256            256              256
                     │
 Utonia (frozen m3) ─┘
```

- **Variable-length path** (v2, v3m1-0, v4m1-0, v4m1-1, utonia-v1m1-0, utonia-v2m1-0, utonia-v3m1-0, utonia-v5m1-0, utonia-v5m2-0): padding_mask + LN projection
- **Fixed-length path** (v0, v1, v3m1-1, utonia-v4m1-0, utonia-v5m1-1, utonia-v5m2-1): no padding needed

## AGCO dataset configs

AGCO 3DETR detection configs target outdoor agricultural LiDAR scenes
(tractor / harvester / trailer / car / hopper) collected from three sensors
(lslidar, ouster, rslidar). They use `AgcoBBoxV1` + `AgcoBBoxConfig`
(oriented boxes, SUN-RGBD-style angle encoding with `num_angle_bin=12`)
and a per-sensor cropping + gravity-leveling pipeline.

### Naming conventions

A config filename `det-3detr-<vXmY>-<S>-<C>-agco[-<sensor>][-tiny5].py` decomposes as:

| Token | Meaning |
|---|---|
| `vXmY` | Model "version" (X) and sub-version (Y). See table below. |
| `-0` / `-1` | `-0` = normal train/val/test splits. `-1` = overfit (val and test both point to `split="train"`). |
| `-3cls` | Class subset `("tractor", "harvester", "trailer")`. Absent ⇒ 5-class (adds `car`, `hopper`). Boxes outside the active set are dropped at load time. |
| `-ouster` / `-rslidar` | Single-sensor variant: `sensors=["ouster"]` (40k pts) or `["rslidar"]` (30k pts). Absent ⇒ lslidar (100k pts). |
| `-tiny5` | 5-sample debug subset via `split_prefix="agco_tiny5"` + a `sample_allowlist_file`. All splits load the same tiny set. |
| `-debug` / `-debug-v2` | Smoke-test variants of v0m1-0: short schedules, deterministic sampling. |

### `m#` sub-axis convention (within `v1`, `v2`, `v3`)

The sub-version index has a rough "stack-of-features" convention:

| `m#` | Adds on top of `m1` |
|---|---|
| **m1** | Baseline architecture for that backbone family. |
| **m2** | + AGCO scene-scale knobs: `center_offset_normalized=True`, `fixed_pc_dims`, `num_queries=32`, `max_num_obj=16`. |
| **m3** | + SUN-RGBD-style criterion: matcher `class=1/objectness=5/giou=3/center=5`, loss `giou=0/no_object=0.1`. |
| **m4** | + FPS post-downsampling of the pre-encoder output to a fixed 2048-token budget. |

**`v4` breaks this convention** — it uses `m#` as a `num_queries` axis instead:
`v4m1`=128, `v4m2`=384, `v4m3`=32 (v4m3 is functionally equivalent to v3m2).

### Model "version" semantics on AGCO

| Family | Pre-encoder | Encoder | Sub-versions used today |
|---|---|---|---|
| `v0m1` | PointnetSAPreEncoder (2048 pts) | Vanilla(256d, 3L) | `v0m1` only — native 3DETR baseline. |
| `v1m1`/`v1m2`/`v1m3` | PointnetSAPreEncoder (2048 pts) | Vanilla(256d, 3L) | m1 baseline, m2 scene-scale knobs, m3 SUN-like loss. |
| `v2m1`/`v2m3`/`v2m4` | PTv3m3PreEncoder (Utonia, `enc_finetune`) | Vanilla(576d, 3L) | m1 baseline, m3 scene-scale + SUN loss, m4 + FPS(2048). |
| `v3m1`/`v3m2`/`v3m3` | PTv3PreEncoder (no FPS) | IdentityEncoder3DETR | m1 baseline (encoder_dim=512), m2 scene-scale knobs, m3 + SUN loss. |
| `v3m4` | PTv3PreEncoder (FPS `npoint=2048`) | Vanilla(256d, 3L) | Only m4 — PTv3 with FPS to fixed-length, plus a transformer encoder. Mirrors scannet `v3m1-1`. |
| `v4m1`/`v4m2`/`v4m3` | PTv3PreEncoder (no FPS) | IdentityEncoder3DETR | Query-density sweep on the v3m2 base: 128 / 384 / 32. |

### Active 3-class config matrix

These are the configs in current rotation (3-class line). All use `epoch=720`, `eval_epoch=20`, AdamW (lr=5e-4, wd=0.1), OneCycleLR (cos), `enable_amp=False`, oriented boxes (`num_angle_bin=12`), gravity-leveled, fixed_pc_dims, ±60° FOV + spherical crops on all splits, `center_offset_normalized=True`, `max_num_obj=16`. The `-0`/`-1` axis is split behavior; only the `-0` row is shown — `-1` variants exist where listed and differ only by val/test split = train.

| Config (3cls) | Pre-encoder | Encoder | enc_dim | num_queries | Criterion | -1 exists? | lslidar | ouster | rslidar |
|---|---|---|---|---|---|---|---|---|---|
| `v1m1` | PointNet++ SA (2048) | Vanilla(256, 3L) | 256 | 128 | 3DETR | ✓ | ✓ | — | — |
| `v1m2` | PointNet++ SA (2048) | Vanilla(256, 3L) | 256 | 32 | 3DETR | ✓ | ✓ | ✓ | — |
| `v1m3` | PointNet++ SA (2048) | Vanilla(256, 3L) | 256 | 32 | SUN-like | ✓ | ✓ | ✓ | ✓ (`-0` only) |
| `v2m1` | Utonia (`enc_finetune`) | Vanilla(576, 3L) | 576 | 128 | 3DETR | ✓ | ✓ | — | — |
| `v2m3` | Utonia (`enc_finetune`) | Vanilla(576, 3L) | 576 | 32 | SUN-like | ✓ (ouster only) | — | ✓ | ✓ (`-0` only) |
| `v2m4` | Utonia + FPS 2048 | Vanilla(576, 3L) | 576 | 32 | SUN-like | — | ✓ (`-0` only) | ✓ (`-0` only) | ✓ (`-0` only) |
| `v3m1` | PTv3 (no FPS) | Identity | 512 | 128 | 3DETR | ✓ | ✓ | — | — |
| `v3m2` | PTv3 (no FPS) | Identity | 512 | 32 | 3DETR | ✓ | ✓ | ✓ | — |
| `v3m3` | PTv3 (no FPS) | Identity | 512 | 32 | SUN-like | ✓ (ouster only) | — | ✓ | ✓ (`-0` only) |
| `v3m4` | PTv3 + FPS 2048 | Vanilla(256, 3L) | 256 | 32 | SUN-like | — | ✓ (`-0` only) | ✓ (`-0` only) | ✓ (`-0` only) |
| `v4m1` | PTv3 (no FPS) | Identity | 512 | **128** | 3DETR | ✓ | ✓ | ✓ | — |
| `v4m2` | PTv3 (no FPS) | Identity | 512 | **384** | 3DETR | ✓ | ✓ | ✓ | — |
| `v4m3` | PTv3 (no FPS) | Identity | 512 | **32** | 3DETR | ✓ | ✓ | ✓ | — |

Tiny-5 debug configs: `v4m1-1-3cls-agco-tiny5` (lslidar) and `v4m1-1-3cls-agco-ouster-tiny5`. Despite the v4m1 name, both use `num_queries=32` (so model-wise they match v3m2; the v4m1 file label is historical).

### Legacy 5-class configs

| Config (5cls) | Architecture | num_queries | Criterion | Splits | Notes |
|---|---|---|---|---|---|
| `v0m1-0-agco` | PointNet++ SA + Vanilla(256, 3L) | 128 | 3DETR | normal | Native 3DETR baseline on AGCO. No gravity leveling, no fixed_pc_dims, no center_offset_normalized. min_inliers=500 (global). |
| `v0m1-1-agco` | as v0m1-0 | 128 | 3DETR | overfit | val/test = train. |
| `v0m1-0-agco-debug` | inherits v0m1-0 | 128 | 3DETR | normal | epoch=2, deterministic sampling, seed=123. |
| `v0m1-0-agco-debug-v2` | inherits debug + gravity align + objectness/center matcher costs | 128 | 3DETR (extended matcher) | normal | epoch=40. Diagnostic config — not a v1m3-style SUN criterion. |
| `v1m1-{0,1}-agco` | PointNet++ SA + Vanilla(256, 3L) | 128 | 3DETR | normal/overfit | Adds AGCO geometry: gravity leveling, ±60° FOV + spherical crops on all splits, min_inliers=350. |
| `v2m1-{0,1}-agco` | Utonia (`enc_finetune`) + Vanilla(576, 3L) | 128 | 3DETR | normal/overfit | batch_size=2, gradient_accumulation_steps=4. |
| `v3m1-{0,1}-agco` | PTv3 (no FPS) + Identity | 128 | 3DETR | normal/overfit | batch_size=4, gradient_accumulation_steps=2. |

These are kept for ablation history; current experiments use the 3-class line.

### Family narratives

- **v0m1** — Replicates native 3DETR on AGCO data without any AGCO-specific scene-scale handling. Useful as a "what does stock 3DETR do here" reference; weakened by the ±0.5 m per-query center cap on outdoor scenes.

- **v1m1** — v0m1 + AGCO geometry (gravity leveling, ±60° FOV + spherical crops, calibrated sensor→RTK transform). Same model and criterion, just real AGCO frame handling.

- **v1m2** — v1m1 + AGCO scene-scale knobs: `center_offset_normalized=True` (lifts the per-query reach cap), `fixed_pc_dims`, `num_queries=32` (matched to AGCO scene density), `max_num_obj=16`. Still 3DETR criterion.

- **v1m3** — v1m2 + SUN-RGBD-style criterion. Replaces 3DETR's GIoU-heavy loss with a class/objectness/center-driven matcher (`loss_giou=0`, `loss_no_object=0.1`). Works better when oriented-box GIoU is expensive/noisy.

- **v2m1** — Swap PointNet++ for Utonia (pretrained PT-v3m3 VFM) with the last encoder stage trainable (`freeze_backbone="enc_finetune"`). XYZ-only input, zero-padded RGB/normal channels.

- **v2m3** — Utonia + v1m2 scene-scale knobs + SUN-like criterion. The "AGCO-tuned Utonia" line.

- **v2m4** — v2m3 + FPS to a fixed 2048-token budget on the Utonia encoder output. Mirrors scannet `utonia-v5m1-1`.

- **v3m1** — PTv3 (not Utonia) as pre-encoder with Identity encoder (PTv3's serialized attention handles feature mixing). Variable-length voxel output with padding-mask threading.

- **v3m2** — v3m1 + AGCO scene-scale knobs. `num_queries=32`. The current PTv3 baseline.

- **v3m3** — v3m2 + SUN-like criterion. PTv3 counterpart of v1m3.

- **v3m4** — Different shape: PTv3 + FPS(2048) + a 3-layer Vanilla transformer encoder (instead of Identity). Mirrors scannet `v3m1-1`. SUN-like criterion.

- **v4m1 / v4m2 / v4m3** — Query-density sweep on the v3m2 base (PTv3 + Identity + AGCO knobs, 3DETR criterion): 128 / 384 / 32 queries. **v4m3 is functionally equivalent to v3m2**; it's kept as the canonical sparse-query reference point for the sweep.

### Shared settings

All AGCO 3-class configs share:
- **Optimizer**: AdamW, lr=5e-4, weight_decay=0.1
- **Scheduler**: OneCycleLR (cosine, `pct_start=0.10`, `div_factor=500`)
- **Schedule**: 720 epochs, eval every 20 epochs
- **Decoder**: TransformerDecoder3DETR, decoder_dim=256, nhead=4, nlayers=8, ffn_dim=256
- **AMP**: disabled
- **batch_size**: 8 (total) for most configs.
  - lslidar v3/v4 (PTv3 + 100k pts): `batch_size=4`, `gradient_accumulation_steps=2` for VRAM.
  - Utonia v2m1: `batch_size=2`, `gradient_accumulation_steps=4`.
- **Oriented boxes**: AgcoBBoxConfig, num_angle_bin=12
- **Calibration stack**: `apply_t_rtk=True` (+ `require_calibration=True`), `apply_r_global=True`
- **Gravity leveling**: ON for all v1m1+ configs (`apply_r_level_to_points`, `apply_r_level_to_boxes`, `require_gravity_align`)
- **Crops** (deterministic, applied to all splits): `FovCropDetection(azimuth_deg=(-60, 60))` + `SphericalCropDetection(per_sensor)` with `max_dist`=60/40/20 m for lslidar/ouster/rslidar.
- **Train-only augs**: `RandomFlipDetection(p_x=0, p_y=0.5)` + `RandomRotateZDetection(±5°)`. (`RandomScale`, `RandomJitter`, `RandomCuboid` are registered but not used in current configs.)
- **min_inliers** (per-sensor dict): `{lslidar: 350, ouster: 200, rslidar: 80}` for v1m2+. Older v0m1/v1m1 use a global int (500/350).
- **max_num_obj**: 16 for v1m2+; 64 in legacy v0/v1m1.

### AGCO-specific dataset knobs (`AgcoBBoxV1`)

- `included_classes` — defines active classes and their order. Boxes outside this set are dropped at load time.
- `min_inliers` — global int or per-sensor dict; parents must pass for their children (e.g. trailer hoppers) to be expanded.
- `sensors` — list of sensors to load (no cross-sensor fusion).
- `fixed_pc_dims` — per-sensor `{min, max}` for consistent center/size normalization.
- `apply_t_rtk` / `require_calibration` — sensor → RTK transform via `calibration.yml`.
- `apply_r_global` — apply global pitch/roll component.
- `apply_r_level_to_{points,boxes}` + `require_gravity_align` — gravity leveling from `gravity_align.npz`.
- `utonia_preprocess` — applies Utonia's pretraining-time preprocessing and right-pads point clouds from 3 ch to the 9-ch [xyz, rgb, normal] contract.
- `sample_allowlist_file` — exact per-sample subset selection (tiny5).
- `deterministic_debug` / `deterministic_seed` — deterministic point subsampling for debug runs.

### AGCO detection transforms

Registered in `pointcept/datasets/det_transform.py`:

| Transform | Main knobs |
|---|---|
| `FovCropDetection` | `azimuth_deg`, `elevation_deg`, `crop_points`, `per_sensor` |
| `SphericalCropDetection` | `max_dist`, `min_dist`, `per_sensor` |
| `RandomFlipDetection` | `p_x`, `p_y` |
| `RandomRotateZDetection` | `angle_deg=(lo, hi)` |
| `PointSubsampleDetection` | `num_points` (+ optional `deterministic`/`seed`) |
| `RandomScaleDetection` | `scale=(lo,hi)`, `apply_to_sizes=True` |
| `RandomJitterDetection` | `sigma`, `clip` |
| `RandomCuboidDetection` | `min_points`, `aspect`, `min_crop`, `max_crop` |

Legacy `augment` and `random_cuboid_min_points` kwargs were removed.
