# 3DETR Config Overview

This note is split by dataset:

- **ScanNet configs**: `configs/scannet/det-3detr-*.py`
- **AGCO configs**: `configs/agco/det-3detr-*.py`

## ScanNet dataset configs

ScanNet 3DETR configs are 18-class detection with axis-aligned boxes and use
the `ScanNetDetectionDataset` dataloader (`pointcept/datasets/scannet_detection.py`,
VoteNet-style preprocessed layout).

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

AGCO detection configs use `AgcoBBoxV1` + `AgcoBBoxConfig` (oriented boxes,
`num_angle_bin=12`) and vary mostly by dataset/runtime/criterion knobs rather
than by major model-composition changes.

### AGCO config families

| Family | Core idea | Typical configs |
|---|---|---|
| `v0/v1` | Base 3DETR stack (PointNet++ SA + vanilla encoder) | `v0m1-*`, `v1m1-*`, `v1m2-*`, `v1m3-*` |
| `v3` | PTv3 pre-encoder + identity/vanilla encoder | `v3m1-*`, `v3m2-*` |
| `v4` | AGCO-focused PTv3 settings (normalized center offsets, query sweeps, fixed dims) | `v4m1-*`, `v4m2-*`, `v4m3-*` |
| `*-ouster` | Sensor-specific ouster runs (40k points) | all `*-ouster` |
| `*-1` | Overfit/debug split behavior (`val/test -> train`) | `*m*-1*` |
| `*tiny5` | 5-sample deterministic overfit diagnostics via allowlist | `*tiny5*` |

### AGCO settings that matter most

1. **Class subset + label mapping**
- `included_classes=(...)` defines active classes and their order.
- Common modern setting is 3-class: `("tractor", "harvester", "trailer")`.
- Boxes outside `included_classes` are dropped at dataset load time.

2. **Oriented-box parameterization**
- `num_angle_bin=12` (SUN-RGBD style angle cls+residual) for AGCO.
- Not axis-aligned ScanNet-style detection.

3. **Coordinate/calibration stack (AGCO-specific)**
- `apply_t_rtk=True` + `require_calibration=True`: sensor->RTK transform.
- `apply_r_global=True`: apply global pitch/roll component.
- `apply_r_level_to_points=True`, `apply_r_level_to_boxes=True`:
  gravity leveling from `gravity_align.npz`.
- `require_gravity_align=True` to enforce leveling assets.

4. **Consistent normalization bounds**
- `fixed_pc_dims={...}` stabilizes center/size normalization across scenes.

5. **Sensor-specific dataset geometry**
- `sensors=["lslidar"]` or `sensors=["ouster"]`.
- Typical points: `100_000` (lslidar) vs `40_000` (ouster).

6. **Detection crop policy**
- Common deterministic crop pair:
  - `FovCropDetection(azimuth_deg=(-60, 60), crop_points=True)`
  - `SphericalCropDetection(max_dist/min_dist, per_sensor=...)`

7. **Center prediction parameterization**
- `center_offset_normalized=True` in newer AGCO configs.

8. **Query budget sweeps**
- AGCO experiments vary `num_queries` (e.g. `32`, `128`, `384`).

9. **Criterion/matcher variants**
- AGCO defaults: often `loss_giou_weight=1`, matcher center/objectness cost 0.
- SUN-like variants (`v1m3`) use matcher `{class=1, giou=3, center=5, objectness=5}`,
  with `loss_giou_weight=0` and `loss_no_object_weight=0.1`.

10. **GIoU execution mode**
- `SetCriterion3DETR` supports:
  - `giou_mode="optimized"` (split fast path)
  - `giou_mode="legacy"` (pre-optimization criterion flow, final `[-1,1]` clamp)

11. **Overfit/debug behavior**
- `*-1` configs set `val/test split="train"` for overfit checks.
- `tiny5` configs use `split_prefix="agco_tiny5"` + `sample_allowlist_file=...`.

### AGCO-specific dataset knobs (`AgcoBBoxV1`)

- `min_inliers` can be global int or per-sensor dict.
- `max_num_obj` is often reduced (e.g. 16) for AGCO scene density.
- `sample_allowlist_file` enables exact per-sample subset selection (tiny5).
- `deterministic_debug` / `deterministic_seed` exist in dataset safety-net path.

### AGCO detection transforms

| Transform | Main knobs |
|---|---|
| `FovCropDetection` | `azimuth_deg`, `elevation_deg`, `crop_points`, `per_sensor` |
| `SphericalCropDetection` | `max_dist`, `min_dist`, `per_sensor` |
| `RandomFlipDetection` | `p_x`, `p_y` |
| `RandomRotateZDetection` | `angle_deg=(lo, hi)` |
| `PointSubsampleDetection` | `num_points` (+ optional deterministic mode) |
| `RandomScaleDetection` | `scale=(lo,hi)`, `apply_to_sizes=True` |
| `RandomJitterDetection` | `sigma`, `clip` |
| `RandomCuboidDetection` | `min_points`, `aspect`, `min_crop`, `max_crop` |

Legacy `augment` and `random_cuboid_min_points` kwargs were removed.
