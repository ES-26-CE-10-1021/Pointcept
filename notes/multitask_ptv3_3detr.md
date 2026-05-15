# Multi-task PTv3 + 3DETR (semseg + detection)

`MultiTask3DETRSegmentor` (`pointcept/models/detection_3detr/multi_task.py`)
trains a shared PT-v3m1 backbone with two task heads:

- **Semantic segmentation** off the standard PTv3 decoder output (full U-Net,
  unpooled to root resolution — identical to a vanilla `DefaultSegmentorV2`
  setup).
- **3DETR detection** (transformer encoder + decoder + box criterion) tapping
  the **encoder bottleneck**, after `self.backbone.enc` runs but before
  `self.backbone.dec` walks the parent chain.

For *standalone* PTv3 semseg keep using `DefaultSegmentorV2` +
`configs/scannet/semseg-pt-v3m1-0-base.py`. For *standalone* 3DETR detection
keep using `Model3DETRDetector` + a PTv3-based pre-encoder. This class is
purpose-built for the joint-loss multi-task path.

## Forward flow

```
input (B, N, 3+C)
   │
   └─► dense2point ──► serialization ──► sparsify ──► embedding
                                                          │
                                                          ▼
                                               ┌─► self.backbone.enc ──┐
                                               │                       │
                                               │     point2dense ◄─────┤   (snapshot — NOT mutated by dec)
                                               │     enc_xyz, enc_feat,│
                                               │     padding_mask      │
                                               │                       │
                                               │                       ▼
                                               │            ┌──────────────────────┐
                                               │            │ 3DETR det branch     │
                                               │            │  • det_encoder       │
                                               │            │  • encoder→decoder   │
                                               │            │    projection        │
                                               │            │  • FPS query sampling│
                                               │            │  • det_decoder       │
                                               │            │  • MLP box heads     │
                                               │            │  • SetCriterion3DETR │
                                               │            └──────────────────────┘
                                               │                       │
                                               │                       ▼
                                               │                  L_det
                                               │
                                               └─► self.backbone.dec ──► dec_point
                                                                          │
                                                                          ▼
                                                                  unpool to root
                                                                          │
                                                                          ▼
                                                                  Linear seg_head
                                                                          │
                                                                          ▼
                                                                  CE + Lovasz
                                                                          │
                                                                          ▼
                                                                  L_seg
```

## Why snapshot before `dec` runs

`self.backbone.dec` walks the encoder's `pooling_parent` chain in-place,
popping `pooling_parent` / `pooling_inverse` keys off each level. By the time
it returns, the original `enc_point` Python reference points at a Point
several stages up (with feature dim already transitioned away from
`enc_channels[-1]`). The det branch therefore needs the bottleneck *as dense
tensors* before `dec` runs — see `_backbone_forward` in
`pointcept/models/detection_3detr/multi_task.py`.

## Loss combination

Default is **uncertainty weighting** following Cipolla et al. (the same scheme
PattFormer uses for joint det+seg on LiDAR), in its literal `c_i = 2` form:

```
L_total = (1 / (c_seg σ_seg²)) L_seg + log σ_seg
       + (1 / (c_det σ_det²)) L_det + log σ_det
```

Implementation detail: each σ is stored as a learnable scalar
`log_sigma_sq_seg` / `log_sigma_sq_det` (`s_i = log σ_i²`), with
σ_i = exp(s_i / 2) so the `+ log σ_i` regulariser becomes `+ 0.5 * s_i`. Init
zero → σ = 1, which on the first step is equivalent to fixed weights of 1.0
each — making A/B against the fixed-weight baseline straightforward.

The within-branch weights (3DETR's `loss_weight_dict` over giou / sem_cls /
center / size / no_object / angle_cls / angle_reg, and the seg `Criteria`
list of CE + Lovasz) keep their hand-tuned defaults — only the two top-level
branch weights become learnable.

Set `loss_weighting="fixed"` in the config to fall back to the static
`seg_weight * L_seg + det_weight * L_det` combination for ablation against
the prior fixed-weight baseline.

### Optimizer wiring

The σ params are **scale parameters, not weights**, so they should not be
weight-decayed. Both multi-task configs include:

```python
param_dicts = [dict(keyword="log_sigma_sq", weight_decay=0.0)]
```

`pointcept/utils/optimizer.py:23` consumes this by substring-matching the
parameter names. The σ params land in their own param group with
`weight_decay=0` (visible in the optimizer log line at startup).

## Configurable parameters

`MultiTask3DETRSegmentor.__init__` kwargs:

| Kwarg | Type | Default | Purpose |
|---|---|---|---|
| `backbone` | dict | required | Built via `build_model`. Must be `PT-v3m1` with `enc_mode=False` (full U-Net required for the seg branch). |
| `backbone_grid_size` | float | required | Written into `Point["grid_size"]` — must match the upstream `GridSampleDetection.grid_size`. |
| `num_seg_classes` | int | required | Output classes for the Linear seg head. |
| `backbone_out_channels` | int | required | = `dec_channels[0]` of the backbone. Channels of the unpooled decoder output that feed `seg_head`. |
| `seg_criteria` | list[dict] | required | Built via `build_criteria`. Typically `[CrossEntropyLoss, LovaszLoss]`. |
| `seg_ignore_index` | int | `-1` | Forwarded to the seg criteria; per-point labels matching this value are skipped in the loss. |
| `det_encoder` | dict | required | 3DETR transformer encoder (`IdentityEncoder3DETR` / `VanillaTransformerEncoder3DETR` / `MaskedTransformerEncoder3DETR`). |
| `det_decoder` | dict | required | `TransformerDecoder3DETR`. |
| `det_dataset_config` | dict | required | `ScanNetDetectionConfig` or `AgcoBBoxConfig` — supplies `num_semcls` / `num_angle_bin` to the box heads + criterion. |
| `det_criterion` | dict | required | `SetCriterion3DETR` config. |
| `encoder_dim` | int | required | Channel count of the encoder bottleneck (= `enc_channels[-1]`). Drives `encoder_to_decoder_projection`. |
| `decoder_dim` | int | `256` | Decoder + MLP-head channel count. |
| `num_queries` | int | `128` | Box queries (= max detections per scene). |
| `position_embedding` | str | `"fourier"` | `"fourier"` or `"sine"`. |
| `mlp_dropout` | float | `0.3` | Dropout in the MLP box heads. |
| `projection_norm` | str | `"ln"` | `"ln"` (LayerNorm, padding-safe — required for variable-length PTv3 output) or `"bn1d"`. |
| `loss_weighting` | str | `"uncertainty"` | `"uncertainty"` (Cipolla learnable σ) or `"fixed"` (static `seg_weight` / `det_weight` for ablation). |
| `c_seg` | float | `2.0` | Cipolla `c_i` for the seg branch — `2` for regression-like, `1` for classification-like. The seg branch is a CE+Lovasz mixture; `2` matches the literal thesis form. |
| `c_det` | float | `2.0` | Cipolla `c_i` for the det branch. |
| `seg_weight` | float | `1.0` | Used only when `loss_weighting="fixed"`. |
| `det_weight` | float | `1.0` | Used only when `loss_weighting="fixed"`. |

### Returned dicts

**Train mode** (when `loss_weighting="uncertainty"`):

```python
{
    "loss":      <combined scalar tensor>,
    "loss_seg":  <detached>,
    "loss_det":  <detached>,
    "sigma_seg": <detached scalar>,   # surfaced per-iter via InformationWriter
    "sigma_det": <detached scalar>,
}
```

The `"fixed"` path omits `sigma_seg` / `sigma_det`.

**Eval mode** (always):

```python
{
    "seg_logits":  (B*N, num_seg_classes) float,
    "outputs":     <last-layer 3DETR box predictions dict>,
    "aux_outputs": <list of intermediate-layer predictions>,
}
```

## Evaluator + tester

- **`CombinedSegDetEvaluator`** (`pointcept/engines/hooks/evaluator.py`):
  per-`eval_epoch` hook. One pass over the val loader runs `model.train()`
  to accumulate `loss`, then `model.eval()` to gather AP + IoU. Logs
  `val/loss`, `val/AP25`, `val/AP50`, `val/mIoU`, `val/mAcc`, `val/allAcc`,
  per-class IoU/Acc, and (when `loss_weighting="uncertainty"`) `val/sigma_seg`,
  `val/sigma_det`. Reports AP50 as the best-checkpoint metric.
- **`CombinedSegDetTester`** (`pointcept/engines/test.py`): end-of-training
  tester with the same combined accumulators.

## Existing configs

| Config | Seg classes | Det classes | Notes |
|---|---|---|---|
| `configs/scannet/multitask-3detr-ptv3-v0m1-0-scannet.py` | 20 (NYU mapping via `DETECTION_NYU40_IDS_SEMSEG`) | 18 | ScanNet multi-task baseline. PT-v3m1 full U-Net + Vanilla Transformer det encoder + 3DETR decoder. Uses `ScanNetDetectionDataset(load_segment=True)`. |
| `configs/agco/multitask-3detr-ptv3-v0m1-0-agco.py` | 5 (background, tractor, harvester, trailer, car) | 3 (`included_classes=("tractor", "harvester", "trailer")`) | AGCO multi-task. Identity det encoder + 3DETR decoder. Uses `AgcoBBoxV1(load_segment=True, segment_subdir="segment")`. Hopper is detection-only (segmentation labels do not contain hopper points — they're labelled as `trailer` or `background`). The detection class set may be broadened to all 5 of the AGCO defaults if you want symmetric det+seg coverage; that's an independent knob. |

Both configs:

- Set `loss_weighting="uncertainty"` (default) with `c_seg=c_det=2.0`.
- Set `param_dicts=[dict(keyword="log_sigma_sq", weight_decay=0.0)]` so the
  σ params aren't decayed.
- Use `CombinedSegDetEvaluator` + `CombinedSegDetTester` in the hooks list /
  tester block.

### AGCO segment label layout

`<root>/<sensor>/segment/<ts>.npy` — uint16 array of length matching the raw
point cloud. Cast to int64 by `AgcoBBoxV1` and threaded through the transform
pipeline in lockstep with `point_cloud` via `_PER_POINT_KEYS` in
`det_transform.py`. Class index mapping:

| Index | Class |
|---|---|
| 0 | background |
| 1 | tractor |
| 2 | harvester |
| 3 | trailer |
| 4 | car |

Verified against 31/2277 sampled files from
`/home/dreez/Downloads/agco_data/agco2026/static_4/full_split/lslidar/segment/`
(observed unique values `[0, 1, 2, 3, 4]`).

## Adding a multi-task config from a base PTv3 segmentation config

Roughly: take `configs/scannet/semseg-pt-v3m1-0-base.py` (or your dataset's
equivalent), then:

1. Switch `model.type` to `MultiTask3DETRSegmentor`.
2. Move the existing `backbone` config under the new model's `backbone` key.
   Add `enc_mode=False` if not already set, and bump `in_channels=3` (XYZ-only
   input is what the detection dataset emits; if you have RGB available wire
   it through via the dataset's `use_color`-equivalent).
3. Move the seg head bits (`num_classes` → `num_seg_classes`,
   `backbone_out_channels`, `criteria` → `seg_criteria`).
4. Add the 3DETR detection branch (`det_encoder`, `det_decoder`,
   `det_dataset_config`, `det_criterion`, `encoder_dim`, `num_queries`, …).
5. Set `loss_weighting="uncertainty"` (or `"fixed"`) plus `c_seg=c_det=2.0`.
6. Add `param_dicts=[dict(keyword="log_sigma_sq", weight_decay=0.0)]`.
7. Switch the dataset to its detection variant with `load_segment=True`,
   include `GridSampleDetection` in the transform list (PTv3 needs one
   feature per voxel).
8. Replace the seg evaluator hook + tester with `CombinedSegDetEvaluator` and
   `CombinedSegDetTester`. Make sure the config defines
   `data.num_classes`, `data.ignore_index`, `data.names` (used by the seg
   half of the combined hook) and top-level `num_semcls` + `class_names`
   (used by the det half).

## Out of scope

- **Per-sub-loss σ** (one σ per giou / sem_cls / center / size / angle_cls /
  angle_reg / CE / Lovasz). The within-branch weights stay as the existing
  hand-tuned `loss_weight_dict` defaults — only the two top-level branch
  weights become learnable, matching what PattFormer reports.
- **More advanced multi-task strategies** (GradNorm, PCGrad, MGDA). Deferred
  until uncertainty weighting is exercised in real training and we have a
  baseline.
- **Removing the `loss_weighting="fixed"` path.** Kept for ablation and
  sanity checks against the prior fixed-weight baseline.
