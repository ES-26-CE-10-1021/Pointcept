# PTv3 + 3DETR Combination (feature/ptv3-3detr-combination)

Summary of the additions on `feature/ptv3-3detr-combination` that combine
PointTransformerV3 with 3DETR for 3D object detection on ScanNet. This
document covers the code additions, the different ways the two models have
been combined, and the purpose of each config variant under
`configs/scannet/det-3detr-*`.

## Why this branch exists

The base 3DETR pipeline (`pointcept/models/detection_3detr/`) was originally
wired up with a PointNet++ set-abstraction pre-encoder. This branch replaces
(or augments) that pre-encoder with PointTransformerV3 and explores several
ways of bridging the two:

- **Encoder-only PTv3** as a coarse, variable-length pre-encoder (with or
  without FPS downsampling to a fixed length).
- **Full PTv3 U-Net** (encoder + decoder) producing high-resolution features
  that are handed straight to the 3DETR decoder.
- **Projection-normalization alternatives** for the
  `encoder_to_decoder_projection` MLP to avoid BatchNorm-on-padding
  corruption in variable-length scenes.

## Code additions

### `pointcept/models/detection_3detr/ptv3.py`

New file. Two pre-encoder classes wrap `PointTransformerV3` and bridge the
dense-tensor interface 3DETR expects `(xyz, features)` with PTv3's
`Point`-based interface.

- **`PTv3PreEncoder`** (`enc_mode=True`, encoder-only)
  - Coarse output at PTv3's deepest voxel resolution.
  - `npoint=None` (default): returns the `Point` directly; downstream
    `point2dense()` pads to max scene length and threads a `padding_mask`
    through the 3DETR decoder's cross-attention.
  - `npoint=K`: after encoding, `point2dense()` is called and FPS selects K
    well-distributed points per scene, producing a fixed-length
    `(xyz, features, inds)` tuple that matches `PointnetSAPreEncoder`'s
    interface. Padded positions are pushed to `1e6` so FPS ignores them.

- **`PTv3UNetPreEncoder`** (`enc_mode=False`, full U-Net)
  - Runs the full PTv3 encoder and decoder.
  - Returns the decoded `Point` at the initial voxel resolution (~40 K
    voxels at `grid_size=0.02`), giving the 3DETR decoder high-resolution
    features with multi-scale context from skip connections.

Both classes are registered with `MODULES` under the names
`"PTv3PreEncoder"` and `"PTv3UNetPreEncoder"` respectively.

### `pointcept/models/detection_3detr/model.py`

- **Variable-length handling.** `run_encoder()` accepts variable-length
  outputs from PTv3 (via `point2dense()`), and the returned `padding_mask`
  is threaded through the rest of the pipeline so the 3DETR decoder's
  cross-attention ignores padded slots. The FPS query sampler masks padded
  positions the same way (pushing them to `1e6`).

- **`projection_norm` option.** The `encoder_to_decoder_projection`
  `GenericMLP` used to be hard-wired to `BatchNorm1d`, which can silently
  break when the batch contains zero-padded positions (BN treats the padding
  as real points and corrupts its running statistics). The current
  `Model3DETRDetector` implementation supports:
  - `"bn1d"` — original 3DETR default. Appropriate when the input is
    fixed-length (PointNet++ or FPS variants).
  - `"ln"` — `LayerNorm`. Padding-safe because normalization is applied
    per sample rather than across padded point slots.
    In this branch's configs, the variable-length PTv3 variants use `"ln"`
    rather than a masked-BN mode.

### Tests

- `tests/test_3detr_point_integration.py` — `Point` ↔ dense conversion,
  including the padding-mask and gradient-flow edge cases.
- `tests/test_ptv3_3detr_integration.py` — end-to-end forward/backward of
  `PTv3PreEncoder` wired into `Model3DETRDetector`.
- `tests/test_3detr_cross_validation.py` — audit of how the Pointcept
  config differs from the native 3DETR defaults.
- `tests/test_lr_schedule_comparison.py` — sanity check that the
  Pointcept-side LR schedule matches native's cosine + warmup.

## Configs

All configs live in `configs/scannet/`. The `v{N}m1-{M}` suffix convention
is: `v{N}` = architecture variant, `m1` = model revision, `{M}` = minor
experiment iteration.

### Baselines

| Config                              | Pre-encoder   | Encoder            | Notes                                                                                                                                                                             |
| ----------------------------------- | ------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det-3detr-v0m1-0-scannet.py`       | PointNet++ SA | MaskedTransformer  | Native 3DETR defaults: 720 epochs, cosine+warmup, no AMP, `loss_giou_weight=0`, `loss_no_object_weight=0.2`, native matcher costs. Reference point for the cross-validation test. |
| `det-3detr-v0m1-0-scannet-180ep.py` | PointNet++ SA | MaskedTransformer  | Shortened `v0` (180 epochs, 9-epoch warmup) for quick comparisons on 1 GPU.                                                                                                       |
| `det-3detr-v1m1-0-scannet.py`       | PointNet++ SA | VanillaTransformer | Vanilla 3DETR under the Pointcept-side training schedule (90 epochs, OneCycleLR). The baseline the PTv3 variants are compared against.                                            |

### PTv3 encoder-only → IdentityEncoder

| Config                              | Notes                                                                                                                                                                             |
| ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det-3detr-v2m1-0-scannet.py`       | PTv3 encoder-only (`PTv3PreEncoder`, `npoint=None`) feeding `IdentityEncoder3DETR` directly into the 3DETR decoder. Variable-length, padding-mask routed through cross-attention. |
| `det-3detr-v2m1-0-scannet-180ep.py` | Same architecture as `v2m1-0`, retuned for a 180-epoch schedule with larger batch/workers for single-GPU runs.                                                                    |

### PTv3 encoder-only → Vanilla Transformer encoder (v3 family)

These three configs share the same architecture — PTv3 encoder-only →
3-layer `VanillaTransformerEncoder3DETR` → 3DETR decoder — and differ only
in how the `encoder_to_decoder_projection` MLP is normalized. They were
created specifically to isolate the BN-on-padding problem.

| Config                        | `projection_norm` | Purpose                                                                                            |
| ----------------------------- | ----------------- | -------------------------------------------------------------------------------------------------- |
| `det-3detr-v3m1-0-scannet.py` | `bn1d` (default)  | Baseline. Known to be affected by padded-position corruption when scenes have variable length.     |
| `det-3detr-v3m1-1-scannet.py` | `ln`              | LayerNorm fix — per-sample, so padding cannot corrupt statistics.                                  |
| `det-3detr-v3m1-2-scannet.py` | `bn1d_masked`     | Keeps BN but zeros padded positions before/after the projection so they never enter BN statistics. |

### PTv3 encoder + FPS downsampling (v4)

| Config                        | Notes                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det-3detr-v4m1-0-scannet.py` | `PTv3PreEncoder(npoint=2048)` — PTv3 encoder followed by FPS to a fixed 2048 points per scene. No padding mask is needed downstream, so the default `bn1d` projection is safe. `enc_channels[-1]=256` is chosen to match `decoder_dim`, eliminating the dimension mismatch that previously went through the projection MLP. |

### PTv3 U-Net → IdentityEncoder (v5)

| Config                        | Notes                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det-3detr-v5m1-0-scannet.py` | `PTv3UNetPreEncoder` — full PTv3 U-Net (encoder + decoder) producing dense features at the initial voxel grid. Handed directly to the 3DETR decoder via `IdentityEncoder3DETR`; the rationale is that the U-Net's multi-scale skip connections already provide the cross-scale context that a transformer encoder would otherwise add. |

## Quick mental map

```
 PointNet++ SA ─┐
                │
 PTv3 enc-only ─┼─▶ [optional encoder: Identity | Vanilla | Masked] ─▶ 3DETR decoder
    (+ FPS?)    │
                │
 PTv3 U-Net ────┘

 Variable-length path uses padding_mask + (LN | masked BN) projection.
 Fixed-length path (PointNet++ SA, PTv3 + FPS) uses default BN projection.
```

## Status at time of writing (2026-04-10)

- All variants load, train, and test successfully; unit tests under
  `tests/test_ptv3_3detr_integration.py` and
  `tests/test_3detr_point_integration.py` pass.
- The branch is ahead of `dev` by the PTv3-combination work plus the
  recently merged PTv3 U-Net addition from `feature/ptv3-integration-v2`
  (merge commit on 2026-04-10).
- Next planned direction: using a frozen pre-trained Utonia backbone as
  another pre-encoder variant, mirroring the PTv3 integration pattern
  established here.
