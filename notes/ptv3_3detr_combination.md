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

- **`PTv3m3PreEncoder`** (`PT-v3m3`, Utonia — frozen VFM, `enc_mode=True`)
  - Subclasses `PointTransformerV3` from
    `pointcept.models.point_transformer_v3.point_transformer_v3m3_utonia`
    (the Utonia variant of PTv3, with Point3DRoPE and a slightly different
    forward ordering) and is registered as `"PTv3m3PreEncoder"`.
  - Pulls the pretrained checkpoint from HuggingFace via
    `third_party.utonia.utonia.model.load(name=..., ckpt_only=True)`. The
    download is wrapped in a DDP-safe rank-0 + barrier pattern so the
    first-run HF fetch doesn't hammer the lock file from every rank.
  - The full backbone hyperparameter set (`in_channels`, `enc_depths`,
    `enc_channels`, `enc_num_head`, `rope_base`, …) is driven by the
    checkpoint config bundled with the download. The detection config
    only needs `pretrained`, `grid_size`, `enc_mode`, `freeze_backbone`
    (`"enc"` | `"enc_finetune"` | `"none"`), and optional
    `config_overrides`.
  - **Zero-padding hack.** The ScanNet detection dataset only provides
    xyz (and optionally rgb), while Utonia was pretrained on a 9-ch
    `[xyz, rgb, normal]` input. `_build_padded_feat` assembles a
    `(B, target_c, N)` tensor on-device with xyz, then rgb if available,
    then zero-filled channels for any remaining slots. `target_c` is read
    at runtime from `self.embedding.in_channels` so it tracks whatever
    the checkpoint was trained with. Causal Modality Blinding makes
    Utonia tolerate absent modalities as long as they're explicit zeros.
  - **Backbone training policy via `freeze_backbone`:**
    1. `"enc"` (default) freezes embedding + encoder, keeps decoder
       (if present) trainable.
    2. `"enc_finetune"` freezes everything except the last encoder stage.
    3. `"none"` keeps the full backbone trainable end-to-end.
  - Uses the same `(xyz, features)` ↔ `Point` bridge as the other
    adapters, so the rest of the 3DETR pipeline (padding mask through
    cross-attention, FPS query sampling, LN projection) remains
    unchanged.

### Utonia variant — `det-3detr-utonia-v1m1-0-scannet.py`

- Uses `PTv3m3PreEncoder(pretrained="utonia", grid_size=0.02,
  enc_mode=True, freeze_backbone="enc")` as the
  pre-encoder. The full Utonia hyperparameter set comes from the
  HuggingFace checkpoint; the config only pins what's policy.
- `encoder_dim = 576` (Utonia's deepest-stage output width) on both
  the `VanillaTransformerEncoder3DETR` and the root
  `encoder_to_decoder_projection`. `decoder_dim = 256` unchanged.
- Keeps `projection_norm="ln"` (the PT-v3m3 output is still
  variable-length per scene).
- The `CheckpointLoader` hook is left bare — Utonia weights are
  fetched inside the pre-encoder, not via the loader's
  `keywords`/`replacement` remap.
- First-run HF download note: under DDP the rank-0 + barrier dance can
  still time out if the 137M-parameter file has to download over a slow
  link. For production training, warm the cache once before launching
  with ``python -c "from third_party.utonia.utonia.model import load;
  load(name='utonia', ckpt_only=True)"``.
- For partial/full finetuning ablations, switch
  `freeze_backbone="enc_finetune"` (last encoder stage trainable) or
  `freeze_backbone="none"` (full backbone trainable).

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
- `tests/test_utonia_3detr_integration.py` — parallel suite for
  `PTv3m3PreEncoder`: forward shape, the zero-padding helper, frozen
  `requires_grad`/`eval` invariants, the no_grad detach + re-enable
  bridge, a peak-VRAM regression guard, and an end-to-end tiny
  `Model3DETRDetector` forward+backward. A separate `test_utonia_load_hf`
  exercises the real HuggingFace download path when
  `RUN_UTONIA_HF_TEST=1` is set.
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
3-layer `VanillaTransformerEncoder3DETR` → 3DETR decoder — and were
created to isolate projection-normalization behavior on variable-length
inputs. All PTv3-based configs now use `projection_norm="ln"` (LayerNorm)
which is padding-safe since it computes per-sample statistics. The
`bn1d_masked` option was removed because zeroing padded positions does not
actually prevent them from affecting BN mean/variance.

### PTv3 encoder + FPS downsampling (v3m1-1)

| Config                        | Notes                                                                                                                                                                                                                                                                                                                       |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det-3detr-v3m1-1-scannet.py` | `PTv3PreEncoder(npoint=2048)` — PTv3 encoder followed by FPS to a fixed 2048 points per scene. No padding mask is needed downstream. `enc_channels[-1]=256` is chosen to match `decoder_dim`, eliminating the dimension mismatch in encoder_to_decoder_projection. |

### PTv3 U-Net → IdentityEncoder (v4)

| Config                        | Notes                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `det-3detr-v4m1-0-scannet.py` | `PTv3UNetPreEncoder` — full PTv3 U-Net (encoder + decoder) producing dense features at the initial voxel grid. Handed directly to the 3DETR decoder via `IdentityEncoder3DETR`; the rationale is that the U-Net's multi-scale skip connections already provide the cross-scale context that a transformer encoder would otherwise add. |

### Frozen Utonia pre-encoder (utonia-v1)

| Config                                   | Notes                                                                                                                                                                                                                                                                                            |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `det-3detr-utonia-v1m1-0-scannet.py`     | `PTv3m3PreEncoder(pretrained="utonia")` — pretrained Utonia VFM (PT-v3m3) run as a frozen feature extractor, followed by a 3-layer `VanillaTransformerEncoder3DETR` at `encoder_dim=576`. Utonia weights never receive gradients. Missing modalities (normals, optionally rgb) are zero-padded on-device. |

## Quick mental map

```
 PointNet++ SA      ─┐
                     │
 PTv3 enc-only       ─┤
    (+ FPS?)         │
                     ├─▶ [optional encoder: Identity | Vanilla | Masked] ─▶ 3DETR decoder
 PTv3 U-Net         ─┤
                     │
 Utonia (frozen m3) ─┘

 Variable-length path uses padding_mask + LN projection.
 Fixed-length path (PointNet++ SA, PTv3 + FPS) also uses LN for consistency.
 The Utonia path is variable-length and uses the same LN projection.
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
