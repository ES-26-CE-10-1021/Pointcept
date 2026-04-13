# 3DETR Config Overview

All configs live in `configs/scannet/det-3detr-*.py` and train 3DETR for 18-class
axis-aligned object detection on ScanNet.

## Config summary

| Config           | Pre-Encoder                     | pre_enc out dim | Encoder                 | encoder_dim | decoder_dim | Epochs | Purpose                            |
| ---------------- | ------------------------------- | --------------- | ----------------------- | ----------- | ----------- | ------ | ---------------------------------- |
| **v0m1-0**       | PointNet++ SA (2048 pts)        | 256             | VanillaTransformer (3L) | 256         | 256         | 720    | Native 3DETR baseline              |
| **v0m1-0-180ep** | _(inherits v0)_                 | 256             | _(inherits v0)_         | 256         | 256         | 180    | Shortened v0 for quick comparison  |
| **v1m1-0**       | PointNet++ SA (2048 pts)        | 256             | VanillaTransformer (3L) | 256         | 256         | 90     | Pointcept integration sanity check |
| **v2m1-0**       | PTv3 (enc→512)                  | 512             | IdentityEncoder         | 512         | 256         | 720    | PTv3; no transformer encoder       |
| **v2m1-0-180ep** | PTv3 (enc→512)                  | 512             | IdentityEncoder         | 512         | 256         | 180    | Shortened v2 for quick comparison  |
| **v3m1-0**       | PTv3 (enc→512)                  | 512             | VanillaTransformer (3L) | 512         | 256         | 720    | PTv3 + transformer encoder         |
| **v3m1-1**       | PTv3 (enc→256) + FPS (2048 pts) | 256             | VanillaTransformer (3L) | 256         | 256         | 720    | PTv3 + FPS to fixed-length output  |
| **v4m1-0**       | PTv3UNet (dec→64)               | 64              | IdentityEncoder         | 64          | 256         | 720    | PTv3 U-Net; no transformer encoder |
| **v4m1-1**       | PTv3UNet (dec→64)               | 64              | VanillaTransformer (3L) | 64          | 256         | 720    | PTv3 U-Net + transformer encoder   |
| **utonia-v1m1-0**| Frozen Utonia PT-v3m3 (enc→576) | 576             | VanillaTransformer (3L) | 576         | 256         | 720    | Frozen pretrained Utonia VFM       |

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
savings and forced eval mode for feature determinism. Same 3-layer vanilla
transformer encoder as v3m1-0, but widened to `encoder_dim=576` to match
Utonia's deepest-stage output.

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

- **Variable-length path** (v2, v3m1-0, v4m1-0, v4m1-1, utonia-v1m1-0): padding_mask + LN projection
- **Fixed-length path** (v0, v1, v3m1-1): no padding needed
