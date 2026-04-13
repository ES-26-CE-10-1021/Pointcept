# 3DETR Config Overview

All configs live in `configs/scannet/det-3detr-*.py` and train 3DETR for 18-class
axis-aligned object detection on ScanNet.

## Config summary

| Config | Pre-Encoder | Encoder | encoder_dim | projection_norm | Epochs | Purpose |
|--------|-------------|---------|-------------|-----------------|--------|---------|
| **v0m1-0** | PointNet++ SA (2048 pts) | VanillaTransformer (256d, 3L) | 256 | bn1d (default) | 720 | Native 3DETR baseline — matches `third_party/3detr/scripts/scannet_ep1080.sh` settings |
| **v0m1-0-180ep** | _(inherits v0)_ | _(inherits v0)_ | 256 | bn1d (default) | 180 | Shortened v0 for quick comparison |
| **v1m1-0** | PointNet++ SA (2048 pts) | VanillaTransformer (256d, 3L) | 256 | bn1d (default) | 90 | Pointcept-wrapped 3DETR with different LR schedule |
| **v2m1-0** | PTv3 (enc_channels→512) | IdentityEncoder | 512 | ln | 720 | PTv3 replaces PointNet++ SA; Identity skips transformer encoder |
| **v2m1-0-180ep** | PTv3 (enc_channels→512) | IdentityEncoder | 512 | ln | 180 | Shortened v2 for quick comparison |
| **v3m1-0** | PTv3 (enc_channels→512) | VanillaTransformer (512d, 3L) | 512 | ln | 720 | PTv3 + transformer encoder for additional feature refinement |
| **v3m1-1** | PTv3 (enc_channels→256) + FPS (2048 pts) | VanillaTransformer (256d, 3L) | 256 | ln | 720 | PTv3 + FPS to fixed-length output (no padding needed) |
| **v4m1-0** | PTv3UNet (dec_channels→64) | IdentityEncoder | 64 | ln | 720 | Full PTv3 U-Net with skip connections; high-res features |

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

### v4 — PTv3 U-Net
Full PTv3 U-Net (encoder + decoder with skip connections) producing high-resolution
features at the original voxel grid (~40K voxels at 0.02m). Uses `IdentityEncoder`
since the U-Net's multi-scale skip connections already provide cross-scale context.
Output dim is 64 (dec_channels[0]), projected to 256 for the decoder.

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
 PointNet++ SA ─┐
                │
 PTv3 enc-only ─┼──▶ [optional encoder: Identity | Vanilla] ──▶ 3DETR decoder
   (+ FPS?)     │
                │
 PTv3 U-Net ────┘

 Variable-length path (v2, v3m1-0, v4): padding_mask + LN projection
 Fixed-length path (v0, v1, v3m1-1):    no padding needed
```
