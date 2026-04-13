# Extending 3DETR with Segmentation Heads

## Current Architecture

3DETR's decoder outputs per-query (per-box) features:
```
box_features: (nlayers, nqueries, B, C)  — e.g. (8, 256, B, 256)
```

Each query predicts box attributes via `mlp_heads`:
- `sem_cls_head` — class logits (18 + 1 background)
- `center_head` — box center offset
- `size_head` — box dimensions
- `angle_cls_head` / `angle_residual_head` — rotation

## Where to Add New Heads

### Per-box attributes → add to `mlp_heads`
If you want to predict something *about each detected object* (e.g. a material
label, an articulation state), just add another MLP entry to `mlp_heads` in
`Model3DETRDetector._build_mlp_heads()`. These operate on `box_features` which
are per-query.

### Semantic segmentation → separate encoder-side branch
Semantic segmentation needs **per-point** predictions (40,000 points), not
per-query (256 queries). The encoder features are per-point, so the seg head
branches off before the decoder:

```
PTv3 encoder → enc_features (N, B, C)  [per-point]
                ├── 3DETR decoder → box_features → box predictions
                └── seg head (MLP) → (N, B, num_classes)  [per-point class logits]
```

The seg head would be a simple MLP on `enc_features` (available after
`run_encoder()` in `forward()`). Needs its own loss (e.g. cross-entropy with
per-point `sem_label`), added to the total loss alongside detection losses.

### Instance segmentation → query × point interaction
Instance seg needs to assign each point to a detected instance. This requires
combining per-query features (which object) with per-point features (which
point). Common approach: a **mask head** that dot-products or cross-attends
query features against encoder point features:

```
enc_features:  (N, B, C)          ← per-point
box_features:  (nqueries, B, C)   ← per-object
mask_logits = einsum('qbc,nbc->bqn', box_features, enc_features)  → (B, nqueries, N)
```

Each query then predicts: class + box + binary point mask for its instance.
This is the approach used by Mask3D and similar methods.

### Combined (detection + semantic seg + instance seg)

```
PTv3 encoder → enc_features (per-point)
                ├── 3DETR decoder → box_features (per-query)
                │    ├── mlp_heads → box class, center, size, angle
                │    └── mask_head → einsum(queries, points) → instance masks
                └── seg_head (MLP) → per-point semantic class logits
```

## Implementation Notes

- `enc_features` is available in `Model3DETRDetector.forward()` after
  `run_encoder()` — shape `(N', B, C)` in transformer convention
- With PTv3 encoder, `enc_features` may have a `padding_mask` (variable-length
  scenes). Any per-point head must respect this mask in its loss computation
- The seg head loss would need per-point semantic labels, which the current
  `ScanNetDetectionDataset` already loads (`sem_label.npy`) but only uses for
  the output dict's `sem_seg_labels` field — not currently consumed by the model
- For instance masks, the Hungarian matching in `SetCriterion3DETR` would need
  to be extended to include a mask loss term (e.g. binary CE or dice loss)
