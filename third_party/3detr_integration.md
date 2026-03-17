# 3DETR Integration into Pointcept

## Overview

3DETR (A Transformer-based 3D Object Detection model) has been integrated into the Pointcept
framework. It can be trained and tested on ScanNet for bounding box detection using Pointcept's
standard config system, with full support for swapping encoders (e.g. PointTransformerV3).

---

## New Files

| File | Purpose |
|------|---------|
| `pointcept/models/detection_3detr/dataset_config.py` | `ScanNetDetectionConfig` — class mappings, box parametrisation |
| `pointcept/models/detection_3detr/components.py` | 4 registered MODULES: `PointnetSAPreEncoder`, `VanillaTransformerEncoder3DETR`, `MaskedTransformerEncoder3DETR`, `TransformerDecoder3DETR` |
| `pointcept/models/detection_3detr/criterion.py` | `SetCriterion3DETR` — Hungarian matching + box losses |
| `pointcept/models/detection_3detr/model.py` | `Model3DETRDetector` — main model with optional `pre_encoder` |
| `pointcept/datasets/scannet_detection.py` | `ScanNetDetectionDataset` — reads VoteNet-style detection data |
| `configs/scannet/det-3detr-v1m1-0-scannet.py` | Full training config |

## Modified Files

- `pointcept/models/__init__.py` — added 3DETR import
- `pointcept/datasets/__init__.py` — added dataset import
- `pointcept/engines/test.py` — added `ObjDetTester` (AP25/AP50)
- `pointcept/engines/hooks/evaluator.py` — added `ObjDetEvaluator` hook

---

## Key Design Points

- **Modularity**: each of `pre_encoder`, `encoder`, `decoder` is an independently registered
  module. To use PTv3 later, set `pre_encoder=None` and provide a PTv3 adapter as the encoder.
- **`mlp_dims`**: first element is extra features *beyond XYZ* — use `[0, 64, 128, 256]` for
  XYZ-only, `[3, 64, 128, 256]` for RGB.
- **`input_feature_dim`**: when `pre_encoder=None`, set this to the number of extra feature
  channels beyond XYZ (e.g. `0` for XYZ-only, `3` for RGB) so an input projection is built.
- **Data**: update `data_root` and `meta_data_dir` in the config to point to your existing
  VoteNet-style detection data directories.
- **Training**: `sh scripts/train.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr -g 4`

---

## Running Tests

The commands below run progressively deeper checks. All require the `pointcept` conda environment
and a CUDA GPU (FPS sampling is CUDA-only).

```bash
cd /home/andreas/3D-Perception/Pointcept
conda activate pointcept
```

### Test 1 — Full import (no regressions)
```bash
python -c "import pointcept; import pointcept.models; import pointcept.datasets; import pointcept.engines; print('OK')"
```

### Test 2 — Registry entries
```bash
python -c "
from pointcept.models.builder import MODELS, MODULES
from pointcept.models.losses.builder import LOSSES
from pointcept.datasets.builder import DATASETS
from pointcept.engines.test import TESTERS
from pointcept.engines.hooks.builder import HOOKS
for reg, key in [
    (MODELS,   'Model3DETRDetector'),
    (MODULES,  'PointnetSAPreEncoder'),
    (MODULES,  'VanillaTransformerEncoder3DETR'),
    (MODULES,  'MaskedTransformerEncoder3DETR'),
    (MODULES,  'TransformerDecoder3DETR'),
    (MODULES,  'ScanNetDetectionConfig'),
    (LOSSES,   'SetCriterion3DETR'),
    (DATASETS, 'ScanNetDetectionDataset'),
    (TESTERS,  'ObjDetTester'),
    (HOOKS,    'ObjDetEvaluator'),
]:
    assert key in reg._module_dict, f'MISSING: {key}'
print('All 10 registry entries present')
"
```

### Test 3 — Config validation
```bash
python -c "
from pointcept.utils.config import Config
cfg = Config.fromfile('configs/scannet/det-3detr-v1m1-0-scannet.py')
assert cfg.model.type == 'Model3DETRDetector'
assert cfg.data.train.type == 'ScanNetDetectionDataset'
assert cfg.test.type == 'ObjDetTester'
print('Config OK:', cfg.model.type)
"
```

### Test 4 — Dataset loading (real data)
```bash
python -c "
from pointcept.datasets import ScanNetDetectionDataset
ds = ScanNetDetectionDataset(
    root_dir='/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data',
    meta_data_dir='/home/andreas/3D-Perception/votenet/scannet/meta_data',
    split='val',
)
print(f'{len(ds)} val scenes')
item = ds[0]
print('point_clouds:', item['point_clouds'].shape)   # (40000, 3)
print('gt_box_corners:', item['gt_box_corners'].shape)  # (64, 8, 3)
print('gt_box_present sum:', item['gt_box_present'].sum())
"
```

### Test 5 — Model forward pass (CUDA required)
```bash
python -c "
import torch
from pointcept.models.detection_3detr import Model3DETRDetector

model = Model3DETRDetector(
    pre_encoder=dict(type='PointnetSAPreEncoder', npoint=512, radius=0.2, nsample=32,
                     mlp_dims=[0, 64, 128, 256], normalize_xyz=True),
    encoder=dict(type='VanillaTransformerEncoder3DETR', encoder_dim=256, nhead=4,
                 nlayers=3, ffn_dim=128, dropout=0.1),
    decoder=dict(type='TransformerDecoder3DETR', decoder_dim=256, nhead=4,
                 nlayers=8, ffn_dim=256, dropout=0.1),
    dataset_config=dict(type='ScanNetDetectionConfig'),
    encoder_dim=256, decoder_dim=256, num_queries=256,
).eval().cuda()

B, N = 2, 4096
inp = {
    'point_clouds': torch.randn(B, N, 3).cuda(),
    'point_cloud_dims_min': torch.zeros(B, 3).cuda(),
    'point_cloud_dims_max': torch.ones(B, 3).cuda(),
}
with torch.no_grad():
    out = model(inp)
print('box_corners:', out['outputs']['box_corners'].shape)  # (2, 256, 8, 3)
print('sem_cls_logits:', out['outputs']['sem_cls_logits'].shape)  # (2, 256, 19)
print('aux layers:', len(out['aux_outputs']))  # 7
"
```

### Test 6 — Training step with criterion
```bash
python -c "
import torch
from pointcept.models.detection_3detr import Model3DETRDetector

model = Model3DETRDetector(
    pre_encoder=dict(type='PointnetSAPreEncoder', npoint=512, radius=0.2, nsample=32,
                     mlp_dims=[0, 64, 128, 256], normalize_xyz=True),
    encoder=dict(type='VanillaTransformerEncoder3DETR', encoder_dim=256, nhead=4,
                 nlayers=2, ffn_dim=128, dropout=0.1),
    decoder=dict(type='TransformerDecoder3DETR', decoder_dim=256, nhead=4,
                 nlayers=2, ffn_dim=256, dropout=0.1),
    dataset_config=dict(type='ScanNetDetectionConfig'),
    encoder_dim=256, decoder_dim=256, num_queries=64,
    criterion=dict(
        type='SetCriterion3DETR',
        matcher_cfg=dict(cost_class=1.0, cost_objectness=0.1, cost_giou=1.0, cost_center=5.0),
        loss_weight_dict=dict(
            loss_giou_weight=1.0, loss_sem_cls_weight=1.0, loss_no_object_weight=0.25,
            loss_angle_cls_weight=0.1, loss_angle_reg_weight=0.5,
            loss_center_weight=5.0, loss_size_weight=1.0,
        ),
        num_semcls=18, num_angle_bin=1,
    ),
).train().cuda()

B, N, nq = 2, 1024, 64
gt_present = torch.zeros(B, 64).cuda(); gt_present[:, :5] = 1.0
inp = {
    'point_clouds': torch.randn(B, N, 3).cuda(),
    'point_cloud_dims_min': torch.zeros(B, 3).cuda(),
    'point_cloud_dims_max': torch.ones(B, 3).cuda(),
    'gt_box_corners': torch.rand(B, 64, 8, 3).cuda(),
    'gt_box_centers': torch.rand(B, 64, 3).cuda(),
    'gt_box_centers_normalized': torch.rand(B, 64, 3).cuda(),
    'gt_box_sizes': torch.rand(B, 64, 3).cuda(),
    'gt_box_sizes_normalized': torch.rand(B, 64, 3).cuda(),
    'gt_box_angles': torch.zeros(B, 64).cuda(),
    'gt_angle_class_label': torch.zeros(B, 64, dtype=torch.long).cuda(),
    'gt_angle_residual_label': torch.zeros(B, 64).cuda(),
    'gt_box_sem_cls_label': torch.randint(0, 18, (B, 64)).cuda(),
    'gt_box_present': gt_present,
}
out = model(inp)
out['loss'].backward()
print(f'loss = {out[\"loss\"].item():.4f}')
print('loss_dict keys:', list(out['loss_dict'].keys())[:5])
"
```

### Test 7 — pre_encoder=None (for future PTv3 integration)
```bash
python -c "
import torch
from pointcept.models.detection_3detr import Model3DETRDetector

model = Model3DETRDetector(
    pre_encoder=None,         # skip PointNet++ — encoder gets raw point cloud
    input_feature_dim=0,      # XYZ only; set to 3 for RGB
    encoder=dict(type='VanillaTransformerEncoder3DETR', encoder_dim=256, nhead=4,
                 nlayers=3, ffn_dim=128, dropout=0.1),
    decoder=dict(type='TransformerDecoder3DETR', decoder_dim=256, nhead=4,
                 nlayers=8, ffn_dim=256, dropout=0.1),
    dataset_config=dict(type='ScanNetDetectionConfig'),
    encoder_dim=256, decoder_dim=256, num_queries=256,
).eval().cuda()

B, N = 2, 1024
inp = {
    'point_clouds': torch.randn(B, N, 3).cuda(),
    'point_cloud_dims_min': torch.zeros(B, 3).cuda(),
    'point_cloud_dims_max': torch.ones(B, 3).cuda(),
}
with torch.no_grad():
    out = model(inp)
print('box_corners:', out['outputs']['box_corners'].shape)  # (2, 256, 8, 3)
"
```

### Test 8 — End-to-end with real ScanNet data
```bash
python -c "
import torch
from torch.utils.data import DataLoader
from pointcept.datasets import ScanNetDetectionDataset
from pointcept.models.detection_3detr import Model3DETRDetector

ds = ScanNetDetectionDataset(
    root_dir='/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data',
    meta_data_dir='/home/andreas/3D-Perception/votenet/scannet/meta_data',
    split='val', num_points=20000, use_color=False, augment=False,
)
loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2)
batch = next(iter(loader))
batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

model = Model3DETRDetector(
    pre_encoder=dict(type='PointnetSAPreEncoder', npoint=1024, radius=0.2, nsample=32,
                     mlp_dims=[0, 64, 128, 256], normalize_xyz=True),
    encoder=dict(type='VanillaTransformerEncoder3DETR', encoder_dim=256, nhead=4,
                 nlayers=3, ffn_dim=128, dropout=0.0),
    decoder=dict(type='TransformerDecoder3DETR', decoder_dim=256, nhead=4,
                 nlayers=8, ffn_dim=256, dropout=0.0),
    dataset_config=dict(type='ScanNetDetectionConfig'),
    encoder_dim=256, decoder_dim=256, num_queries=256,
).eval().cuda()

with torch.no_grad():
    out = model(batch)
print('Eval forward OK, box_corners:', out['outputs']['box_corners'].shape)
"
```

---

## Training

Update `data_root` and `meta_data_dir` in the config, then:

```bash
# 4-GPU training (recommended)
sh scripts/train.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr_exp -g 4

# Single GPU (for debugging)
sh scripts/train.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr_debug -g 1

# Resume
sh scripts/train.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr_exp -g 4 -r true
```

## Evaluation

```bash
sh scripts/test.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr_exp -w model_best -g 4
```

This logs `AP25` and `AP50` mAP to the experiment log file.

---

## Future: Swapping in PointTransformerV3

To use PTv3 as the encoder instead of PointNet++ + vanilla transformer:

1. Set `pre_encoder=None` and `input_feature_dim=<C>` in the config (or write a PTv3 adapter
   that produces `(xyz, features, inds)` and register it as a MODULE).
2. Set `encoder` to your PTv3 adapter module.
3. The decoder, criterion, and dataset need no changes.
