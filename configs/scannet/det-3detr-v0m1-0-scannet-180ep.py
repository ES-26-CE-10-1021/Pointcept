"""
3DETR on ScanNet — v0: Native 3DETR defaults (180 epochs)

Shortened version of det-3detr-v0m1-0-scannet.py for quick comparison
against native 3DETR. All hyperparams match native defaults except epoch count.

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-v0m1-0-scannet-180ep -n 3detr_180ep_pointcept -g 1
"""

_base_ = ["det-3detr-v0m1-0-scannet.py"]

# ── Schedule override ────────────────────────────────────────────────────────
# 180 epochs, 9-epoch warmup (pct_start = 9/180 = 0.05)
epoch = 180
eval_epoch = 10

# batch_size = 8 for 1 GPU (matching native batchsize_per_gpu=8)
batch_size = 8
num_worker = 16

seed = 0  # match native 3DETR --seed 0

scheduler = dict(
    type="OneCycleLR",
    max_lr=[5e-4],
    pct_start=0.05,            # 9/180 = 5% warmup
    anneal_strategy="cos",
    div_factor=500.0,          # initial_lr = 5e-4 / 500 = 1e-6
    final_div_factor=1.0,      # final_lr = 1e-6
    cycle_momentum=False,      # native 3DETR uses constant beta1=0.9
)
