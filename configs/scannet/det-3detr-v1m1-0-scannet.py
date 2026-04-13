"""
3DETR on ScanNet — 3D Object Detection (18 classes, axis-aligned boxes)

Vanilla 3DETR with PointNet++ pre-encoder + Transformer encoder/decoder.
Trained on ScanNet detection data (VoteNet-style pre-processed format).

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-v1m1-0-scannet -n my_3detr_exp -g 4

Data:
    Update `data_root` and `meta_data_dir` to point to your
    scannet_train_detection_data/ and meta_data/ directories.
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8      # total across all GPUs
num_worker = 16
mix_prob = 0         # detection dataset does not support MixUp
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    # PointNet++ pre-encoder: sub-samples to 2048 points
    pre_encoder=dict(
        type="PointnetSAPreEncoder",
        npoint=2048,
        radius=0.2,
        nsample=64,
        mlp_dims=[0, 64, 128, 256],   # XYZ only (no colour). PointnetSAModule adds 3 for XYZ automatically.
        normalize_xyz=True,
    ),
    # Vanilla Transformer encoder (no masking / downsampling)
    encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=256,
        nhead=4,
        nlayers=3,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
    ),
    # Cross-attention decoder for box queries
    decoder=dict(
        type="TransformerDecoder3DETR",
        decoder_dim=256,
        nhead=4,
        nlayers=8,
        ffn_dim=256,
        dropout=0.1,
    ),
    # ScanNet dataset metadata (class count, box parametrisation, etc.)
    dataset_config=dict(type="ScanNetDetectionConfig"),
    encoder_dim=256,
    decoder_dim=256,
    num_queries=256,
    position_embedding="fourier",
    mlp_dropout=0.3,
    # Detection criterion (Hungarian matching + weighted box losses)
    criterion=dict(
        type="SetCriterion3DETR",
        matcher_cfg=dict(
            cost_class=1.0,
            cost_objectness=0.0,     # native default (disabled)
            cost_giou=2.0,           # native default
            cost_center=0.0,         # native default (disabled)
        ),
        loss_weight_dict=dict(
            loss_giou_weight=1.0,
            loss_sem_cls_weight=1.0,
            loss_no_object_weight=0.25,
            loss_angle_cls_weight=0.1,
            loss_angle_reg_weight=0.5,
            loss_center_weight=5.0,
            loss_size_weight=1.0,
        ),
        num_semcls=18,
        num_angle_bin=1,       # ScanNet: axis-aligned boxes, no rotation bins
    ),
)

# ── Scheduler ─────────────────────────────────────────────────────────────────
epoch = 90
eval_epoch = 10      # evaluate every N epochs

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[5e-4],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset_type = "ScanNetDetectionDataset"
data_root = "/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data"
meta_data_dir = "/home/andreas/3D-Perception/votenet/scannet/meta_data"

# Semantic class names (18 detection categories)
num_semcls = 18
class_names = [
    "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter",
    "desk", "curtain", "refrigerator", "showercurtrain",
    "toilet", "sink", "bathtub", "garbagebin",
]

data = dict(
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="train",
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=True,
        random_cuboid_min_points=30000,
    ),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=False,
    ),
    test=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=False,
    ),
)

# ── Hooks ─────────────────────────────────────────────────────────────────────
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ObjDetEvaluator"),   # replaces SemSegEvaluator
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# ── Tester ────────────────────────────────────────────────────────────────────
test = dict(type="ObjDetTester", verbose=True)
