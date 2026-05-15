"""
3DETR on ScanNet — v0 + COLOR: Native 3DETR defaults with RGB input.

Differs from `det-3detr-v0m1-0-scannet.py` only by `use_color=True` and
`PointnetSAPreEncoder.mlp_dims[0]=3` (3 extra channels beyond XYZ).

Matches the native third_party/3detr training settings as closely as possible.
All criterion settings match native 3DETR defaults
(see third_party/3detr/scripts/scannet_ep1080.sh).
Key settings:
  - 720 total epochs with cosine+warmup LR schedule (not 90 + OneCycleLR)
  - batch_size=8 (4 per GPU on 2 GPUs, matching native's batchsize=8)
  - No AMP (native does not use mixed precision)

The native paper uses 8x V100 GPUs (total batch=64). Adjust batch_size and
num_worker when changing GPU count.

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-v0m1-0-scannet-color -n 3detr_v0_color -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8       # total across all GPUs (4 per GPU on 2 GPUs)
num_worker = 16      # total across all GPUs (8 per GPU on 2 GPUs)
mix_prob = 0         # detection dataset does not support MixUp
enable_amp = False   # native 3DETR does not use AMP
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
        mlp_dims=[3, 64, 128, 256],
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
    # ScanNet dataset metadata
    dataset_config=dict(type="ScanNetDetectionConfig"),
    encoder_dim=256,
    decoder_dim=256,
    num_queries=256,
    position_embedding="fourier",
    mlp_dropout=0.3,
    # Detection criterion — native 3DETR defaults
    criterion=dict(
        type="SetCriterion3DETR",
        matcher_cfg=dict(
            cost_class=1.0,
            cost_objectness=0.0,     # native default (disabled)
            cost_giou=2.0,           # native default
            cost_center=0.0,         # native default (disabled)
        ),
        loss_weight_dict=dict(
            loss_giou_weight=1.0,    # native default
            loss_sem_cls_weight=1.0,
            loss_no_object_weight=0.25,  # native default
            loss_angle_cls_weight=0.1,
            loss_angle_reg_weight=0.5,
            loss_center_weight=5.0,
            loss_size_weight=1.0,
        ),
        num_semcls=18,
        num_angle_bin=1,
    ),
)

# ── Schedule ──────────────────────────────────────────────────────────────────
# Native: 720 epochs, 9-epoch linear warmup (1e-6 → 5e-4), cosine decay to 1e-6.
# Approximated here with OneCycleLR:
#   Phase 1 (1.25%): lr ramps from 5e-4/500=1e-6 to 5e-4  (matches warm_lr → base_lr)
#   Phase 2 (98.75%): lr decays from 5e-4 to 5e-4/500=1e-6 (matches base_lr → final_lr)
epoch = 720
eval_epoch = 20

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[5e-4],
    pct_start=0.0125,            # NOTE: The closest to native lr is 9/720 = 1.25% warmup (native: warm_lr_epochs=9) (it was found though that 0.05 was better)
    anneal_strategy="cos",
    div_factor=500.0,          # initial_lr = 5e-4 / 500 = 1e-6 (native: warm_lr)
    final_div_factor=1.0,      # final_lr = 5e-4 / 500 = 1e-6 (native: final_lr)
    cycle_momentum=False,      # native 3DETR uses constant beta1=0.9
)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset_type = "ScanNetDetectionDataset"
data_root = "/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data"
meta_data_dir = "/home/andreas/3D-Perception/votenet/scannet/meta_data"

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
        use_color=True,
        use_height=False,
        transform=[
            dict(type="RandomCuboidDetection", min_points=30000),
            dict(type="RandomFlipDetection", p_x=0.5, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            dict(type="PointSubsampleDetection", num_points=40000),
        ],
    ),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=True,
        use_height=False,
        transform=[
            dict(type="PointSubsampleDetection", num_points=40000),
        ],
    ),
    test=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=True,
        use_height=False,
        transform=[
            dict(type="PointSubsampleDetection", num_points=40000),
        ],
    ),
)

# ── Hooks ─────────────────────────────────────────────────────────────────────
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ObjDetEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# ── Tester ────────────────────────────────────────────────────────────────────
test = dict(type="ObjDetTester", verbose=True)
