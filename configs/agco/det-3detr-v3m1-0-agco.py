"""
3DETR on AGCO — v3: PTv3 pre-encoder + Identity encoder + Transformer decoder.

Mirrors configs/scannet/det-3detr-v2m1-0-scannet.py (PTv3PreEncoder feeding an
IdentityEncoder3DETR, with the cross-attention TransformerDecoder3DETR for box
queries) adapted to AgcoBBoxV1:
  - 5 classes, oriented boxes via AgcoBBoxConfig (num_angle_bin=12).
  - num_queries=128 (fewer objects per scan than ScanNet indoor scenes).
  - PTv3 grid_size=0.05 — outdoor LiDAR is much sparser than ScanNet's indoor
    voxelization (0.02).

Augmentation pipeline matches configs/agco/det-3detr-v1m1-0-agco.py:
  - SphericalCropDetection (max 60 m) + FovCropDetection (±60° azimuth) on all
    splits, so train/val/test see the same observable wedge.
  - RandomFlipDetection(p_y=0.5) + RandomRotateZDetection(±5°) on train only.
  - PointSubsampleDetection enforces num_points as the final step.

Gravity alignment is enabled on all splits.

Usage:
    sh scripts/train.sh -d agco -c det-3detr-v3m1-0-agco -n 3detr_agco_v3 -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 4       # PTv3 + 100k pts is heavier than v1; halved vs scannet v2
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# Subset of AGCO classes to train/eval on. Boxes for any class not listed here
# are dropped at dataset load time, so the model never sees them as targets and
# any prediction that fires on them is penalised as background. Order defines
# the model class indices (0..K-1).
included_classes = ("tractor", "harvester", "trailer", "car", "hopper")
num_semcls = len(included_classes)
num_angle_bin = 12

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    # PTv3 Encoder (mirrors scannet v2m1-0; grid_size bumped for outdoor LiDAR)
    pre_encoder=dict(
        type="PTv3PreEncoder",
        grid_size=0.05,
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    # Identity mapping (placeholder for skipping the secondary encoder)
    encoder=dict(
        type="IdentityEncoder3DETR",
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
    dataset_config=dict(
        type="AgcoBBoxConfig",
        num_angle_bin=num_angle_bin,
        included_classes=included_classes,
    ),
    encoder_dim=512,    # must match enc_channels[-1]; encoder_to_decoder_projection handles 512->256
    decoder_dim=256,
    num_queries=128,
    position_embedding="fourier",
    mlp_dropout=0.3,
    projection_norm="ln",
    criterion=dict(
        type="SetCriterion3DETR",
        matcher_cfg=dict(
            cost_class=1.0,
            cost_objectness=0.0,
            cost_giou=2.0,
            cost_center=0.0,
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
        num_semcls=num_semcls,
        num_angle_bin=num_angle_bin,
    ),
)

# ── Schedule ──────────────────────────────────────────────────────────────────
epoch = 100
eval_epoch = 10

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[5e-4],
    pct_start=0.10,
    anneal_strategy="cos",
    div_factor=500.0,
    final_div_factor=1.0,
    cycle_momentum=False,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset_type = "AgcoBBoxV1"
data_root = "/mnt/data/pointcloud_datasets/Pointcept/agco2026"
meta_data_dir = "/mnt/data/pointcloud_datasets/Pointcept/agco2026/meta_data"
sensors = ["lslidar"]
num_points = 100_000

class_names = list(included_classes)
min_inliers = 350

# Shared deterministic crops (lslidar effective range + ±60° FOV wedge).
det_crop_transforms = [
    dict(type="SphericalCropDetection", max_dist=60.0, min_dist=0.0),
    dict(type="FovCropDetection", azimuth_deg=(-60.0, 60.0), crop_points=True),
]


data = dict(
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,

        split="train",
        split_prefix="agco",
        loop=1,

        sensors=sensors,
        use_intensity=False,

        num_points=num_points,
        min_inliers=min_inliers,
        included_classes=included_classes,

        apply_t_rtk=True,
        require_calibration=True,

        apply_r_global=True,

        apply_r_level_to_boxes=True,
        apply_r_level_to_points=True,
        require_gravity_align=True,
        residual_rpy_warn_deg=10.0,

        transform=[
            *det_crop_transforms,
            dict(type="RandomFlipDetection", p_x=0.0, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),

    val=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,

        split="val",
        split_prefix="agco",

        sensors=sensors,
        use_intensity=False,

        num_points=num_points,
        min_inliers=min_inliers,
        included_classes=included_classes,

        apply_t_rtk=True,
        require_calibration=True,

        apply_r_global=True,

        apply_r_level_to_boxes=True,
        apply_r_level_to_points=True,
        require_gravity_align=True,
        residual_rpy_warn_deg=10.0,

        transform=[
            *det_crop_transforms,
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),

    test=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,

        split="test",
        split_prefix="agco",

        sensors=sensors,
        use_intensity=False,

        num_points=num_points,
        min_inliers=min_inliers,
        included_classes=included_classes,

        apply_t_rtk=True,
        require_calibration=True,

        apply_r_global=True,

        apply_r_level_to_boxes=True,
        apply_r_level_to_points=True,
        require_gravity_align=True,
        residual_rpy_warn_deg=10.0,

        transform=[
            *det_crop_transforms,
            dict(type="PointSubsampleDetection", num_points=num_points),
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
test = dict(type="ObjDetTester", verbose=True, save_predictions=True)
