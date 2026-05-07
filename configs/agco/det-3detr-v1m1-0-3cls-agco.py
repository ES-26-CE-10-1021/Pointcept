"""
3DETR on AGCO — v1m1-0-3cls: PointNet++ SA pre-encoder + vanilla Transformer + spherical
and ±60° FOV crops, with only 3 classes: tractor, harvester, trailer.

Identical to configs/agco/det-3detr-v1m1-0-agco.py except `included_classes` is reduced
to 3 (discards car and hopper). Boxes for excluded classes are dropped at dataset load time.

Usage:
    sh scripts/train.sh -d agco -c det-3detr-v1m1-0-3cls-agco -n 3detr_agco_v1_3cls -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# Subset of AGCO classes to train/eval on: only 3 classes (no car, no hopper).
# Order defines the model class indices (0..K-1).
included_classes = ("tractor", "harvester", "trailer")
num_semcls = len(included_classes)
num_angle_bin = 12

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    pre_encoder=dict(
        type="PointnetSAPreEncoder",
        npoint=2048,
        radius=0.2,
        nsample=64,
        mlp_dims=[0, 64, 128, 256],
        normalize_xyz=True,
    ),
    encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=256,
        nhead=4,
        nlayers=3,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
    ),
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
    encoder_dim=256,
    decoder_dim=256,
    num_queries=128,
    position_embedding="fourier",
    mlp_dropout=0.3,
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
            # dict(type="RandomScaleDetection", scale=(0.9, 1.1), apply_to_sizes=True),
            # dict(type="RandomJitterDetection", sigma=0.005, clip=0.02),
            # dict(type="RandomCuboidDetection", min_points=30000),
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
