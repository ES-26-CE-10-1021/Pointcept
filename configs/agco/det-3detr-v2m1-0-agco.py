"""
3DETR on AGCO — v2: Utonia (PT-v3m3) pre-encoder with last-stage fine-tune.

Mirrors configs/scannet/det-3detr-utonia-v5m1-0-scannet.py adapted to AgcoBBoxV1
(oriented boxes, 5 classes, num_angle_bin=12, lslidar single-sensor) with the
same spherical + ±60° FOV crops as configs/agco/det-3detr-v1m1-0-agco.py.

Key choices:
  - ``pre_encoder.pretrained = "utonia"``  (HF checkpoint auto-loaded by
    PTv3m3PreEncoder; ``in_channels=9`` is locked to the checkpoint).
  - ``utonia_preprocess=True`` on AgcoBBoxV1 right-pads point_clouds from
    XYZ (3 ch) to 9 ch with zeros for the missing rgb / normal channels.
  - ``freeze_backbone="enc_finetune"``: embedding + early encoder stages
    frozen; the last encoder stage trains end-to-end with the 3DETR head.
  - Gravity alignment enabled on all splits.

Usage:
    sh scripts/train.sh -d agco -c det-3detr-v2m1-0-agco -n 3detr_agco_v2_utonia -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 2
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1
gradient_accumulation_steps = 4

# Subset of AGCO classes to train/eval on. Boxes for any class not listed here
# are dropped at dataset load time, so the model never sees them as targets and
# any prediction that fires on them is penalised as background. Order defines
# the model class indices (0..K-1).
included_classes = ("tractor", "harvester", "trailer", "car", "hopper")
num_semcls = len(included_classes)
num_angle_bin = 12

# Utonia's deepest-stage output width (PT-v3m3 enc_channels[-1]).
UTONIA_ENC_DIM = 576

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    pre_encoder=dict(
        type="PTv3m3PreEncoder",
        pretrained="utonia",
        grid_size=0.05,  # outdoor LiDAR — coarser than ScanNet's 0.01
        enc_mode=True,
        freeze_backbone="enc_finetune",
    ),
    encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=UTONIA_ENC_DIM,
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
    encoder_dim=UTONIA_ENC_DIM,
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
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=500.0,
    final_div_factor=1.0,
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
        utonia_preprocess=True,

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
        utonia_preprocess=True,

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
        utonia_preprocess=True,

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
