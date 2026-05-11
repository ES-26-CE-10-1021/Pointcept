"""
3DETR on AGCO — v4m3-0-3cls: Scene-scaled center prediction, normal
train/val/test splits, num_queries=32.

Diff vs configs/agco/det-3detr-v3m2-1-3cls-agco.py:
  - center_offset_normalized=True   (scale the center MLP offset by
        scene_scale instead of treating it as raw metres; lifts the
        ±0.5 m per-query cap that breaks AGCO-scale scenes — see
        BoxProcessor.compute_predicted_center).
  - num_queries: 32                 (kept low; pair with the center fix
        to see if a sparse query set + reachable targets is sufficient).
  - clip_grad: 0.1 -> 1.0           (the previous tight clip was tuned
        for a regression head that could only move 50 cm/query; with
        the larger effective offset range, gradients are larger and
        the old clip strangles learning).
  - max_num_obj: 64 -> 16           (AGCO scenes carry ≤5 GT boxes; M=16
        keeps comfortable headroom but trims memory on the angle/cls
        head and GIoU matcher grid).

Usage:
    sh scripts/train.sh -d agco -c det-3detr-v4m3-0-3cls-agco -n v4m3 -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 4
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 1.0
gradient_accumulation_steps = 2

included_classes = ("tractor", "harvester", "trailer")
num_semcls = len(included_classes)
num_angle_bin = 12
max_num_obj = 16

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
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
    encoder=dict(
        type="IdentityEncoder3DETR",
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
        max_num_obj=max_num_obj,
    ),
    encoder_dim=512,
    decoder_dim=256,
    num_queries=32,
    position_embedding="fourier",
    mlp_dropout=0.3,
    projection_norm="ln",
    center_offset_normalized=True,
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
min_inliers = dict(lslidar=350, ouster=200, rslidar=80)

fixed_pc_dims = {
    "lslidar": dict(min=[ 0.0, -55.0, -11.0], max=[60.0, 55.0, 10.0]),
    "ouster":  dict(min=[ 0.0, -36.0, -10.0], max=[40.0, 36.0,  8.0]),
    "rslidar": dict(min=[ 0.0,  -8.5,  -8.0], max=[20.0,  8.5,  6.0]),
}

det_crop_transforms = [
    dict(
        type="FovCropDetection",
        azimuth_deg=(-60, 60),
        elevation_deg=None,
        crop_points=True,
        per_sensor={
            "lslidar": dict(azimuth_deg=(-60, 60)),
            "ouster":  dict(azimuth_deg=(-60, 60)),
            "rslidar": dict(azimuth_deg=(-60, 60)),
        },
    ),
    dict(
        type="SphericalCropDetection",
        max_dist=60.0,
        min_dist=1.0,
        per_sensor={
            "lslidar": dict(max_dist=60.0),
            "ouster":  dict(max_dist=40.0),
            "rslidar": dict(max_dist=20.0),
        },
    ),
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
        fixed_pc_dims=fixed_pc_dims,
        max_num_obj=max_num_obj,

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
        fixed_pc_dims=fixed_pc_dims,
        max_num_obj=max_num_obj,

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
        fixed_pc_dims=fixed_pc_dims,
        max_num_obj=max_num_obj,

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
