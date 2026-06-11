"""
Seg-only PTv3 on AGCO — v3m4 ouster variant.

Mirrors the segmentation branch of ``multitask-3detr-ptv3-v3m4-0-3cls-agco-ouster.py``
without the detection branch. Same backbone (PT-v3m1 full U-Net,
enc_channels=(32,64,128,256,512), enc_num_head=(2,4,8,16,32),
dec_channels=(64,64,128,256)), same dataset / transforms / loss / schedule.
Used as the seg-only baseline for the three-way comparison:
  det-only (v3m4) ↔ multi-task (v3m4) ↔ seg-only (this config).

Dataset: ``AgcoBBoxV1(load_segment=True, segment_subdir="segment",
seg_label_map={4: 0})``. Per-point labels live at
``<root>/<sensor>/segment/<ts>.npy``. On-disk class index space:

    0 = background
    1 = tractor
    2 = harvester
    3 = trailer
    4 = car   ← remapped to background via seg_label_map

Usage:
    sh scripts/train.sh -d agco -c semseg-ptv3-v3m4-0-3cls-agco-ouster \\
        -n seg_v3m4_ouster_720ep -g 1
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 1.0
gradient_accumulation_steps = 1

# ── Seg metadata ─────────────────────────────────────────────────────────────
num_seg_classes = 4  # background + tractor + harvester + trailer
seg_ignore_index = -1
seg_class_names = ["background", "tractor", "harvester", "trailer"]

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type="Dense3DETRSegmentor",
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    backbone_grid_size=0.05,
    num_seg_classes=num_seg_classes,
    backbone_out_channels=64,
    seg_criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=seg_ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0,
             ignore_index=seg_ignore_index),
    ],
    seg_ignore_index=seg_ignore_index,
)

# ── Schedule ─────────────────────────────────────────────────────────────────
epoch = 1
eval_epoch = 1

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

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_type = "AgcoBBoxV1"
data_root = "/mnt/data/pointcloud_datasets/Pointcept/agco2026"
meta_data_dir = "/mnt/data/pointcloud_datasets/Pointcept/agco2026/meta_data"
sensors = ["ouster"]
num_points = 40_000

# Detection classes are still listed for AgcoBBoxV1's bookkeeping, but boxes
# are not consumed by Dense3DETRSegmentor.
included_classes = ("tractor", "harvester", "trailer")
max_num_obj = 16
min_inliers = dict(lslidar=350, ouster=200, rslidar=80)

fixed_pc_dims = {
    "lslidar": dict(min=[0.0, -55.0, -11.0], max=[60.0, 55.0, 10.0]),
    "ouster": dict(min=[0.0, -36.0, -10.0], max=[40.0, 36.0, 8.0]),
    "rslidar": dict(min=[0.0, -8.5, -8.0], max=[20.0, 8.5, 6.0]),
}

det_crop_transforms = [
    dict(
        type="FovCropDetection",
        azimuth_deg=(-60, 60),
        elevation_deg=None,
        crop_points=True,
        per_sensor={
            "lslidar": dict(azimuth_deg=(-60, 60)),
            "ouster": dict(azimuth_deg=(-60, 60)),
            "rslidar": dict(azimuth_deg=(-60, 60)),
        },
    ),
    dict(
        type="SphericalCropDetection",
        max_dist=60.0,
        min_dist=1.0,
        per_sensor={
            "lslidar": dict(max_dist=60.0),
            "ouster": dict(max_dist=40.0),
            "rslidar": dict(max_dist=20.0),
        },
    ),
]

_dataset_kwargs = dict(
    type=dataset_type,
    root_dir=data_root,
    meta_data_dir=meta_data_dir,
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
    load_segment=True,
    segment_subdir="segment",
    seg_label_map={4: 0},  # car → background
)

data = dict(
    num_classes=num_seg_classes,
    ignore_index=seg_ignore_index,
    names=seg_class_names,
    train=dict(
        **_dataset_kwargs,
        split="train",
        loop=1,
        transform=[
            *det_crop_transforms,
            dict(type="RandomFlipDetection", p_x=0.0, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            dict(type="GridSampleDetection", grid_size=0.05),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),
    val=dict(
        **_dataset_kwargs,
        split="train",
        transform=[
            *det_crop_transforms,
            dict(type="GridSampleDetection", grid_size=0.05),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),
    test=dict(
        **_dataset_kwargs,
        split="train",
        transform=[
            *det_crop_transforms,
            dict(type="GridSampleDetection", grid_size=0.05),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),
)

# ── Hooks ────────────────────────────────────────────────────────────────────
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# ── Tester ───────────────────────────────────────────────────────────────────
test = dict(type="AgcoSemSegDenseTester", verbose=True, save_predictions=True)
