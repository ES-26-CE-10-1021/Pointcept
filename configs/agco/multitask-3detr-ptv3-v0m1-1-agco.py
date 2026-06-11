"""
Multi-task PTv3 + 3DETR on AGCO — overfitting variant (train split for all).

Identical to multitask-3detr-ptv3-v0m1-0-agco.py except val and test both
point at the train split. Use this to verify the model can overfit before
committing to a full training run.

Usage:
    sh scripts/train.sh -d agco -c multitask-3detr-ptv3-v0m1-1-agco -n my_multitask_overfitting -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 2
gradient_accumulation_steps = 4  # effective batch size = 8
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# ── Class metadata ───────────────────────────────────────────────────────────
included_classes = ("tractor", "harvester", "trailer")
num_semcls = len(included_classes)
num_angle_bin = 12

num_seg_classes = 4  # background + tractor + harvester + trailer (matches detection)
seg_ignore_index = -1
seg_class_names = ["background", "tractor", "harvester", "trailer"]

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type="MultiTask3DETRSegmentor",
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
    det_encoder=dict(type="IdentityEncoder3DETR"),
    det_decoder=dict(
        type="TransformerDecoder3DETR",
        decoder_dim=256,
        nhead=4,
        nlayers=8,
        ffn_dim=256,
        dropout=0.1,
    ),
    det_dataset_config=dict(
        type="AgcoBBoxConfig",
        num_angle_bin=num_angle_bin,
        included_classes=included_classes,
    ),
    det_criterion=dict(
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
    encoder_dim=512,
    decoder_dim=256,
    num_queries=128,
    position_embedding="fourier",
    mlp_dropout=0.3,
    projection_norm="ln",
    loss_weighting="uncertainty",
    c_seg=2.0,
    c_det=2.0,
    seg_weight=1.0,
    det_weight=1.0,
)

param_dicts = [dict(keyword="log_sigma_sq", weight_decay=0.0)]

# ── Schedule ─────────────────────────────────────────────────────────────────
epoch = 100
eval_epoch = 20

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[5e-4, 5e-4],  # one entry per param_group (default + log_sigma_sq)
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
sensors = ["lslidar"]
num_points = 100_000

class_names = list(included_classes)
min_inliers = 350

det_crop_transforms = [
    dict(type="SphericalCropDetection", max_dist=60.0, min_dist=0.0),
    dict(type="FovCropDetection", azimuth_deg=(-60.0, 60.0), crop_points=True),
]

_dataset_kwargs = dict(
    type=dataset_type,
    root_dir=data_root,
    meta_data_dir=meta_data_dir,
    split="train",
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
    load_segment=True,
    segment_subdir="segment",
    seg_label_map={4: 0},  # car → background (semseg matches detection classes)
)

data = dict(
    num_classes=num_seg_classes,
    ignore_index=seg_ignore_index,
    names=seg_class_names,
    train=dict(
        **_dataset_kwargs,
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
        transform=[
            *det_crop_transforms,
            dict(type="GridSampleDetection", grid_size=0.05),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),
    test=dict(
        **_dataset_kwargs,
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
    dict(type="CombinedSegDetEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# ── Tester ───────────────────────────────────────────────────────────────────
test = dict(type="CombinedSegDetTester", verbose=True, save_predictions=True)
