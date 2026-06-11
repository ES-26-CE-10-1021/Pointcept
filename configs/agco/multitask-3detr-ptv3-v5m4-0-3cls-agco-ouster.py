"""
Multi-task PTv3 + 3DETR on AGCO (semseg + detection) — v5m4 ouster variant.

Identical to ``multitask-3detr-ptv3-v2m4-0-3cls-agco-ouster.py`` except the
Utonia backbone is **fully frozen** (``freeze_backbone="enc"``) instead of
``"enc_finetune"``. The fresh PT-v3m1-style decoder bolted onto the
encoder-only Utonia checkpoint remains fully trainable, as do the
detection-branch transformer encoder/decoder and the semseg head — so this
isolates the contribution of fine-tuning Utonia's last encoder stage on
the joint semseg+detection task.

Architecture:

    input (B, N, 3) ─► Utonia embedding (frozen) ─► Utonia encoder (frozen) ─► bottleneck ─┐
                                                                                            │
                                ┌───────────────────────────────────────────────────────────┘
                                │
            FPS 2048 ─► Vanilla encoder (3L, 576d) ─► proj ─► 3DETR decoder ─► boxes
                                │
            fresh PT-v3m1-style decoder (dec_channels=(64,64,128,256)) ─► unpool ─► seg_logits

Pairs with the detection-only ``det-3detr-v5m2-0-3cls-agco-ouster.py``
(same fully-frozen Utonia + FPS 2048 + Vanilla encoder recipe) for a
clean det-only ↔ multi-task ablation under the foundation-model regime.

Dataset: ``AgcoBBoxV1(load_segment=True, segment_subdir="segment",
seg_label_map={4: 0}, utonia_preprocess=True)``. Per-point labels live at
``<root>/<sensor>/segment/<ts>.npy``. On-disk class index space:

    0 = background
    1 = tractor
    2 = harvester
    3 = trailer
    4 = car   ← remapped to background via seg_label_map
    (hopper is detection-only and never appears in segment files)

Semseg and detection share the same foreground classes
(tractor, harvester, trailer).

Usage:
    sh scripts/train.sh -d agco -c multitask-3detr-ptv3-v5m4-0-3cls-agco-ouster \\
        -n mt_v5m4_ouster_720ep -g 1
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

# ── Class metadata ───────────────────────────────────────────────────────────
included_classes = ("tractor", "harvester", "trailer")
num_semcls = len(included_classes)
num_angle_bin = 12
max_num_obj = 16

num_seg_classes = 4  # background + tractor + harvester + trailer
seg_ignore_index = -1
seg_class_names = ["background", "tractor", "harvester", "trailer"]

# Utonia's deepest-stage output width (PT-v3m3 enc_channels[-1]).
UTONIA_ENC_DIM = 576

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type="MultiTask3DETRSegmentor",
    backbone=dict(
        type="PTv3m3PreEncoder",
        pretrained="utonia",
        grid_size=0.05,
        enc_mode=False,
        freeze_backbone="enc",  # full encoder freeze (embedding + encoder)
        # Fresh decoder bolted onto the encoder-only Utonia checkpoint.
        # PT-v3m1 base decoder shape; dec_rope_enable=False because RoPE
        # in PT-v3m3 requires head_dim divisible by 3 and 64/4=16 isn't.
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        dec_rope_enable=False,
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
    det_encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=UTONIA_ENC_DIM,
        nhead=4,
        nlayers=3,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
    ),
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
        giou_on_aux_outputs=False,
        matcher_cfg=dict(
            cost_class=1.0,
            cost_objectness=5.0,
            cost_giou=3.0,
            cost_center=5.0,
        ),
        loss_weight_dict=dict(
            loss_giou_weight=0.0,
            loss_sem_cls_weight=1.0,
            loss_no_object_weight=0.1,
            loss_angle_cls_weight=0.1,
            loss_angle_reg_weight=0.5,
            loss_center_weight=5.0,
            loss_size_weight=1.0,
        ),
        num_semcls=num_semcls,
        num_angle_bin=num_angle_bin,
    ),
    encoder_dim=UTONIA_ENC_DIM,
    decoder_dim=256,
    num_queries=32,
    position_embedding="fourier",
    mlp_dropout=0.3,
    projection_norm="ln",
    center_offset_normalized=True,
    det_fps_npoint=2048,
    # Multi-task loss weighting — uncertainty (Cipolla, c_i=2) by default.
    loss_weighting="uncertainty",
    c_seg=2.0,
    c_det=2.0,
    seg_weight=1.0,
    det_weight=1.0,
)

# Exclude the learnable σ params from weight decay.
param_dicts = [dict(keyword="log_sigma_sq", weight_decay=0.0)]

# ── Schedule ─────────────────────────────────────────────────────────────────
epoch = 720
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
sensors = ["ouster"]
num_points = 40_000

class_names = list(included_classes)
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
    utonia_preprocess=True,
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
    seg_label_map={4: 0},  # car → background (semseg matches detection classes)
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
        split="val",
        transform=[
            *det_crop_transforms,
            dict(type="GridSampleDetection", grid_size=0.05),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
    ),
    test=dict(
        **_dataset_kwargs,
        split="test",
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
