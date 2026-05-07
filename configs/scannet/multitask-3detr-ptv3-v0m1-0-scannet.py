"""
Multi-task PTv3 + 3DETR on ScanNet (semseg + detection).

Architecture: ``MultiTask3DETRSegmentor`` — full PTv3 U-Net backbone (encoder +
decoder, matching ``configs/scannet/semseg-pt-v3m1-0-base.py``), with the
semseg head reading the decoder output (unpooled to root resolution) and the
3DETR detection branch (Vanilla Transformer encoder + 3DETR decoder + box
heads) tapping the encoder bottleneck.

Standalone PTv3 semseg training is unchanged — still goes through
``DefaultSegmentorV2`` + ``configs/scannet/semseg-pt-v3m1-0-base.py``. This
config is purpose-built for the joint-loss multi-task path.

Dataset: ``ScanNetDetectionDataset(load_segment=True)`` — emits the 3DETR
box keys plus the per-point ``segment`` array (mapped via
``nyu40id2class_semseg``, with detection-class labels in [0..19] and
ignore-label = -100 for non-detection-class pixels).

Usage:
    sh scripts/train.sh -d scannet -c multitask-3detr-ptv3-v0m1-0-scannet -n my_multitask -g 4

Data:
    Update ``data_root`` and ``meta_data_dir`` to point to your
    ``scannet_train_detection_data/`` and ``meta_data/`` directories.
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8       # total across all GPUs
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# ── Class metadata ───────────────────────────────────────────────────────────
# Detection: 18 classes (NYU-40 subset).
num_semcls = 18
class_names = [
    "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter",
    "desk", "curtain", "refrigerator", "showercurtrain",
    "toilet", "sink", "bathtub", "garbagebin",
]
# Semseg: 20 classes via the dataset's NYU-40 → semseg-class mapping
# (DETECTION_NYU40_IDS_SEMSEG, see scannet_detection.py:30). Non-mapped pixels
# get the ignore label IGNORE_LABEL=-100 inside ScanNetDetectionDataset.
num_seg_classes = 20
seg_ignore_index = -100
seg_class_names = [
    "wall", "floor", "cabinet", "bed", "chair",
    "sofa", "table", "door", "window", "bookshelf",
    "picture", "counter", "desk", "curtain", "refrigerator",
    "showercurtrain", "toilet", "sink", "bathtub", "garbagebin",
]

# ── Model ────────────────────────────────────────────────────────────────────
model = dict(
    type="MultiTask3DETRSegmentor",
    # PT-v3m1 full U-Net backbone (mirrors configs/scannet/semseg-pt-v3m1-0-base.py).
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,                       # XYZ only; matches detection-pipeline input
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
        enc_mode=False,                      # full U-Net required for multi-task
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    backbone_grid_size=0.02,
    # Semseg head + criteria.
    num_seg_classes=num_seg_classes,
    backbone_out_channels=64,                # = dec_channels[0]
    seg_criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=seg_ignore_index),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0,
             ignore_index=seg_ignore_index),
    ],
    seg_ignore_index=seg_ignore_index,
    # 3DETR detection branch (Vanilla Transformer encoder + 3DETR decoder).
    det_encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=512,
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
    det_dataset_config=dict(type="ScanNetDetectionConfig"),
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
        num_angle_bin=1,
    ),
    encoder_dim=512,                          # = enc_channels[-1]
    decoder_dim=256,
    num_queries=256,
    position_embedding="fourier",
    mlp_dropout=0.3,
    projection_norm="ln",
    # Multi-task loss weights — tune as needed.
    seg_weight=1.0,
    det_weight=1.0,
)

# ── Scheduler ────────────────────────────────────────────────────────────────
epoch = 720
eval_epoch = 20

optimizer = dict(type="AdamW", lr=5e-4, weight_decay=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[5e-4],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=500.0,
    final_div_factor=1.0,
)

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset_type = "ScanNetDetectionDataset"
data_root = "/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data"
meta_data_dir = "/home/andreas/3D-Perception/votenet/scannet/meta_data"

data = dict(
    num_classes=num_seg_classes,
    ignore_index=seg_ignore_index,
    names=seg_class_names,
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="train",
        num_points=40000,
        use_color=False,
        use_height=False,
        load_segment=True,
        transform=[
            dict(type="RandomCuboidDetection", min_points=30000),
            dict(type="GridSampleDetection", grid_size=0.02),
            dict(type="PointSubsampleDetection", num_points=40000),
            dict(type="RandomFlipDetection", p_x=0.5, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
        ],
    ),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=False,
        use_height=False,
        load_segment=True,
        transform=[
            dict(type="GridSampleDetection", grid_size=0.02),
            dict(type="PointSubsampleDetection", num_points=40000),
        ],
    ),
    test=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=False,
        use_height=False,
        load_segment=True,
        transform=[
            dict(type="GridSampleDetection", grid_size=0.02),
            dict(type="PointSubsampleDetection", num_points=40000),
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
test = dict(type="CombinedSegDetTester", verbose=True)
