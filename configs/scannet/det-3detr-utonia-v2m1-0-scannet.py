"""
3DETR on ScanNet — Utonia (PT-v3m3) frozen pre-encoder, v2.

v2 = v1 + Utonia-canonical preprocessing (scale 0.5, CenterShift z+, grid
0.01), color and normals both absent → PT-v3m3 receives XYZ only with
zero-padded RGB and normal slots.

Differences from ``det-3detr-utonia-v1m1-0-scannet.py``:
  - ``pre_encoder.grid_size = 0.01``  (matches voxel topology after 0.5× scale)
  - ``data.*.utonia_preprocess = True``  (scale 0.5 + CenterShift before aug)
  - ``data.*.use_color = False``  (same as v1; listed explicitly for clarity)

Normals are never loaded by the detection dataset; their slots are zero-padded
on-device by ``PTv3m3PreEncoder._build_padded_feat``.

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-utonia-v2m1-0-scannet -n my_exp -g 4
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8       # total across all GPUs
num_worker = 32
mix_prob = 0         # detection dataset does not support MixUp
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# Utonia's deepest-stage output width (PT-v3m3 enc_channels[-1]).
UTONIA_ENC_DIM = 576

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    pre_encoder=dict(
        type="PTv3m3PreEncoder",
        pretrained="utonia",
        grid_size=0.01,
        enc_mode=True,
        freeze_backbone="enc",
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
    dataset_config=dict(type="ScanNetDetectionConfig"),
    encoder_dim=UTONIA_ENC_DIM,
    decoder_dim=256,
    num_queries=256,
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
        num_semcls=18,
        num_angle_bin=1,
    ),
)

# ── Scheduler ─────────────────────────────────────────────────────────────────
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
        use_color=False,
        use_height=False,
        utonia_preprocess=True,
        transform=[
            dict(type="RandomCuboidDetection", min_points=30000),
            dict(type="RandomFlipDetection", p_x=0.5, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            dict(type="GridSampleDetection", grid_size=0.01),
            dict(type="PointSubsampleDetection", num_points=40000),
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
        utonia_preprocess=True,
        transform=[
            dict(type="GridSampleDetection", grid_size=0.01),
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
        utonia_preprocess=True,
        transform=[
            dict(type="GridSampleDetection", grid_size=0.01),
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
