"""
3DETR on ScanNet — Utonia (PT-v3m3) full U-Net frozen pre-encoder + FPS, v4m2.

v4m2 = v4m1 with ``enc_mode=False``: the full pretrained Utonia U-Net
(encoder + decoder) runs frozen, emitting high-resolution per-point
features at the shallowest decoder stage. These are then FPS-downsampled
to the same 2048-token budget as v4m1 so the two configs can be compared
head-to-head: *"does the decoder's skip-connected detail improve
detection once token count is held fixed?"*

Key differences from ``det-3detr-utonia-v4m1-0-scannet.py``:
  - ``pre_encoder.enc_mode = False``  (run Utonia's decoder)
  - ``encoder.encoder_dim`` / ``model.encoder_dim = UTONIA_DEC_DIM = 54``
    (PT-v3m3 ScanNet ``dec_channels[0]``, see
    ``configs/utonia/semseg-utonia-v1m1-0b-scannet-dec.py:26``)
  - ``pre_encoder.freeze_backbone = "full"`` still — both the encoder
    and the decoder are frozen. Fine-tuning the decoder is out of scope
    for this experiment.

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-utonia-v4m2-0-scannet -n my_exp -g 4
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8       # total across all GPUs
num_worker = 32
mix_prob = 0         # detection dataset does not support MixUp
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# Utonia's shallowest decoder stage output width (PT-v3m3 dec_channels[0]).
# Source: configs/utonia/semseg-utonia-v1m1-0b-scannet-dec.py:26.
UTONIA_DEC_DIM = 54

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    pre_encoder=dict(
        type="PTv3m3PreEncoder",
        pretrained="utonia",
        grid_size=0.01,
        enc_mode=False,
        npoint=2048,
        freeze_backbone="full",
    ),
    encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=UTONIA_DEC_DIM,
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
    encoder_dim=UTONIA_DEC_DIM,
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
        use_color=True,
        use_height=False,
        augment=True,
        random_cuboid_min_points=30000,
        utonia_preprocess=True,
    ),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=True,
        use_height=False,
        augment=False,
        utonia_preprocess=True,
    ),
    test=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=True,
        use_height=False,
        augment=False,
        utonia_preprocess=True,
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
