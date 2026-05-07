"""
3DETR on ScanNet — Utonia frozen encoder + fresh trainable PTv3 decoder, v5m2-0.

v5m2 tests "attach a randomly-initialized PTv3 decoder to the frozen
Utonia encoder." The Utonia checkpoint is encoder-only, so there are no
pretrained decoder weights — the decoder is constructed with the
hyperparameters below and trains from scratch end-to-end alongside the
3DETR head.

v5m2-0 specifics:
  - ``pre_encoder.enc_mode = False``  (build and run the decoder path)
  - ``pre_encoder.freeze_backbone = "enc"``  (embedding + encoder frozen;
    decoder is trainable)
  - ``pre_encoder.dec_*`` explicitly set (ckpt carries ``dec_*=None``).
    Values mirror ``configs/utonia/semseg-utonia-v1m1-0b-scannet-dec.py``;
    all head dimensions are 18 (= 54/3, 108/6, 216/12, 432/24) so
    PT-v3m3's 3D-RoPE ``head_dim % 3 == 0`` assertion is satisfied.
  - no FPS → variable-length output at the shallowest decoder stage
    (``dec_channels[0] = 54``), padding_mask + LN projection

``_load_pretrained_state`` logs every ``dec.*`` key as "missing — decoder
will train from random initialization" — that is expected for this
config; not an error.

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-utonia-v5m2-0-scannet -n my_exp -g 4
"""

_base_ = ["../_base_/default_runtime.py"]

# ── Training ─────────────────────────────────────────────────────────────────
batch_size = 8
num_worker = 32
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 0.1

# Shallowest decoder stage width — matches dec_channels[0] below. Head
# dim = 54/3 = 18 satisfies PT-v3m3's RoPE divisibility assertion.
UTONIA_DEC_DIM = 54

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    pre_encoder=dict(
        type="PTv3m3PreEncoder",
        pretrained="utonia",
        grid_size=0.01,
        enc_mode=False,
        freeze_backbone="enc",
        # Required: Utonia ckpt has dec_*=None, so we must specify. These
        # match the semseg-utonia-v1m1-0b-scannet-dec.py reference.
        dec_depths=(2, 2, 2, 2),
        dec_channels=(54, 108, 216, 432),
        dec_num_head=(3, 6, 12, 24),
        dec_patch_size=(1024, 1024, 1024, 1024),
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
        utonia_preprocess=True,
        transform=[
            dict(type="RandomCuboidDetection", min_points=30000),
            dict(type="GridSampleDetection", grid_size=0.01),
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
        use_color=True,
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
        use_color=True,
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
