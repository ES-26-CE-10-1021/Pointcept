"""
3DETR on ScanNet — Utonia (PT-v3m3) frozen pre-encoder variant.

Identical topology to ``det-3detr-v3m1-0-scannet.py`` (PTv3 pre-encoder
+ Vanilla Transformer encoder + 3DETR decoder) except the pre-encoder
is a **frozen, pretrained Utonia VFM** (``PT-v3m3``). Only the 3DETR
transformer encoder, decoder, and MLP heads receive gradients —
Utonia's weights are never updated.

Utonia was pretrained on a 9-dim ``[xyz, rgb, normal]`` input. The
ScanNet detection dataset yields only XYZ (``use_color=False``), so
the missing RGB/normal slots are zero-padded on-device inside
``PTv3m3PreEncoder``. Causal Modality Blinding during pretraining
makes Utonia tolerate zero-padded modalities.

The full Utonia hyperparameter set (``in_channels``, ``enc_depths``,
``enc_channels``, ``enc_num_head``, ``order``, ``stride``, etc.) is
driven by the checkpoint config bundled with the HuggingFace download,
not by this file — keeping the config short and immune to hyperparameter
drift when Utonia releases new checkpoints.

First-run HF download note: under DDP, ``PTv3m3PreEncoder`` downloads
the checkpoint only on rank 0 and fans out via cache on other ranks.
A cold first-run download of a 137M-parameter checkpoint can exceed
the default NCCL barrier timeout on a slow link — in production, warm
the cache once before launching distributed training::

    python -c "from third_party.utonia.utonia.model import load; \\
               load(name='utonia', ckpt_only=True)"

Usage:
    sh scripts/train.sh -d scannet -c det-3detr-utonia-v1m1-0-scannet -n my_utonia_exp -g 4

Data:
    Update ``data_root`` and ``meta_data_dir`` to point to your
    scannet_train_detection_data/ and meta_data/ directories.
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
# The VanillaTransformerEncoder3DETR and encoder_to_decoder_projection
# must match this. If a future checkpoint changes its enc width, update
# both references below.
UTONIA_ENC_DIM = 576

# ── Model ─────────────────────────────────────────────────────────────────────
model = dict(
    type="Model3DETRDetector",
    # Frozen Utonia VFM as pre-encoder. All other hyperparameters
    # (in_channels=9, enc_depths, enc_channels, rope_base, …) come from
    # the HuggingFace checkpoint config — we only pin the bits that are
    # policy, not architecture.
    pre_encoder=dict(
        type="PTv3m3PreEncoder",
        pretrained="utonia",
        grid_size=0.02,
        enc_mode=True,
        freeze_backbone="full",
    ),
    # Vanilla Transformer encoder (no masking / downsampling). Width
    # must match Utonia's deepest-stage output.
    encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=UTONIA_ENC_DIM,
        nhead=4,
        nlayers=3,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
    ),
    # Cross-attention decoder for box queries (unchanged from v3m1-0).
    decoder=dict(
        type="TransformerDecoder3DETR",
        decoder_dim=256,
        nhead=4,
        nlayers=8,
        ffn_dim=256,
        dropout=0.1,
    ),
    # ScanNet dataset metadata (class count, box parametrisation, etc.)
    dataset_config=dict(type="ScanNetDetectionConfig"),
    encoder_dim=UTONIA_ENC_DIM,  # drives encoder_to_decoder_projection
    decoder_dim=256,
    num_queries=256,
    position_embedding="fourier",
    mlp_dropout=0.3,
    # LayerNorm: padding-safe with variable-length PTv3 output. The
    # frozen Utonia output is still variable-length per scene, so we
    # keep 'ln' exactly as in v2/v3.
    projection_norm="ln",
    # Detection criterion (Hungarian matching + weighted box losses).
    criterion=dict(
        type="SetCriterion3DETR",
        matcher_cfg=dict(
            cost_class=1.0,
            cost_objectness=0.0,     # native default (disabled)
            cost_giou=2.0,           # native default
            cost_center=0.0,         # native default (disabled)
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
        num_angle_bin=1,       # ScanNet: axis-aligned boxes, no rotation bins
    ),
)

# ── Scheduler ─────────────────────────────────────────────────────────────────
epoch = 720
eval_epoch = 20      # evaluate every N epochs

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

# Semantic class names (18 detection categories)
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
        augment=True,
        random_cuboid_min_points=30000,
    ),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=False,
    ),
    test=dict(
        type=dataset_type,
        root_dir=data_root,
        meta_data_dir=meta_data_dir,
        split="val",
        num_points=40000,
        use_color=False,
        use_height=False,
        augment=False,
    ),
)

# ── Hooks ─────────────────────────────────────────────────────────────────────
# CheckpointLoader has no keywords/replacement — Utonia weights are pulled
# directly by PTv3m3PreEncoder from HuggingFace on __init__. The bare
# CheckpointLoader is kept only for training-resume semantics.
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
