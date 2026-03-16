_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 16  # bs: total bs in all gpus
gradient_accumulation_steps = 1
num_worker = 15
mix_prob = 0.8
empty_cache = True
enable_amp = True
sync_bn = True
num_worker_per_gpu = 15
EPOCHS = 50
enable_wandb = True
wandb_project = "PTv3-late-fusion"

# dataset settings
dataset_type = "AgcoRealDinoDataset"
data_root = "/mnt/data/pointcloud_datasets/agco_ttcbtg/"
ignore_index = -1
label_names = [
    "ground",
    "tractor",
    "harvester",
    "trailer",
    "truck",
    "truck trailer",
    "buildings",
    "trees",
]

# model settings
# Late fusion: vanilla PTv3 backbone, DINO features concatenated after backbone via KNN
model = dict(
    type="DINOEnhancedSegmentor",
    num_classes=len(label_names),
    backbone_out_channels=64,
    dino_feat_size=1280,
    project_dino_feat=False, # Projects dino feature vector to backbone_out_channels before concat
    normalize_dino_feat=False, # Normalizes backbone and dino features before concat
    backbone=dict(
        type="PT-v3m1",
        in_channels=3,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
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
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)


# scheduler settings
epoch = EPOCHS
eval_epoch = EPOCHS // 5
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.002, 0.0002],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.0002)]

# Register dino_feat, dino_coord, origin_coord for subsampling by GridSample
_index_valid_keys = [
    "coord", "color", "normal", "superpoint",
    "strength", "segment", "instance",
    "dino_feat", "dino_coord", "origin_coord",
]

data = dict(
    num_classes=len(label_names),
    ignore_index=ignore_index,
    names=label_names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict={"index_valid_keys": _index_valid_keys}),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-0.3, -0.3], axis="y", p=1.0),
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Copy coord to dino_coord and origin_coord after GridSample
            # (dino_feat already subsampled alongside coord by GridSample)
            dict(type="Copy", keys_dict={"coord": "dino_coord"}),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "dino_feat", "dino_coord", "origin_coord"),
                offset_keys_dict=dict(offset="coord", dino_offset="dino_coord", origin_offset="origin_coord"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict={"index_valid_keys": _index_valid_keys}),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(type="RandomRotate", angle=[-0.3, -0.3], axis="y", p=1.0),
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="Copy", keys_dict={"coord": "dino_coord"}),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse",
                      "dino_feat", "dino_coord", "origin_coord"),
                offset_keys_dict=dict(offset="coord", dino_offset="dino_coord", origin_offset="origin_coord"),
                feat_keys=("coord",),
            ),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="Update", keys_dict={"index_valid_keys": _index_valid_keys}),
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
            dict(type="Copy", keys_dict={"coord": "dino_coord"}),
            dict(type="Copy", keys_dict={"coord": "origin_coord"}),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "dino_feat", "dino_coord", "origin_coord"),
                    offset_keys_dict=dict(offset="coord", dino_offset="dino_coord", origin_offset="origin_coord"),
                    feat_keys=("coord",),
                ),
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1 / 2], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis="z", center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3 / 2], axis="z", center=[0, 0, 0], p=1)],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
