_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
gradient_accumulation_steps = 1
num_worker = 3
mix_prob = 0.8
empty_cache = True
enable_amp = True
sync_bn = True
num_worker_per_gpu = 2
EPOCHS = 50
enable_wandb = True
wandb_project = "PTv3-intensity-prediction-w-inj"

hooks = [
    dict(type='CheckpointLoader'),
    dict(type='ModelHook'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='RegressionEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
]
train = dict(type='DefaultTrainer')
test = dict(type='RegressionEvaluator', verbose=True)
# dataset settings
dataset_type = "AgcoRealDinoDataset"
data_root = "/media/ai/T7/agco_ttcbtg/"
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
model = dict(
    type="DefaultIntensityRegressor",
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1-injection-bottleneck",
        in_channels=3, # <---- Num features in point cloud
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
        dec_dino_injection=(True, False, False, False),
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
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="L1Loss"),
    ],
)


# scheduler settings
epoch = EPOCHS
eval_epoch = EPOCHS
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


data = dict(
    num_classes=len(label_names),
    ignore_index=ignore_index,
    names=label_names,
    # train=dict(
    #     type=dataset_type,
    #     split="train",
    #     data_root=data_root,
    #     transform=[
    #         # dict(type="AssertDinoFeat"),
    #         # dict(type="LimitMaxPoints", max_points=70000),
    #         # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
    #         # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
    #         dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
    #         # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
    #         dict(type="RandomRotate", angle=[0.1, 0.1], axis="y", p=1.0),
    #         dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
    #         dict(type="RandomScale", scale=[0.9, 1.1]),
    #         # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
    #         dict(type="RandomFlip", p=0.5),
    #         dict(type="RandomJitter", sigma=0.005, clip=0.02),
    #         # dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
    #         dict(
    #             type="GridSample",
    #             grid_size=0.05,
    #             hash_type="fnv",
    #             mode="train",
    #             return_grid_coord=True,
    #         ),
    #         # dict(type="SphereCrop", point_max=1000000, mode="random"),
    #         # dict(type="CenterShift", apply_z=False),
    #         dict(type="ToTensor"),
    #         dict(
    #             type="Collect",
    #             keys=("coord", "grid_coord", "segment", "dino_feat", "strength"),
    #             feat_keys=("coord",), # "strength"),
    #         ),
    #         # dict(type="AssertDinoFeat"),
    #
    #     ],
    #     test_mode=False,
    #     ignore_index=ignore_index,
    # ),
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            # dict(type="AssertDinoFeat"),
            # dict(type="LimitMaxPoints", max_points=70000),
            # dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            # dict(type="RandomRotateTargetAngle", angle=(1/2, 1, 3/2), center=[0, 0, 0], axis="z", p=0.75),
            # dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            # dict(type="RandomRotate", angle=[-1/6, 1/6], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[0.11, 0.09], axis="y", p=1.0, center=[0,0,0]),
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            # dict(type="RandomShift", shift=[0.2, 0.2, 0.2]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            # dict(type="ElasticDistortion", distortion_params=[[0.02, 0.04], [0.08, 0.16]]),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # dict(type="SphereCrop", point_max=100000, mode="random"),
            # dict(type="CenterShift", apply_z=False),
            # dict(type="VisualizePostTransforms"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "dino_feat", "strength"),
                feat_keys=("coord",), # "strength"),
            ),
            # dict(type="AssertDinoFeat"),

        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    # val=dict(
    #     type=dataset_type,
    #     split="val",
    #     data_root=data_root,
    #     transform=[
    #         dict(type="Copy", keys_dict={"segment": "origin_segment", "strength": "origin_strength"}),
    #         dict(type="RandomRotate", angle=[-0.3, -0.3], axis="y", p=1.0),
    #         dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
    #         dict(
    #             type="GridSample",
    #             grid_size=0.05,
    #             hash_type="fnv",
    #             mode="train",
    #             return_grid_coord=True,
    #             return_inverse=True,
    #         ),
    #         dict(type="ToTensor"),
    #         dict(
    #             type="Collect",
    #             keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "dino_feat", "strength", "origin_strength"),
    #             feat_keys=("coord",), # "strength"),
    #         ),
    #     ],
    #     test_mode=False,
    #     ignore_index=ignore_index,
    # ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment", "strength":"origin_strength"}),
            dict(type="RandomRotate", angle=[0.1, 0.1], axis="y", p=1.0),
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse", "dino_feat", "strength", "origin_strength"),
                feat_keys=("coord",), # "strength"),
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
            dict(type="PointClip", point_cloud_range=(-75.2, -75.2, -4, 75.2, 75.2, 2)),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
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
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", ), # "strength"),
                ),
            ],
            aug_transform=[
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
            ],
        ),
        ignore_index=ignore_index,
    ),
)
