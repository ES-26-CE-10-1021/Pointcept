"""
3DETR on AGCO — v1m3-0-3cls-ouster-dino: PointNet++ + DINO fusion (ouster).

Architecture:
  PointnetSAPreEncoderWithDino(2048 pts)
  + DinoInjectionEncoder wrapping VanillaTransformerEncoder3DETR(256d, 3L)
  + TransformerDecoder3DETR(256d, 8L). num_queries=32,
  center_offset_normalized=True, max_num_obj=16.

DINO features are gathered at the same FPS-selected center points the SA
layer emits (one DINO vector per output point, no neighborhood pooling),
and added on top of the ball-aggregated SA features inside the encoder slot.

Dataset:
  AgcoBBoxDinoV1, sensors=["ouster"], num_points=40_000, 3-class,
  DINO-projected point features, gravity leveled, fixed_pc_dims,
  ±60° FOV + spherical crops (max_dist=40 m), min_inliers=200.

Criterion (SUN-like):
  matcher class=1/objectness=5/giou=3/center=5;
  loss_giou=0, loss_no_object=0.1, loss_center=5, loss_size=1,
  loss_angle_cls=0.1, loss_angle_reg=0.5. giou_on_aux_outputs=False.

Usage:
  sh scripts/train.sh -d agco -c det-3detr-v1m3-0-3cls-agco-ouster-dino -n v1m3_ouster_dino -g 2
"""

_base_ = ["../_base_/default_runtime.py"]

# -- Training -----------------------------------------------------------------
batch_size = 8
num_worker = 16
mix_prob = 0
enable_amp = False
find_unused_parameters = False
clip_grad = 1.0
gradient_accumulation_steps = 1

included_classes = ("tractor", "harvester", "trailer")
num_semcls = len(included_classes)
num_angle_bin = 12
max_num_obj = 16

# -- DINO ---------------------------------------------------------------------
dino_feature_dim = 1280
dino_camera = "jai_left"
dino_dir = "dino_patch_h16plus_full_res"

# -- Model --------------------------------------------------------------------
model = dict(
    type="Model3DETRDetectorWithDino",
    pre_encoder=dict(
        type="PointnetSAPreEncoderWithDino",
        npoint=2048,
        radius=0.2,
        nsample=64,
        mlp_dims=[0, 64, 128, 256],
        normalize_xyz=True,
    ),
    encoder=dict(
        type="DinoInjectionEncoder",
        inner_encoder=dict(
            type="VanillaTransformerEncoder3DETR",
            encoder_dim=256,
            nhead=4,
            nlayers=3,
            ffn_dim=128,
            dropout=0.1,
            activation="relu",
        ),
        dino_dim=dino_feature_dim,
        encoder_dim=256,
    ),
    decoder=dict(
        type="TransformerDecoder3DETR",
        decoder_dim=256,
        nhead=4,
        nlayers=8,
        ffn_dim=256,
        dropout=0.1,
    ),
    dataset_config=dict(
        type="AgcoBBoxConfig",
        num_angle_bin=num_angle_bin,
        included_classes=included_classes,
    ),
    encoder_dim=256,
    decoder_dim=256,
    num_queries=32,
    position_embedding="fourier",
    mlp_dropout=0.3,
    center_offset_normalized=True,
    criterion=dict(
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
)

# -- Schedule -----------------------------------------------------------------
epoch = 720
eval_epoch = 20

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

# -- Dataset ------------------------------------------------------------------
dataset_type = "AgcoBBoxDinoV1"
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

common_dataset_kwargs = dict(
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
    # DINO
    use_dino=True,
    dino_camera=dino_camera,
    dino_subdir=dino_dir,
)

data = dict(
    train=dict(
        type=dataset_type,
        split="train",
        loop=1,
        transform=[
            *det_crop_transforms,
            dict(type="RandomFlipDetection", p_x=0.0, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
        **common_dataset_kwargs,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        loop=1,
        transform=[
            *det_crop_transforms,
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
        **common_dataset_kwargs,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        loop=1,
        transform=[
            *det_crop_transforms,
            dict(type="PointSubsampleDetection", num_points=num_points),
        ],
        **common_dataset_kwargs,
    ),
)

# -- Hooks --------------------------------------------------------------------
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="ObjDetEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# -- Tester -------------------------------------------------------------------
test = dict(type="ObjDetTester", verbose=True, save_predictions=True)
