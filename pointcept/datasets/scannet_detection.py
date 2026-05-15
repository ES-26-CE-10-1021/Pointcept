"""
ScanNet 3D Object Detection Dataset

Reads VoteNet-style ScanNet detection data (scene_vert.npy, scene_bbox.npy,
scene_ins_label.npy, scene_sem_label.npy) and returns batches suitable for
3DETR training and evaluation.

Data format is identical to third_party/3detr/datasets/scannet.py but
integrated into the Pointcept DATASETS registry.

Paths to the detection data (scannet_train_detection_data/) and meta_data/
must be specified in the config. This data was originally prepared by the
VoteNet project and is separate from Pointcept's per-scene .npy files.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from .builder import DATASETS

# Mean RGB colour used in 3DETR / VoteNet ScanNet preprocessing
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

# NYU-40 class IDs used for the 18 detection categories in ScanNet
DETECTION_NYU40_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)
DETECTION_NYU40_IDS_SEMSEG = np.array(
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)

IGNORE_LABEL = -100


def _random_cuboid_augment(point_cloud, instance_bboxes, per_point_labels,
                           min_points=30000):
    """Simple random cuboid crop (reimplemented to avoid 3detr import)."""
    try:
        import sys, os
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../third_party/3detr")
        )
        if path not in sys.path:
            sys.path.insert(0, path)
        from utils.random_cuboid import RandomCuboid
        aug = RandomCuboid(min_points=min_points)
        return aug(point_cloud, instance_bboxes, per_point_labels)
    except Exception:
        # Fall back: no cuboid crop
        return point_cloud, instance_bboxes, per_point_labels


def _random_sampling(pc, num_points):
    """Sub-sample or repeat-sample to exactly num_points."""
    n = pc.shape[0]
    if n >= num_points:
        choices = np.random.choice(n, num_points, replace=False)
    else:
        choices = np.random.choice(n, num_points, replace=True)
    return pc[choices], choices


def _rotz(t):
    """Rotation matrix around Z axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rotate_aligned_boxes(input_boxes, rot_mat):
    centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
    new_centers = np.dot(centers, np.transpose(rot_mat))
    dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
    new_x = np.zeros((dx.shape[0], 4))
    new_y = np.zeros((dx.shape[0], 4))
    for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
        crnrs = np.zeros((dx.shape[0], 3))
        crnrs[:, 0] = crnr[0] * dx
        crnrs[:, 1] = crnr[1] * dy
        crnrs = np.dot(crnrs, np.transpose(rot_mat))
        new_x[:, i] = crnrs[:, 0]
        new_y[:, i] = crnrs[:, 1]
    new_dx = 2.0 * np.max(new_x, 1)
    new_dy = 2.0 * np.max(new_y, 1)
    new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)
    return np.concatenate([new_centers, new_lengths], axis=1)


@DATASETS.register_module()
class ScanNetDetectionDataset(Dataset):
    """
    ScanNet dataset for 3D object detection (18 classes, axis-aligned boxes).

    Reads VoteNet-style pre-processed data:
        <root_dir>/<scan_name>_vert.npy       — (N, 6) xyzrgb vertices
        <root_dir>/<scan_name>_bbox.npy       — (M, 7) boxes: cx cy cz dx dy dz cls_nyu40
        <root_dir>/<scan_name>_ins_label.npy  — (N,) instance labels
        <root_dir>/<scan_name>_sem_label.npy  — (N,) semantic NYU-40 labels

    Args:
        root_dir (str): path to scannet_train_detection_data/
        meta_data_dir (str): path to directory containing scannetv2_train.txt
            and scannetv2_val.txt
        split (str): 'train' or 'val'
        num_points (int): number of points to sample per scene
        use_color (bool): append RGB colour (normalised) to point features
        use_height (bool): append height above floor as an extra feature
        transform (list[dict] | None): config-driven detection-aware
            augmentation pipeline (registered transforms in
            ``pointcept/datasets/det_transform.py``). Replaces the legacy
            ``augment`` / ``random_cuboid_min_points`` kwargs. Pipeline
            operates on the raw ``(point_cloud, gt_box_*_raw, ...)`` dict
            in step. The dataset still enforces ``num_points`` as a final
            safety net.
        utonia_preprocess (bool): if True, apply scale-0.5 + XY-mean /
            Z-min center-shift to points and box centers / sizes inline
            **before** the transform pipeline. Matches the canonical
            preprocessing the Utonia (PT-v3m3) checkpoint was pretrained
            with.
        load_segment (bool): if True, expose per-point semantic labels
            under the ``"segment"`` key in the returned dict. Mapped via
            the same NYU-40 → semseg-class table the dataset already uses
            for its internal ``sem_seg_labels`` derivation. Plumbed
            through the transform pipeline in lockstep with
            ``point_cloud`` for the multi-task semseg branch.
    """

    def __init__(
        self,
        root_dir,
        meta_data_dir,
        split="train",
        num_points=40000,
        use_color=False,
        use_height=False,
        transform=None,
        loop=1,
        utonia_preprocess=False,
        load_segment=False,
    ):
        assert split in ("train", "val"), f"Unknown split: {split}"
        self.root_dir = root_dir
        self.num_points = num_points
        self.use_color = use_color
        self.use_height = use_height
        from .transform import Compose
        self.transform = Compose(transform or [])
        self.loop = loop
        self.utonia_preprocess = utonia_preprocess
        self.load_segment = bool(load_segment)

        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(DETECTION_NYU40_IDS))
        }
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(DETECTION_NYU40_IDS_SEMSEG))
        }
        self.max_num_obj = 64

        # Normalisation range for box centres (unit cube)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]

        # Build scan list
        all_scan_names = set(
            os.path.basename(x)[0:12]
            for x in os.listdir(root_dir)
            if x.startswith("scene")
        )
        split_file = os.path.join(meta_data_dir, f"scannetv2_{split}.txt")
        with open(split_file, "r") as f:
            requested = f.read().splitlines()
        self.scan_names = [s for s in requested if s in all_scan_names]
        print(
            f"[ScanNetDetectionDataset] {split}: kept {len(self.scan_names)} / "
            f"{len(requested)} scans"
        )

    def __len__(self):
        return len(self.scan_names) * self.loop

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _shift_scale_points(pts, src_range, dst_range=None):
        """Normalise pts from src_range to [0, 1] (or dst_range)."""
        if dst_range is None:
            dst_range = [
                np.zeros((1, 3), dtype=np.float32),
                np.ones((1, 3), dtype=np.float32),
            ]
        scale = (dst_range[1] - dst_range[0]) / (
            src_range[1] - src_range[0] + 1e-6
        )
        bias = dst_range[0] - src_range[0] * scale
        return pts * scale + bias

    @staticmethod
    def _scale_points(pts, mult_factor):
        return pts * mult_factor

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx % len(self.scan_names)]

        mesh_vertices = np.load(os.path.join(self.root_dir, scan_name) + "_vert.npy")
        instance_bboxes = np.load(
            os.path.join(self.root_dir, scan_name) + "_bbox.npy"
        )

        # --- Build per-point arrays ---
        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3].astype(np.float32)
            pcl_color = mesh_vertices[:, 3:6].astype(np.float32)
        else:
            point_cloud = mesh_vertices[:, 0:6].astype(np.float32)
            if self.utonia_preprocess:
                point_cloud[:, 3:] = point_cloud[:, 3:] / 255.0
            else:
                point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:6].copy()

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1).astype(point_cloud.dtype)],
                axis=1,
            )

        # --- Build raw box arrays (axis-aligned: yaw=0). ---
        n_boxes = int(instance_bboxes.shape[0])
        centers_raw = instance_bboxes[:, 0:3].astype(np.float32)
        sizes_raw = instance_bboxes[:, 3:6].astype(np.float32)
        angles_raw = np.zeros((n_boxes,), dtype=np.float32)
        if n_boxes > 0:
            labels_raw = np.array(
                [self.nyu40id2class[int(x)] for x in instance_bboxes[:, -1]],
                dtype=np.int64,
            )
        else:
            labels_raw = np.zeros((0,), dtype=np.int64)

        # --- Optionally load per-point semantic labels (multi-task semseg). ---
        segment = None
        if self.load_segment:
            semantic_labels = np.load(
                os.path.join(self.root_dir, scan_name) + "_sem_label.npy"
            )
            segment = np.full_like(semantic_labels, IGNORE_LABEL, dtype=np.int64)
            for c in DETECTION_NYU40_IDS_SEMSEG:
                segment[semantic_labels == c] = self.nyu40id2class_semseg[c]

        # --- Utonia-canonical preprocessing (deterministic; pre-augment). ---
        # RandomScale(0.5) + CenterShift(XY mean, Z min). Applied to points
        # and to raw box centers/sizes so the transform pipeline operates
        # on the canonicalised frame.
        if self.utonia_preprocess:
            point_cloud[:, 0:3] *= 0.5
            centers_raw *= 0.5
            sizes_raw *= 0.5
            x_min, y_min, z_min = point_cloud[:, 0:3].min(axis=0)
            x_max, y_max, _ = point_cloud[:, 0:3].max(axis=0)
            shift = np.array(
                [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min], dtype=np.float32
            )
            point_cloud[:, 0:3] -= shift
            if n_boxes > 0:
                centers_raw -= shift

        # --- Augmentation pipeline (config-driven). ---
        data_dict = {
            "point_cloud": point_cloud,
            "pcl_color": pcl_color,
            "gt_box_centers_raw": centers_raw,
            "gt_box_sizes_raw": sizes_raw,
            "gt_box_angles_raw": angles_raw,
            "gt_box_labels_raw": labels_raw,
        }
        if segment is not None:
            data_dict["segment"] = segment
        data_dict = self.transform(data_dict)
        point_cloud = data_dict["point_cloud"]
        pcl_color = data_dict["pcl_color"]
        segment = data_dict.get("segment")  # None when load_segment=False
        centers_raw = data_dict["gt_box_centers_raw"]
        sizes_raw = data_dict["gt_box_sizes_raw"]
        angles_raw = data_dict["gt_box_angles_raw"]
        labels_raw = data_dict["gt_box_labels_raw"]

        # --- Safety-net subsample (idempotent if pipeline already did it). ---
        if point_cloud.shape[0] != self.num_points:
            point_cloud, choices = _random_sampling(point_cloud, self.num_points)
            pcl_color = pcl_color[choices]
            if segment is not None:
                segment = segment[choices]
        point_cloud = point_cloud.astype(np.float32)

        # --- Pack GT bounding boxes to MAX_NUM_OBJ. ---
        MAX_NUM_OBJ = self.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        num_gt = min(int(centers_raw.shape[0]), MAX_NUM_OBJ)
        target_bboxes_mask[:num_gt] = 1.0
        target_bboxes[:num_gt, 0:3] = centers_raw[:num_gt]
        target_bboxes[:num_gt, 3:6] = sizes_raw[:num_gt]
        raw_angles[:num_gt] = angles_raw[:num_gt]

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

        # Normalised box centres
        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = self._shift_scale_points(
            box_centers[np.newaxis, ...],
            src_range=[
                point_cloud_dims_min[np.newaxis, ...],
                point_cloud_dims_max[np.newaxis, ...],
            ],
            dst_range=self.center_normalizing_range,
        ).squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., np.newaxis]

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = self._scale_points(
            raw_sizes.astype(np.float32)[np.newaxis, ...],
            mult_factor=1.0 / (mult_factor[np.newaxis, ...] + 1e-6),
        ).squeeze(0)

        # GT box corners (8 corners of each box)
        try:
            import sys, os as _os
            _path = _os.path.abspath(
                _os.path.join(_os.path.dirname(__file__), "../../third_party/3detr")
            )
            if _path not in sys.path:
                sys.path.insert(0, _path)
            from utils.box_util import (
                flip_axis_to_camera_np,
                get_3d_box_batch_np,
            )
            box_center_upright = flip_axis_to_camera_np(box_centers[np.newaxis, ...])
            box_corners = get_3d_box_batch_np(
                raw_sizes.astype(np.float32)[np.newaxis, ...],
                raw_angles.astype(np.float32)[np.newaxis, ...],
                box_center_upright,
            ).squeeze(0)
        except Exception:
            box_corners = np.zeros((MAX_NUM_OBJ, 8, 3), dtype=np.float32)

        # Semantic class labels for GT boxes (already mapped to detection-class space).
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        if num_gt > 0:
            target_bboxes_semcls[:num_gt] = labels_raw[:num_gt]

        out = {
            "point_clouds": point_cloud.astype(np.float32),
            "gt_box_corners": box_corners.astype(np.float32),
            "gt_box_centers": box_centers.astype(np.float32),
            "gt_box_centers_normalized": box_centers_normalized.astype(np.float32),
            "gt_angle_class_label": angle_classes.astype(np.int64),
            "gt_angle_residual_label": angle_residuals.astype(np.float32),
            "gt_box_sem_cls_label": target_bboxes_semcls.astype(np.int64),
            "gt_box_present": target_bboxes_mask.astype(np.float32),
            "scan_idx": np.array(idx, dtype=np.int64),
            "pcl_color": pcl_color.astype(np.float32),
            "gt_box_sizes": raw_sizes.astype(np.float32),
            "gt_box_sizes_normalized": box_sizes_normalized.astype(np.float32),
            "gt_box_angles": raw_angles.astype(np.float32),
            "point_cloud_dims_min": point_cloud_dims_min.astype(np.float32),
            "point_cloud_dims_max": point_cloud_dims_max.astype(np.float32),
        }
        if segment is not None:
            out["segment"] = segment.astype(np.int64)
        return out
