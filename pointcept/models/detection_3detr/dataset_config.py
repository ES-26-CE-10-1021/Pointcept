"""
3DETR Dataset Configuration

Adapted from third_party/3detr/datasets/scannet.py:ScannetDatasetConfig
Provides dataset metadata (class mappings, box parametrization) as a
Pointcept-registered module so it can be specified in configs.
"""

import numpy as np
import torch

from pointcept.models.builder import MODULES


def _setup_3detr_path():
    import sys
    import os

    path = os.path.join(os.path.dirname(__file__), "../../../third_party/3detr")
    path = os.path.abspath(path)
    if path not in sys.path:
        sys.path.insert(0, path)


_setup_3detr_path()

from utils.box_util import flip_axis_to_camera_np, flip_axis_to_camera_tensor
from utils.box_util import get_3d_box_batch_np, get_3d_box_batch_tensor


def flip_axis_to_lidar_np(pc):
    """Inverse of `flip_axis_to_camera_np`: [x, y, z] -> [x, z, -y]."""
    pc2 = np.asarray(pc).copy()
    pc2[..., 0] = pc[..., 0]
    pc2[..., 1] = pc[..., 2]
    pc2[..., 2] = -pc[..., 1]
    return pc2


def flip_axis_to_lidar_tensor(pc):
    """Inverse of `flip_axis_to_camera_tensor`: [x, y, z] -> [x, z, -y]."""
    pc2 = torch.clone(pc)
    pc2[..., 0] = pc[..., 0]
    pc2[..., 1] = pc[..., 2]
    pc2[..., 2] = -pc[..., 1]
    return pc2


def wrap_angle_np(angle):
    """Wrap radians to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


@MODULES.register_module()
class ScanNetDetectionConfig:
    """Dataset configuration for ScanNet 3D object detection (18 classes)."""

    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 64

        self.type2class = {
            "cabinet": 0,
            "bed": 1,
            "chair": 2,
            "sofa": 3,
            "table": 4,
            "door": 5,
            "window": 6,
            "bookshelf": 7,
            "picture": 8,
            "counter": 9,
            "desk": 10,
            "curtain": 11,
            "refrigerator": 12,
            "showercurtrain": 13,
            "toilet": 14,
            "sink": 15,
            "bathtub": 16,
            "garbagebin": 17,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }

        self.num_class_semseg = 20
        self.nyu40ids_semseg = np.array(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class_semseg = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids_semseg))
        }

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
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


@MODULES.register_module()
class AgcoBBoxConfig:
    """Dataset configuration for AGCO 3D object detection (oriented boxes).

    Mirrors third_party/3detr/datasets/sunrgbd.py:SunrgbdDatasetConfig — yaw
    encoded via (angle_class, residual) with `num_angle_bin` bins.

    Disk label scheme (upstream annotation tool):
        0: background (excluded from training targets)
        1: tractor
        2: harvester
        3: trailer
        4: car
        5: hopper

    Model class indices are derived from `included_classes` (default: all
    five — `("tractor", "harvester", "trailer", "car", "hopper")` →
    0..4 in that order). When `included_classes` is a subset, the kept
    names are reassigned 0..K-1 in the order given; pass the same tuple to
    `AgcoBBoxV1` so the dataset's disk-label remap matches.
    """

    _ALL_CLASSES = ("tractor", "harvester", "trailer", "car", "hopper")

    def __init__(
        self,
        num_angle_bin: int = 12,
        included_classes=None,
        max_num_obj: int = 64,
    ):
        self.num_angle_bin = int(num_angle_bin)
        self.max_num_obj = int(max_num_obj)

        included = tuple(
            included_classes if included_classes is not None else self._ALL_CLASSES
        )
        unknown = [n for n in included if n not in self._ALL_CLASSES]
        if unknown:
            raise ValueError(
                f"Unknown class names in included_classes: {unknown}. "
                f"Valid names: {list(self._ALL_CLASSES)}"
            )
        if len(set(included)) != len(included):
            raise ValueError(f"Duplicate entries in included_classes: {included}")

        self.included_classes = included
        self.type2class = {name: i for i, name in enumerate(included)}
        self.class2type = {v: k for k, v in self.type2class.items()}
        self.num_semcls = len(self.type2class)

    # -- angle encoding (SUN-RGBD-style) --------------------------------------

    def angle2class(self, angle):
        """Continuous angle (rad) → (class_id, residual_angle).
        class_id * (2π/N) + residual_angle == angle (mod 2π).
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle = pred_cls * angle_per_class + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle = pred_cls * angle_per_class + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2anglebatch(pred_cls, residual, to_label_format)

    # -- box corners ----------------------------------------------------------

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        return get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        return get_3d_box_batch_np(box_size, box_angle, box_center_upright)
