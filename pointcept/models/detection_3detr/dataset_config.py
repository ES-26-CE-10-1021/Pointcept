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
