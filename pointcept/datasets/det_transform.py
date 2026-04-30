"""
Detection-aware augmentation transforms for oriented-bbox datasets.

These transforms operate on a raw sample dict produced by detection datasets
such as ``AgcoBBoxV1`` *before* the per-sample padding / angle-bin encoding /
GT-corner computation step. They keep point clouds and bounding boxes in sync,
which the segmentation-only transforms in ``transform.py`` do not.

Expected dict keys (any may be absent if the sample has no boxes):

    point_cloud           : (N, 3 or 4) float32 — XYZ in cols 0..3, optional
                            intensity in col 3
    gt_box_centers_raw    : (n, 3) float — pre-pad full-precision centers
    gt_box_sizes_raw      : (n, 3) float — full size (not half)
    gt_box_angles_raw     : (n,)   float — yaw in radians, wrapped to (-π, π]
    gt_box_labels_raw     : (n,)   int64

All transforms must tolerate ``n == 0``.
"""

import numpy as np

from .scannet_detection import _rotz
from .transform import TRANSFORMS


def _wrap_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@TRANSFORMS.register_module()
class RandomFlipDetection(object):
    """Independent X / Y mirror with matching yaw + center updates.

    X-flip negates the X coordinate of points and box centers and sends
    ``yaw → -yaw``. Y-flip mirrors on Y and sends ``yaw → π - yaw``. Both
    yaws are wrapped to (-π, π].
    """

    def __init__(self, p_x=0.5, p_y=0.5):
        self.p_x = float(p_x)
        self.p_y = float(p_y)

    def __call__(self, data_dict):
        if np.random.rand() < self.p_x:
            if "point_cloud" in data_dict:
                data_dict["point_cloud"][:, 0] *= -1
            if "gt_box_centers_raw" in data_dict and len(data_dict["gt_box_centers_raw"]) > 0:
                data_dict["gt_box_centers_raw"][:, 0] *= -1
                data_dict["gt_box_angles_raw"] = _wrap_pi(
                    -data_dict["gt_box_angles_raw"]
                )
        if np.random.rand() < self.p_y:
            if "point_cloud" in data_dict:
                data_dict["point_cloud"][:, 1] *= -1
            if "gt_box_centers_raw" in data_dict and len(data_dict["gt_box_centers_raw"]) > 0:
                data_dict["gt_box_centers_raw"][:, 1] *= -1
                data_dict["gt_box_angles_raw"] = _wrap_pi(
                    np.pi - data_dict["gt_box_angles_raw"]
                )
        return data_dict


@TRANSFORMS.register_module()
class RandomRotateZDetection(object):
    """Uniform rotation about the Z axis.

    ``angle_deg`` is the (low, high) range in degrees from which the rotation
    angle is uniformly sampled. Defaults match ScanNet/3DETR's ±5°.
    """

    def __init__(self, angle_deg=(-5.0, 5.0)):
        lo, hi = float(angle_deg[0]), float(angle_deg[1])
        assert lo <= hi, f"angle_deg must be (low, high); got {angle_deg}"
        self.angle_rad = (np.deg2rad(lo), np.deg2rad(hi))

    def __call__(self, data_dict):
        lo, hi = self.angle_rad
        theta = np.random.uniform(lo, hi)
        rot_mat = _rotz(theta).astype(np.float32)
        if "point_cloud" in data_dict:
            pc = data_dict["point_cloud"]
            pc[:, 0:3] = pc[:, 0:3] @ rot_mat.T
        if "gt_box_centers_raw" in data_dict and len(data_dict["gt_box_centers_raw"]) > 0:
            data_dict["gt_box_centers_raw"] = (
                data_dict["gt_box_centers_raw"] @ rot_mat.T
            )
            data_dict["gt_box_angles_raw"] = _wrap_pi(
                data_dict["gt_box_angles_raw"] + theta
            ).astype(np.float32)
        return data_dict


@TRANSFORMS.register_module()
class RandomScaleDetection(object):
    """Uniform isotropic scale on points + box centers (and optionally sizes)."""

    def __init__(self, scale=(0.9, 1.1), apply_to_sizes=True):
        lo, hi = float(scale[0]), float(scale[1])
        assert lo > 0 and lo <= hi, f"scale must be (low, high) with low > 0; got {scale}"
        self.scale = (lo, hi)
        self.apply_to_sizes = bool(apply_to_sizes)

    def __call__(self, data_dict):
        s = np.random.uniform(*self.scale)
        if "point_cloud" in data_dict:
            data_dict["point_cloud"][:, 0:3] *= s
        if "gt_box_centers_raw" in data_dict and len(data_dict["gt_box_centers_raw"]) > 0:
            data_dict["gt_box_centers_raw"] *= s
            if self.apply_to_sizes:
                data_dict["gt_box_sizes_raw"] *= s
        return data_dict


@TRANSFORMS.register_module()
class RandomJitterDetection(object):
    """Per-point Gaussian jitter on point XYZ. Boxes are not touched."""

    def __init__(self, sigma=0.005, clip=0.02):
        assert clip > 0, "clip must be > 0"
        self.sigma = float(sigma)
        self.clip = float(clip)

    def __call__(self, data_dict):
        if "point_cloud" in data_dict:
            pc = data_dict["point_cloud"]
            jitter = np.clip(
                self.sigma * np.random.randn(pc.shape[0], 3),
                -self.clip,
                self.clip,
            ).astype(pc.dtype)
            pc[:, 0:3] += jitter
        return data_dict


@TRANSFORMS.register_module()
class RandomCuboidDetection(object):
    """Crop a random cuboid sub-volume; drop boxes whose centers fall outside.

    Adapted from ``third_party/3detr/utils/random_cuboid.RandomCuboid`` so it
    can operate on this codebase's split center/size/angle/label arrays
    (rather than a single packed bbox tensor). Behaviour matches the reference
    implementation: 100 retry iterations, aspect-ratio gate, must keep at
    least ``min_points`` points and ≥1 surviving box (when boxes are present).
    On failure, the sample is returned unmodified.
    """

    def __init__(
        self,
        min_points=30000,
        aspect=0.8,
        min_crop=0.5,
        max_crop=1.0,
    ):
        assert 0 < min_crop <= max_crop <= 1.0, (
            f"min_crop / max_crop out of range: {min_crop}, {max_crop}"
        )
        self.min_points = int(min_points)
        self.aspect = float(aspect)
        self.min_crop = float(min_crop)
        self.max_crop = float(max_crop)

    @staticmethod
    def _check_aspect(crop_range, aspect_min):
        xy = np.min(crop_range[:2]) / np.max(crop_range[:2])
        xz = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
        yz = np.min(crop_range[1:]) / np.max(crop_range[1:])
        return (xy >= aspect_min) or (xz >= aspect_min) or (yz >= aspect_min)

    def __call__(self, data_dict):
        if "point_cloud" not in data_dict:
            return data_dict
        pc = data_dict["point_cloud"]
        if pc.shape[0] < self.min_points:
            return data_dict

        centers = data_dict.get("gt_box_centers_raw", np.zeros((0, 3)))
        has_boxes = len(centers) > 0

        range_xyz = np.max(pc[:, 0:3], axis=0) - np.min(pc[:, 0:3], axis=0)

        for _ in range(100):
            crop_range = self.min_crop + np.random.rand(3) * (
                self.max_crop - self.min_crop
            )
            if not self._check_aspect(crop_range, self.aspect):
                continue

            sample_center = pc[np.random.choice(pc.shape[0]), 0:3]
            half = range_xyz * crop_range / 2.0
            min_xyz = sample_center - half
            max_xyz = sample_center + half

            point_mask = np.all(pc[:, 0:3] >= min_xyz, axis=1) & np.all(
                pc[:, 0:3] <= max_xyz, axis=1
            )
            if int(point_mask.sum()) < self.min_points:
                continue

            if has_boxes:
                new_pc_min = np.min(pc[point_mask, 0:3], axis=0)
                new_pc_max = np.max(pc[point_mask, 0:3], axis=0)
                box_mask = np.all(centers >= new_pc_min, axis=1) & np.all(
                    centers <= new_pc_max, axis=1
                )
                if int(box_mask.sum()) == 0:
                    continue
            else:
                box_mask = None

            data_dict["point_cloud"] = pc[point_mask]
            if has_boxes:
                data_dict["gt_box_centers_raw"] = centers[box_mask]
                data_dict["gt_box_sizes_raw"] = data_dict["gt_box_sizes_raw"][box_mask]
                data_dict["gt_box_angles_raw"] = data_dict["gt_box_angles_raw"][box_mask]
                data_dict["gt_box_labels_raw"] = data_dict["gt_box_labels_raw"][box_mask]
            return data_dict

        return data_dict


@TRANSFORMS.register_module()
class PointSubsampleDetection(object):
    """Sub-sample (or repeat-sample) point cloud to exactly ``num_points``.

    3DETR requires a fixed point count downstream. ``AgcoBBoxV1`` will also
    enforce this as a safety net if the pipeline didn't include this
    transform, so adding it explicitly is recommended for clarity but not
    strictly required for correctness.
    """

    def __init__(self, num_points):
        self.num_points = int(num_points)

    def __call__(self, data_dict):
        if "point_cloud" not in data_dict:
            return data_dict
        pc = data_dict["point_cloud"]
        n = pc.shape[0]
        if n == self.num_points:
            return data_dict
        replace = n < self.num_points
        choices = np.random.choice(n, self.num_points, replace=replace)
        data_dict["point_cloud"] = pc[choices]
        return data_dict
