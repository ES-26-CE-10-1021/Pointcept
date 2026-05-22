"""
AGCO bounding-box detection dataset (AgcoBBoxV1) for 3DETR.

Each sample is one `(sensor, timestamp)` pair. Per-timestamp LiDAR scans and
per-timestamp flat bbox annotations are produced by an upstream preprocessing
step (see docs/plans/pytorch_dataset_gravity_integration.md in the
sensor_interface repo). Raw annotations live in the un-levelled world frame and
are levelled at load time via `R_level` from `gravity_align.npz`.

Boxes are oriented (yaw matters) and are encoded for 3DETR with SUN-RGBD-style
`(angle_class, residual)` labels — use with `AgcoBBoxConfig`, not
`ScanNetDetectionConfig`.

On-disk layout (per annotation root):

    <root>/
        gravity_align.npz                   # {"R_level": (3,3) float64}
        calibration.yml
        <sensor>/pointcloud_raw/coord/<ts>.npy      # (N, 3) float32
        <sensor>/pointcloud_raw/intensity/<ts>.npy  # (N,)   float32 (optional)
        <sensor>/annotations/<ts>.json                   # flat per-ts boxes
        <sensor>/global_transforms/<ts>.npy         # (4, 4) float64, sensor→global
        ...

Per-timestamp bboxes JSON:

    {"annotations": [
        {"translation": [x, y, z],
         "rotation": [qx, qy, qz, qw],  # scipy xyzw order
         "dimensions": [dx, dy, dz],    # full box size in metres
         "label": <int; disk scheme: 0=bg, 1=tractor, 2=harvester, 3=trailer, 4=car, 5=hopper>,
         "inliers": <int>,
         "is_visible": <bool>,
         "children": [...]},             # optional child annotations
        ...
    ]}

Split files live in `meta_data_dir/<name>_<split>.txt`, one annotation root per
line (relative to `root_dir`).
"""

import csv
import json
import os
import warnings
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as Rot
from torch.utils.data import Dataset

from .builder import DATASETS
from pointcept.models.detection_3detr.dataset_config import (
    _setup_3detr_path,
    wrap_angle_np,
)
from .scannet_detection import (
    MEAN_COLOR_RGB,  # noqa: F401 - kept for parity; may be used later
    _random_sampling,
)


# Canonical disk label scheme (upstream annotation tool):
#   0=background (excluded), 1=tractor, 2=harvester, 3=trailer, 4=car, 5=hopper
_NAME_TO_DISK_LABEL = {
    "tractor": 1,
    "harvester": 2,
    "trailer": 3,
    "car": 4,
    "hopper": 5,
}

# Default model class order (matches AgcoBBoxConfig default).
_DEFAULT_INCLUDED_CLASSES = ("tractor", "harvester", "trailer", "car", "hopper")


import csv

def load_lidar_to_frame_map(csv_path):
    lookup = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_ts = row["frame"]
            for sensor in [
                "rslidar",
                "ouster",
                "lslidar",
            ]:
                lidar_ts = row[sensor]
                lookup[(sensor, lidar_ts)] = frame_ts
    return lookup

import numpy as np


def project_fisheye_simple_radial(
    pts_cam: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    k: float,
    img_w: int,
    img_h: int,
):
    """
    Simple radial fisheye projection.

    Parameters
    ----------
    pts_cam : (N,3) float
        Points in camera frame.

    f : float
        Focal length.

    cx, cy : float
        Principal point.

    k : float
        Radial fisheye distortion coefficient.

    img_w, img_h : int
        Projection bounds.

    Returns
    -------
    uv : (N,2) float64
        Pixel coordinates.

    valid : (N,) bool
        Valid projection mask.
    """

    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]
    # Must be in front of camera
    in_front = Z > 1e-4
    # Radial distance
    r = np.sqrt(X**2 + Y**2)
    # Fisheye angle
    theta = np.arctan2(r, Z)
    # Simple radial distortion
    theta_d = theta * (1.0 + k * theta**2)
    # Avoid divide-by-zero
    scale = np.zeros_like(theta_d)
    nz = r > 1e-8
    scale[nz] = theta_d[nz] / r[nz]
    # Pixel projection
    u = f * scale * X + cx
    v = f * scale * Y + cy
    uv = np.stack(
        [u, v],
        axis=1,
    )
    # Image bounds
    in_bounds = (
        (u >= 0)
        & (u < img_w)
        & (v >= 0)
        & (v < img_h)
    )
    valid = in_front & in_bounds
    return uv.astype(np.float64), valid

def _build_disk_label_to_class(included_classes):
    """Build {disk_label: model_class_idx} from an ordered name tuple.

    Model class indices are assigned 0..K-1 in the order given by
    `included_classes`. Names not in the tuple are dropped at load time.
    """
    unknown = [n for n in included_classes if n not in _NAME_TO_DISK_LABEL]
    if unknown:
        raise ValueError(
            f"Unknown class names in included_classes: {unknown}. "
            f"Valid names: {sorted(_NAME_TO_DISK_LABEL.keys())}"
        )
    if len(set(included_classes)) != len(included_classes):
        raise ValueError(
            f"Duplicate entries in included_classes: {included_classes}"
        )
    return {
        _NAME_TO_DISK_LABEL[name]: i for i, name in enumerate(included_classes)
    }


def _load_r_level(path, require: bool):
    if os.path.isfile(path):
        data = np.load(path)
        if "R_level" not in data.files:
            raise KeyError(
                f"{path} does not contain an 'R_level' array "
                f"(keys: {list(data.files)})"
            )
        return Rot.from_matrix(np.asarray(data["R_level"], dtype=np.float64))
    if require:
        raise FileNotFoundError(
            f"Missing gravity_align.npz at {path}. Re-run the gravity-aware "
            "annotation tool, or pass require_gravity_align=False to fall back "
            "to identity (development-only)."
        )
    warnings.warn(
        f"gravity_align.npz not found at {path}; falling back to identity "
        "R_level. This means training data will be un-levelled and box yaw "
        "will absorb the sensor tilt. Use only for development.",
        stacklevel=2,
    )
    return Rot.identity()


def _load_t_rtk(calib_path: str, sensor: str, require: bool):
    """Load per-sensor T_rtk from calibration.yml.

    Returns (Rotation, translation_ndarray) or None when file is absent
    and require=False.  RigidTransform is not used — scipy 1.15.2 lacks it.
    """
    import yaml

    if os.path.isfile(calib_path):
        with open(calib_path, "r") as f:
            data = yaml.full_load(f)
        sensors_data = data.get("T_rtk_sensors", {})
        if sensor not in sensors_data:
            raise KeyError(
                f"{calib_path} has no entry for sensor '{sensor}' "
                f"(available: {list(sensors_data.keys())})"
            )
        T = np.asarray(sensors_data[sensor]["matrix"], dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(
                f"{calib_path}[{sensor}]['matrix'] must be 4×4, got {T.shape}"
            )
        rotation = Rot.from_matrix(T[:3, :3])
        translation = T[:3, 3]
        return rotation, translation

    if require:
        raise FileNotFoundError(
            f"Missing calibration.yml at {calib_path}. "
            "Re-run calibration, or pass require_calibration=False to skip."
        )
    warnings.warn(
        f"calibration.yml not found at {calib_path}; T_rtk will not be applied "
        f"for sensor '{sensor}'. Points remain in sensor frame.",
        stacklevel=2,
    )
    return None

def _load_t_rtk_cam(calib_path: str, require: bool):
    """Load per-sensor T_rtk from calibration.yml.

    Returns (Rotation, translation_ndarray) or None when file is absent
    and require=False.  RigidTransform is not used — scipy 1.15.2 lacks it.
    """
    import yaml

    if os.path.isfile(calib_path):
        with open(calib_path, "r") as f:
            data = yaml.full_load(f)
        
        T_rtk_cam = np.asarray(data["T_rtk_cam"]["matrix"], dtype=np.float64)

        return T_rtk_cam

    if require:
        raise FileNotFoundError(
            f"Missing calibration.yml at {calib_path}. "
            "Re-run calibration, or pass require_calibration=False to skip."
        )
    warnings.warn(
        f"calibration.yml not found at {calib_path}; T_rtk will not be applied ",
        stacklevel=2,
    )
    return None


def _load_cam_intrinsics(
    calib_path: str,
    cam: str,
):
    """
    Load camera intrinsics from calibration YAML.

    Returns
    -------
    dict
        {
            "f": float,
            "cx": float,
            "cy": float,
            "k": float,
            "width": int,
            "height": int,
        }
    """

    import os
    import yaml
    if os.path.isfile(calib_path):

        with open(calib_path, "r") as f:
            data = yaml.safe_load(f)

        cameras = data.get("cameras", {})
        if cam not in cameras:
            raise KeyError(
                f"Camera '{cam}' not found in calibration "
                f"(available: {list(cameras.keys())})"
            )

        cam_data = cameras[cam]

        return {
            "f": float(cam_data["f"]),
            "cx": float(cam_data["cx"]),
            "cy": float(cam_data["cy"]),
            "k": float(cam_data["k"]),
            "width": int(cam_data["width"]),
            "height": int(cam_data["height"]),
        }

    raise FileNotFoundError(
        f"Missing calibration.yml at {calib_path}"
    )



def _flip_axis_to_camera_np(points):
    _setup_3detr_path()
    from utils.box_util import flip_axis_to_camera_np

    return flip_axis_to_camera_np(points)


def _get_3d_box_batch_np(sizes, angles, centers_upright):
    _setup_3detr_path()
    from utils.box_util import get_3d_box_batch_np

    return get_3d_box_batch_np(sizes, angles, centers_upright)


def _shift_scale_points(pts, src_range, dst_range):
    scale = (dst_range[1] - dst_range[0]) / (src_range[1] - src_range[0] + 1e-6)
    bias = dst_range[0] - src_range[0] * scale
    return pts * scale + bias


@DATASETS.register_module()
class AgcoBBoxDinoV1(Dataset):
    """AGCO oriented-bbox detection dataset for 3DETR.

    Args:
        root_dir (str): directory containing annotation roots listed in the
            split files.
        meta_data_dir (str): directory containing `<split_prefix>_<split>.txt`.
        split (str): 'train' or 'val'.
        split_prefix (str): prefix of the split txt file. Default 'agco'.
        sensors (list[str]): LiDAR folders to enumerate samples from, e.g.
            ["lslidar"] or ["lslidar", "ouster"]. Each sensor contributes its
            own samples; clouds are NOT fused across sensors.
        num_points (int): points subsampled per scan. Enforced as a final
            safety-net even if the ``transform`` pipeline doesn't include
            ``PointSubsampleDetection``.
        use_intensity (bool): if True, append per-point intensity as an
            extra channel when intensity .npy is present; otherwise zeros.
        use_color (bool): if True, load per-point RGB from
            ``<sensor>/color/<ts>.npy`` (float32, shape ``(N, 3)``, values
            in ``[0, 1]``) and concatenate it to the point cloud directly
            after XYZ. Canonical channel order is ``[xyz, rgb, intensity]``.
        utonia_preprocess (bool): if True, right-pad ``point_clouds`` with
            zero-channels so the per-point feature width is 9
            ``[xyz, rgb=0, normal=0]``, matching the Utonia (PT-v3m3)
            checkpoint's ``in_channels=9``. Padding happens after augmentation
            and the safety-net subsample. Default False.
        transform (list[dict] | None): Pointcept-style augmentation pipeline.
            Each entry is a config dict for a transform registered in the
            ``TRANSFORMS`` registry; the detection-aware transforms in
            ``pointcept/datasets/det_transform.py`` (``RandomFlipDetection``,
            ``RandomRotateZDetection``, ``RandomScaleDetection``,
            ``RandomJitterDetection``, ``RandomCuboidDetection``,
            ``PointSubsampleDetection``) are the supported set. ``None`` /
            ``[]`` disables augmentation; a final-size subsample still runs.
        require_gravity_align (bool): if True (default), missing
            `gravity_align.npz` raises; if False, falls back to identity with
            a warning.
        residual_rpy_warn_deg (float): warn when levelled box pitch/roll
            magnitude exceeds this threshold (only checked when
            `apply_r_level_to_boxes=True`).
        loop (int): dataset-length multiplier (Pointcept convention).
        dataset_config: ignored; present for symmetry with other detection
            datasets that accept a config object. The config used by the
            detector is supplied separately in the model config.
        min_inliers (int | dict[str, int]): minimum `inliers` count for a parent
            bbox to be kept. Children inherit visibility from their parent.
            Pass an ``int`` for a single global threshold, or a
            ``{sensor: int}`` dict to use a different threshold per sensor;
            when a dict is passed it must contain an entry for every sensor in
            ``sensors``.
        included_classes (tuple[str] | None): ordered subset of class names to
            train/eval on. Boxes for any class not listed here are dropped at
            load time, and the kept names are remapped to model class indices
            ``0..K-1`` in the order given. ``None`` (default) keeps all five
            classes in the canonical order
            ``("tractor", "harvester", "trailer", "car", "hopper")``. When
            customising, pass the same tuple to ``AgcoBBoxConfig`` so the
            detector head's ``num_semcls`` and ``type2class`` agree.
        fixed_pc_dims (dict | None): if a dict, use fixed per-sensor bounds for
            normalizing box centers/sizes (and for the
            ``point_cloud_dims_min/max`` returned to the model). Format:
            ``{sensor: {"min": [x, y, z], "max": [x, y, z]}}`` with an entry
            for every sensor in ``sensors``. ``None`` (default) falls back to
            per-sample min/max of the post-augmentation point cloud — this is
            unstable across samples for outdoor LiDAR where extents vary.
        nonempty_oversample (float): if >1.0, duplicate samples that contain at
            least one kept GT box (after ``min_inliers`` / ``included_classes``
            filtering). Empty samples are kept once. The multiplier may be
            fractional: ``2.5`` means each non-empty sample appears 2× plus an
            extra copy for the first 50% (deterministic, by enumeration order).
            Default ``1.0`` (no oversampling). Costs one extra JSON read per
            sample at ``__init__`` time.
        apply_r_level_to_points (bool): if True, rotate point cloud XYZ by
            `R_level`. Default False — matches the visualizer flag default.
        apply_r_level_to_boxes (bool): if True, rotate box centers and
            compose box quaternions with `R_level`. Default False.
        apply_t_rtk (bool): if True, apply per-sensor T_rtk calibration (sensor →
            RTK/world frame) to point cloud XYZ before R_level. Default False.
            When ``require_calibration=False`` and ``calibration.yml`` is missing,
            the map entry is ``None`` and the transform is silently skipped for
            that root/sensor.
        require_calibration (bool): if True and calibration.yml is missing, raise;
            if False (default), warn and skip T_rtk for that root/sensor.
        apply_r_global (bool): if True, apply the pitch and roll components of the
            per-timestamp global transform (`<sensor>/global_transforms/<ts>.npy`,
            a 4×4 matrix) to both point cloud XYZ and box centers/orientations.
            The 3×3 rotation is decomposed via ZYX Euler angles; yaw is discarded
            so the scene retains its original horizontal heading. Translation is
            ignored. Applied to points after T_rtk. Default False.
    """

    def __init__(
        self,
        root_dir,
        meta_data_dir,
        split="train",
        split_prefix="agco",
        sensors=("lslidar",),
        num_points=80000,
        use_intensity=False,
        use_color=False,
        utonia_preprocess=False,
        transform=None,
        require_gravity_align=True,
        residual_rpy_warn_deg=2.0,
        loop=1,
        dataset_config=None,
        min_inliers=67,
        included_classes=None,
        nonempty_oversample: float = 1.0,
        fixed_pc_dims=None,
        apply_r_level_to_points=False,
        apply_r_level_to_boxes=False,
        apply_t_rtk: bool = False,
        require_calibration: bool = False,
        apply_r_global: bool = False,
        deterministic_debug: bool = False,
        deterministic_seed: int = 0,
        debug_roundtrip_check: bool = False,
        load_segment: bool = False,
        segment_subdir: str = "segment",
        seg_label_map: dict | None = None,
        max_num_obj: int = 64,
        sample_allowlist_file: str = None,
        sample_allowlist_mode: str = "exact",
        use_dino=False,
        dino_camera="jai_left",
        dino_subdir="dino_patch_h16plus_full_res",
        dino_feature_dim=2048,
    ):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert len(sensors) > 0, "At least one sensor must be specified"
        self.root_dir = root_dir
        self.meta_data_dir = meta_data_dir
        self.split = split
        self.split_prefix = split_prefix
        self.sensors = tuple(sensors)
        self.num_points = int(num_points)
        self.use_intensity = bool(use_intensity)
        self.use_color = bool(use_color)
        self.utonia_preprocess = bool(utonia_preprocess)
        from .transform import Compose
        self.transform = Compose(transform or [])
        self.require_gravity_align = bool(require_gravity_align)
        self.residual_rpy_warn_rad = np.deg2rad(float(residual_rpy_warn_deg))
        self.loop = int(loop)
        self.max_num_obj = int(max_num_obj)
        if isinstance(min_inliers, dict):
            missing = set(self.sensors) - set(min_inliers.keys())
            assert not missing, (
                f"min_inliers dict is missing entries for sensors: {sorted(missing)}"
            )
            self.min_inliers = {s: int(min_inliers[s]) for s in self.sensors}
        else:
            self.min_inliers = int(min_inliers)
        self.included_classes = tuple(
            included_classes if included_classes is not None
            else _DEFAULT_INCLUDED_CLASSES
        )
        self.disk_label_to_class = _build_disk_label_to_class(self.included_classes)
        self.nonempty_oversample = float(nonempty_oversample)
        assert self.nonempty_oversample >= 1.0, (
            f"nonempty_oversample must be >= 1.0, got {self.nonempty_oversample}"
        )

        # Per-sensor fixed normalization bounds. None -> use per-sample min/max
        # (legacy behavior). Otherwise a dict {sensor: {"min": [..], "max": [..]}}
        # with an entry for every sensor in `self.sensors`.
        self.fixed_pc_dims = fixed_pc_dims
        self._fixed_pc_dims_arr = None
        if self.fixed_pc_dims is not None:
            assert isinstance(self.fixed_pc_dims, dict), (
                f"fixed_pc_dims must be a dict or None, got {type(self.fixed_pc_dims)}"
            )
            missing = set(self.sensors) - set(self.fixed_pc_dims.keys())
            assert not missing, (
                f"fixed_pc_dims is missing entries for sensors: {sorted(missing)}"
            )
            self._fixed_pc_dims_arr = {}
            for s in self.sensors:
                entry = self.fixed_pc_dims[s]
                pc_min = np.asarray(entry["min"], dtype=np.float32)
                pc_max = np.asarray(entry["max"], dtype=np.float32)
                assert pc_min.shape == (3,) and pc_max.shape == (3,), (
                    f"fixed_pc_dims[{s}] min/max must be length-3, got "
                    f"{pc_min.shape} and {pc_max.shape}"
                )
                assert np.all(pc_max > pc_min), (
                    f"fixed_pc_dims[{s}] requires max > min element-wise; got "
                    f"min={pc_min.tolist()} max={pc_max.tolist()}"
                )
                self._fixed_pc_dims_arr[s] = (pc_min, pc_max)
        self.apply_r_level_to_points = bool(apply_r_level_to_points)
        self.apply_r_level_to_boxes = bool(apply_r_level_to_boxes)
        self.apply_t_rtk = bool(apply_t_rtk)
        self.require_calibration = bool(require_calibration)
        self.apply_r_global = bool(apply_r_global)
        self.deterministic_debug = bool(deterministic_debug)
        self.deterministic_seed = int(deterministic_seed)
        self.debug_roundtrip_check = bool(debug_roundtrip_check)
        self.load_segment = bool(load_segment)
        self.segment_subdir = str(segment_subdir)
        self.seg_label_map = dict(seg_label_map) if seg_label_map is not None else None
        self.sample_allowlist_file = sample_allowlist_file
        self.sample_allowlist_mode = str(sample_allowlist_mode)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        if self.sample_allowlist_mode != "exact":
            raise ValueError(
                f"Unsupported sample_allowlist_mode='{self.sample_allowlist_mode}'. "
                "Only 'exact' is supported."
            )
        self.use_dino = use_dino 
        self.dino_camera = dino_camera
        self.dino_subdir = dino_subdir  
        
        self.frame_time_lookup = None



        split_file = os.path.join(meta_data_dir, f"{split_prefix}_{split}.txt")
        with open(split_file, "r") as f:
            roots = [line.strip() for line in f if line.strip()]

        self.roots = []        # absolute paths
        self.r_levels = []     # scipy Rotation, one per root
        self.lookup_tables = {}
        for r in roots:
            abs_root = r if os.path.isabs(r) else os.path.join(root_dir, r)
            if not os.path.isdir(abs_root):
                raise FileNotFoundError(
                    f"Annotation root does not exist: {abs_root}"
                )
            self.roots.append(abs_root)
            self.r_levels.append(
                _load_r_level(
                    os.path.join(abs_root, "gravity_align.npz"),
                    require=self.require_gravity_align,
                )
            )
            if self.use_dino:
                self.lookup_tables[abs_root] = load_lidar_to_frame_map(os.path.join(abs_root, "lidar_timetable.csv"))

        # Build per-(root, sensor) T_rtk map. Loaded once; reused across samples.
        self._t_rtk_map: dict = {}
        self._t_rtk_cam = {}
        self.camera_intrinsics = {}
        for ridx, abs_root in enumerate(self.roots):
            calib_path = os.path.join(abs_root, "calibration.yml")
            for sensor in self.sensors:
                result = _load_t_rtk(
                    calib_path, sensor, require=self.require_calibration
                )
                self._t_rtk_map[(ridx, sensor)] = result
            self._t_rtk_cam[ridx] = _load_t_rtk_cam(calib_path, require=True)
            self.camera_intrinsics[ridx] = _load_cam_intrinsics(calib_path, self.dino_camera)


        # Flatten (root_idx, sensor, ts) samples.
        self.samples = []
        for ridx, abs_root in enumerate(self.roots):
            for sensor in self.sensors:
                coord_dir = os.path.join(
                    abs_root, sensor, "pointcloud_raw", "coord"
                )
                bbox_dir = os.path.join(abs_root, sensor, "annotations")
                if not os.path.isdir(coord_dir) or not os.path.isdir(bbox_dir):
                    continue
                # Enumerate annotations first, then verify the matching coord
                # exists. Avoids silently skipping annotated samples when the
                # coord dir contains extra timestamps.
                timestamps = sorted(
                    os.path.splitext(f)[0]
                    for f in os.listdir(bbox_dir)
                    if f.endswith(".json")
                )
                for ts in timestamps:
                    if os.path.isfile(
                        os.path.join(coord_dir, f"{ts}.npy")
                    ):
                        self.samples.append((ridx, sensor, ts))

        n_total = len(self.samples)

        if self.sample_allowlist_file:
            self.samples = self._filter_samples_with_allowlist(self.samples)
            print(
                f"[AgcoBBoxV1] {split}: allowlist kept {len(self.samples)} / {n_total} samples"
            )
            n_total = len(self.samples)

        if self.nonempty_oversample > 1.0 and n_total > 0:
            empty_samples = []
            nonempty_samples = []
            for s in self.samples:
                ridx, sensor, ts = s
                if self._count_kept_boxes(self.roots[ridx], sensor, ts) > 0:
                    nonempty_samples.append(s)
                else:
                    empty_samples.append(s)
            base = int(np.floor(self.nonempty_oversample))
            frac = self.nonempty_oversample - base
            extras = int(round(frac * len(nonempty_samples)))
            duplicated = nonempty_samples * base + nonempty_samples[:extras]
            self.samples = empty_samples + duplicated
            print(
                f"[AgcoBBoxV1] {split}: oversample={self.nonempty_oversample:g} "
                f"-> {len(empty_samples)} empty + {len(nonempty_samples)} unique "
                f"non-empty (×{len(duplicated)/max(1,len(nonempty_samples)):.2f}) "
                f"= {len(self.samples)} samples (was {n_total})"
            )
        else:
            print(
                f"[AgcoBBoxV1] {split}: {n_total} samples across "
                f"{len(self.roots)} roots, sensors={list(self.sensors)}"
            )

    def __len__(self):
        return len(self.samples) * self.loop

    def _normalize_root_path(self, root_str: str) -> str:
        root_str = root_str.strip()
        if not root_str:
            return root_str
        p = Path(root_str)
        if not p.is_absolute():
            p = Path(self.root_dir) / p
        return str(p.resolve())

    def _load_sample_allowlist(self):
        path = Path(self.sample_allowlist_file)
        if not path.is_absolute():
            path = Path(self.meta_data_dir) / path
        if not path.is_file():
            raise FileNotFoundError(
                f"sample_allowlist_file not found: {path}"
            )

        allow = set()
        malformed = 0
        with open(path, "r") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [x.strip() for x in line.split(",")]
                if len(parts) != 3:
                    malformed += 1
                    continue
                root_str, sensor, ts = parts
                allow.add((self._normalize_root_path(root_str), sensor, ts))

        if malformed > 0:
            warnings.warn(
                f"{path}: ignored {malformed} malformed allowlist lines; expected '<root>,<sensor>,<timestamp>'",
                stacklevel=2,
            )

        if len(allow) == 0:
            raise ValueError(
                f"sample_allowlist_file {path} produced an empty allowlist"
            )
        return allow, str(path)
    
    def _project_dino_features(
        self,
        pts_xyz,
        sparse_dino,
        sensor,
        ridx,
    ):

        sparse_indices = sparse_dino["flat_indices"]
        sparse_features = sparse_dino["patch_features"]

        Ht, Wt = map(
            int,
            sparse_dino["grid_shape"],
        )

        patch_size = 16

        intr = self.camera_intrinsics[ridx]

        f = intr["f"]
        cx = intr["cx"]
        cy = intr["cy"]
        k = intr["k"]

        f_patch = f / patch_size
        cx_patch = cx / patch_size
        cy_patch = cy / patch_size

        t_rtk = self._t_rtk_map[(ridx, sensor)]

        if t_rtk is None:
            return np.zeros(
                (
                    pts_xyz.shape[0],
                    sparse_features.shape[1],
                ),
                dtype=np.float32,
            )

        t_rot, t_trans = t_rtk

        T_rtk_sensor = np.eye(4)

        T_rtk_sensor[:3, :3] = t_rot.as_matrix()
        T_rtk_sensor[:3, 3] = t_trans

        T_cam_sensor = (
            np.linalg.inv(self._t_rtk_cam[ridx])
            @ T_rtk_sensor
        )

        pts_cam = (
            pts_xyz @ T_cam_sensor[:3, :3].T
            + T_cam_sensor[:3, 3]
        )

        uv_patch, valid = (
            project_fisheye_simple_radial(
                pts_cam,
                f=f_patch,
                cx=cx_patch,
                cy=cy_patch,
                k=k,
                img_w=Wt,
                img_h=Ht,
            )
        )

        dino_feats = np.zeros(
            (
                pts_xyz.shape[0],
                sparse_features.shape[1],
            ),
            dtype=np.float32,
        )

        if not np.any(valid):
            return dino_feats

        px = uv_patch[valid, 0].astype(np.int32)
        py = uv_patch[valid, 1].astype(np.int32)

        projected_flat = py * Wt + px

        lookup_pos = np.searchsorted(
            sparse_indices,
            projected_flat,
        )

        inside = lookup_pos < len(sparse_indices)

        matched = np.zeros_like(inside)

        matched[inside] = (
            sparse_indices[lookup_pos[inside]]
            == projected_flat[inside]
        )

        valid_idx = np.where(valid)[0]

        matched_idx = valid_idx[matched]

        dino_feats[matched_idx] = (
            sparse_features[
                lookup_pos[matched]
            ]
        )

        return dino_feats.astype(np.float32)


    def _filter_samples_with_allowlist(self, samples):
        allow, allow_path = self._load_sample_allowlist()

        filtered = []
        seen = set()
        for ridx, sensor, ts in samples:
            key = (str(Path(self.roots[ridx]).resolve()), sensor, ts)
            if key in allow:
                filtered.append((ridx, sensor, ts))
                seen.add(key)

        missing = len(allow - seen)
        if missing > 0:
            warnings.warn(
                f"{allow_path}: {missing} allowlist entries did not match any sample",
                stacklevel=2,
            )

        if len(filtered) == 0:
            raise ValueError(
                f"No samples matched allowlist in {allow_path}. "
                "Check root paths, sensor names, and timestamps."
            )
        return filtered

    # ------------------------------------------------------------------

    def _load_scan(self, abs_root, sensor, ts):
        coord_path = os.path.join(
            abs_root, sensor, "pointcloud_raw", "coord", f"{ts}.npy"
        )
        pts = np.load(coord_path).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(
                f"Unexpected point cloud shape {pts.shape} at {coord_path}"
            )
        pts = pts[:, :3]
        # Some sensors (e.g. Ouster) store an organized grid where invalid
        # returns are encoded as zero-range points. Drop those so the cloud
        # only contains real returns and lines up with the segment / intensity
        # files (which are valid-only).
        valid_mask = np.any(pts != 0, axis=1)
        pts = pts[valid_mask]
        
        if self.use_color:
            color_path = os.path.join(
                abs_root, sensor, "color", f"{ts}.npy"
            )
            rgb = np.load(color_path).astype(np.float32)
            if rgb.ndim != 2 or rgb.shape != (pts.shape[0], 3):
                raise ValueError(
                    f"Unexpected color shape {rgb.shape} at {color_path}; "
                    f"expected ({pts.shape[0]}, 3)"
                )
            pts = np.concatenate([pts, rgb], axis=1)
            
        if self.use_intensity:
            intensity_path = os.path.join(
                abs_root, sensor, "pointcloud_raw", "intensity", f"{ts}.npy"
            )
            if os.path.isfile(intensity_path):
                intensity = np.load(intensity_path).astype(np.float32)
                if intensity.shape[0] == valid_mask.shape[0]:
                    intensity = intensity[valid_mask]
                pts = np.concatenate([pts, intensity[:, None]], axis=1)
            else:
                pts = np.concatenate(
                    [pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=1
                )
        return pts

    def _load_dino(self, abs_root, frame_timestamp):
        dino_path = os.path.join(abs_root, self.dino_camera, self.dino_subdir, f"{frame_timestamp}.npz") 
        dino_patch_feat = np.load(dino_path, allow_pickle=True)
        
        return dino_patch_feat 



    def _count_kept_boxes(self, abs_root, sensor, ts):
        path = os.path.join(abs_root, sensor, "annotations", f"{ts}.json")
        with open(path, "r") as f:
            data = json.load(f)
        threshold = (
            self.min_inliers[sensor]
            if isinstance(self.min_inliers, dict)
            else self.min_inliers
        )
        kept = 0
        for b in data.get("annotations", []):
            if not (b.get("is_visible", True) and b.get("inliers", 0) >= threshold):
                continue
            if int(b.get("label", -1)) in self.disk_label_to_class:
                kept += 1
            for c in b.get("children", []):
                if int(c.get("label", -1)) in self.disk_label_to_class:
                    kept += 1
        return kept

    def _load_boxes(self, abs_root, sensor, ts):
        path = os.path.join(abs_root, sensor, "annotations", f"{ts}.json")
        with open(path, "r") as f:
            data = json.load(f)
        raw = data.get("annotations", [])

        threshold = (
            self.min_inliers[sensor]
            if isinstance(self.min_inliers, dict)
            else self.min_inliers
        )

        # Keep parents that are visible AND have enough inliers; children
        # inherit the parent's visibility.
        filtered = []
        for b in raw:
            if b.get("is_visible", True) and b.get("inliers", 0) >= threshold:
                filtered.append(b)
                filtered.extend(b.get("children", []))

        # Remap disk labels to model class indices; drop background (disk 0),
        # any unrecognised labels, and any classes not in `included_classes`.
        filtered = [b for b in filtered if int(b.get("label", -1)) in self.disk_label_to_class]

        n = len(filtered)
        centers = np.zeros((n, 3), dtype=np.float64)
        sizes = np.zeros((n, 3), dtype=np.float32)
        quats = np.zeros((n, 4), dtype=np.float64)
        labels = np.zeros((n,), dtype=np.int64)
        for i, b in enumerate(filtered):
            centers[i] = b.get("translation")
            sizes[i] = b.get("dimensions")
            quats[i] = b.get("rotation")
            labels[i] = self.disk_label_to_class[int(b["label"])]
        return centers, sizes, quats, labels

    def __getitem__(self, idx):
        ridx, sensor, ts = self.samples[idx % len(self.samples)]
        abs_root = self.roots[ridx]
        R = self.r_levels[ridx]

        point_cloud = self._load_scan(abs_root, sensor, ts)
        segment = None
        if self.load_segment:
            seg_path = os.path.join(
                abs_root, sensor, self.segment_subdir, f"{ts}.npy"
            )
            if not os.path.exists(seg_path):
                raise FileNotFoundError(
                    f"AgcoBBoxV1.load_segment=True but segment file is missing: "
                    f"{seg_path}"
                )
            segment = np.load(seg_path).astype(np.int64)
            if segment.shape[0] != point_cloud.shape[0]:
                raise ValueError(
                    f"Segment label count ({segment.shape[0]}) does not match "
                    f"point count ({point_cloud.shape[0]}) for {seg_path}."
                )
            if self.seg_label_map is not None:
                for src, dst in self.seg_label_map.items():
                    segment[segment == src] = dst
        centers_raw, sizes_raw, quats_xyzw, labels_raw = self._load_boxes(
            abs_root, sensor, ts
        )
        projected_dino = None
        if self.use_dino:
            frame_timestamp = self.lookup_tables[abs_root][sensor, ts] 
            dino_patch_feat = self._load_dino(abs_root, frame_timestamp)
            sparse_dino = dino_patch_feat["features"].item()

            projected_dino = self._project_dino_features(point_cloud[:,:3], sparse_dino, sensor, ridx)
            
            # self.debug_visualize_projected_dino(point_cloud[:,:3], sparse_dino, sensor, ridx)
        

        

        # --- Points: sensor → RTK frame (T_rtk, points only) ---
        if self.apply_t_rtk:
            t_rtk = self._t_rtk_map.get((ridx, sensor))
            if t_rtk is not None:
                t_rot, t_trans = t_rtk
                point_cloud[:, 0:3] = (
                    t_rot.apply(point_cloud[:, 0:3]) + t_trans
                ).astype(np.float32)

        # --- Points + boxes: RTK → global frame (R_global_mat rotation, per-timestamp) ---
        R_global = None
        if self.apply_r_global:
            global_transform_path = os.path.join(
                abs_root, sensor, "global_transforms", f"{ts}.npy"
            )
            R_global_mat = np.load(global_transform_path).astype(np.float64)
            if R_global_mat.shape != (4, 4):
                raise ValueError(
                    f"Global transform at {global_transform_path} must be 4×4, "
                    f"got {R_global_mat.shape}"
                )
            # Extract only pitch and roll; yaw (heading) is discarded so the
            # scene stays in its original horizontal orientation.
            _angles = Rot.from_matrix(R_global_mat[:3, :3]).as_euler("ZYX")
            R_global = Rot.from_euler("ZYX", [0.0, _angles[1], _angles[2]])
            point_cloud[:, 0:3] = R_global.apply(point_cloud[:, 0:3]).astype(
                np.float32
            )

        # --- Points: global → levelled frame (R_level) ---
        if self.apply_r_level_to_points:
            point_cloud[:, 0:3] = R.apply(point_cloud[:, 0:3]).astype(np.float32)

        n_boxes = centers_raw.shape[0]
        if n_boxes > 0:
            q_boxes = Rot.from_quat(quats_xyzw)
            if R_global is not None:
                centers_raw = R_global.apply(centers_raw)
                q_boxes = R_global * q_boxes
            if self.apply_r_level_to_boxes:
                centers_raw = R.apply(centers_raw).astype(np.float32)
                q_boxes = R * q_boxes
            else:
                centers_raw = centers_raw.astype(np.float32)
            q_level = q_boxes.as_quat()
            rpy = Rot.from_quat(q_level).as_euler("xyz")
            yaws_raw = wrap_angle_np(rpy[:, 2]).astype(np.float32)
            if (
                self.apply_r_level_to_boxes
                and np.max(np.abs(rpy[:, :2])) > self.residual_rpy_warn_rad
            ):
                warnings.warn(
                    f"[AgcoBBoxV1] large residual pitch/roll after levelling "
                    f"in root={os.path.basename(abs_root)} ts={ts}: "
                    f"max={np.degrees(np.max(np.abs(rpy[:, :2]))):.2f} deg",
                    stacklevel=2,
                )
        else:
            centers_raw = np.zeros((0, 3), dtype=np.float32)
            yaws_raw = np.zeros((0,), dtype=np.float32)
        sizes_raw = sizes_raw.astype(np.float32)

        # --- Augmentation pipeline (config-driven) ---
        data_dict = {
            "point_cloud": point_cloud,
            "gt_box_centers_raw": centers_raw,
            "gt_box_sizes_raw": sizes_raw,
            "gt_box_angles_raw": yaws_raw,
            "gt_box_labels_raw": labels_raw,
            "sensor": sensor,
            "sample_index": int(idx % len(self.samples)),
        }
        if segment is not None:
            data_dict["segment"] = segment
        if projected_dino is not None: 
            data_dict["dino_feat"] = projected_dino
        data_dict = self.transform(data_dict)
        point_cloud = data_dict["point_cloud"]
        segment = data_dict.get("segment")  # None when load_segment is False
        projected_dino = data_dict.get("dino_feat")
        centers_raw = data_dict["gt_box_centers_raw"]
        sizes_raw = data_dict["gt_box_sizes_raw"]
        yaws_raw = data_dict["gt_box_angles_raw"]
        labels_raw = data_dict["gt_box_labels_raw"]

        # --- Safety-net subsample (idempotent if pipeline already did it) ---
        if point_cloud.shape[0] != self.num_points:
            if self.deterministic_debug:
                rng = np.random.default_rng(self.deterministic_seed + int(idx))
                if point_cloud.shape[0] >= self.num_points:
                    choices = rng.choice(point_cloud.shape[0], self.num_points, replace=False)
                else:
                    choices = rng.choice(point_cloud.shape[0], self.num_points, replace=True)
                point_cloud = point_cloud[choices]
            else:
                point_cloud, choices = _random_sampling(point_cloud, self.num_points)
            if segment is not None:
                segment = segment[choices]
        point_cloud = point_cloud.astype(np.float32)

        # Pad to 9 feature channels for the Utonia (PT-v3m3) pre-encoder,
        # which was pretrained with in_channels=9 ([xyz, rgb, normal]).
        if self.utonia_preprocess:
            n_pad = 9 - point_cloud.shape[1]
            if n_pad < 0:
                raise ValueError(
                    f"utonia_preprocess: point_cloud already has "
                    f"{point_cloud.shape[1]} channels (>9)."
                )
            if n_pad > 0:
                point_cloud = np.concatenate(
                    [point_cloud, np.zeros((point_cloud.shape[0], n_pad), dtype=np.float32)],
                    axis=1,
                )

        # --- Pad to MAX_NUM_OBJ (cap to MAX after augmentation) ---
        M = self.max_num_obj
        num_gt = min(centers_raw.shape[0], M)
        gt_centers = np.zeros((M, 3), dtype=np.float32)
        gt_sizes = np.zeros((M, 3), dtype=np.float32)
        gt_angles = np.zeros((M,), dtype=np.float32)
        gt_sem_cls = np.zeros((M,), dtype=np.int64)
        gt_present = np.zeros((M,), dtype=np.float32)
        gt_angle_cls = np.zeros((M,), dtype=np.int64)
        gt_angle_res = np.zeros((M,), dtype=np.float32)
        gt_centers[:num_gt] = centers_raw[:num_gt]
        gt_sizes[:num_gt] = sizes_raw[:num_gt]
        gt_angles[:num_gt] = yaws_raw[:num_gt]
        gt_sem_cls[:num_gt] = labels_raw[:num_gt]
        gt_present[:num_gt] = 1.0

        # Angle-bin encoding (SUN-RGBD-style, via AgcoBBoxConfig).
        from pointcept.models.detection_3detr.dataset_config import AgcoBBoxConfig

        cfg = AgcoBBoxConfig()  # cheap; no state
        for i in range(num_gt):
            cls_id, res = cfg.angle2class(float(yaws_raw[i]))
            gt_angle_cls[i] = cls_id
            gt_angle_res[i] = res

        # Normalized centers / sizes in the point-cloud bbox.
        if self._fixed_pc_dims_arr is not None:
            point_cloud_dims_min, point_cloud_dims_max = self._fixed_pc_dims_arr[sensor]
            point_cloud_dims_min = point_cloud_dims_min.copy()
            point_cloud_dims_max = point_cloud_dims_max.copy()
        else:
            point_cloud_dims_min = point_cloud[:, :3].min(axis=0).astype(np.float32)
            point_cloud_dims_max = point_cloud[:, :3].max(axis=0).astype(np.float32)
        mult_factor = point_cloud_dims_max - point_cloud_dims_min

        box_centers_normalized = _shift_scale_points(
            gt_centers[np.newaxis, ...],
            src_range=[
                point_cloud_dims_min[np.newaxis, ...],
                point_cloud_dims_max[np.newaxis, ...],
            ],
            dst_range=self.center_normalizing_range,
        ).squeeze(0).astype(np.float32)
        box_centers_normalized = box_centers_normalized * gt_present[..., np.newaxis]

        box_sizes_normalized = (
            gt_sizes / (mult_factor[np.newaxis, :] + 1e-6)
        ).astype(np.float32)

        # GT corners (rotated, using yaw).
        centers_upright = _flip_axis_to_camera_np(gt_centers[np.newaxis, ...])
        box_corners = _get_3d_box_batch_np(
            gt_sizes[np.newaxis, ...],
            gt_angles[np.newaxis, ...],
            centers_upright,
        ).squeeze(0).astype(np.float32)

        roundtrip_diag = {
            "enabled": bool(self.debug_roundtrip_check),
            "num_gt": int(num_gt),
            "center_l2_mean": float("nan"),
            "center_l2_max": float("nan"),
        }
        if self.debug_roundtrip_check and num_gt > 0:
            centers_cam = _flip_axis_to_camera_np(gt_centers[:num_gt][np.newaxis, ...]).squeeze(0)
            centers_back = np.empty_like(centers_cam)
            centers_back[:, 0] = centers_cam[:, 0]
            centers_back[:, 1] = centers_cam[:, 2]
            centers_back[:, 2] = -centers_cam[:, 1]
            center_err = np.linalg.norm(centers_back - gt_centers[:num_gt], axis=1)
            roundtrip_diag["center_l2_mean"] = float(np.mean(center_err))
            roundtrip_diag["center_l2_max"] = float(np.max(center_err))

        out = {
            "point_clouds": point_cloud,
            "point_cloud_dims_min": point_cloud_dims_min,
            "point_cloud_dims_max": point_cloud_dims_max,
            "gt_box_centers": gt_centers,
            "gt_box_centers_normalized": box_centers_normalized,
            "gt_box_sizes": gt_sizes,
            "gt_box_sizes_normalized": box_sizes_normalized,
            "gt_box_angles": gt_angles,
            "gt_angle_class_label": gt_angle_cls,
            "gt_angle_residual_label": gt_angle_res,
            "gt_box_sem_cls_label": gt_sem_cls,
            "gt_box_present": gt_present,
            "gt_box_corners": box_corners,
            "scan_idx": np.array(idx, dtype=np.int64),
            "frame_id": "agco_lidar_canonical",
            "frame_meta": {
                "apply_t_rtk": self.apply_t_rtk,
                "apply_r_global": self.apply_r_global,
                "apply_r_level_to_points": self.apply_r_level_to_points,
                "apply_r_level_to_boxes": self.apply_r_level_to_boxes,
                "sensor": sensor,
                "timestamp": ts,
                "root": os.path.basename(abs_root),
                "deterministic_debug": self.deterministic_debug,
            },
            "roundtrip_diag": roundtrip_diag,
        }
        if segment is not None:
            out["segment"] = segment.astype(np.int64)
        if projected_dino is not None:
            assert len(projected_dino) == len(point_cloud), f"projected dino with length {len(projected_dino)} is not equal to point cloud length of {len(point_cloud)}"
            # print(f"projected dino shape {projected_dino.shape}")

            out["dino_feat"] = projected_dino

        return out
    
    def debug_visualize_projected_dino(
        self,
        point_cloud,
        sparse_dino,
        sensor,
        ridx,
    ):
        """
        Visualize projected DINO features as PCA colors on the point cloud.

        Parameters
        ----------
        point_cloud : (N, C)
            Point cloud. XYZ must be in [:, :3].

        sparse_dino : dict
            sparse_dino["flat_indices"]
            sparse_dino["patch_features"]
            sparse_dino["grid_shape"]

        sensor : str
            LiDAR sensor name.

        ridx : int
            Root index.
        """

        import open3d as o3d
        from sklearn.decomposition import PCA

        pts_xyz = point_cloud[:, :3]

        sparse_indices = sparse_dino["flat_indices"]
        sparse_features = sparse_dino["patch_features"]

        Ht, Wt = map(
            int,
            sparse_dino["grid_shape"],
        )

        patch_size = 16

        intr = self.camera_intrinsics[ridx]

        f = intr["f"]
        cx = intr["cx"]
        cy = intr["cy"]
        k = intr["k"]

        f_patch = f / patch_size
        cx_patch = cx / patch_size
        cy_patch = cy / patch_size

        t_rtk = self._t_rtk_map[(ridx, sensor)]

        if t_rtk is None:
            print("No T_rtk calibration")
            return

        t_rot, t_trans = t_rtk

        T_rtk_sensor = np.eye(4)

        T_rtk_sensor[:3, :3] = t_rot.as_matrix()
        T_rtk_sensor[:3, 3] = t_trans

        T_cam_sensor = (
            np.linalg.inv(self._t_rtk_cam[ridx])
            @ T_rtk_sensor
        )

        # ------------------------------------------------------------
        # Sensor -> camera
        # ------------------------------------------------------------

        pts_cam = (
            pts_xyz @ T_cam_sensor[:3, :3].T
            + T_cam_sensor[:3, 3]
        )

        # ------------------------------------------------------------
        # Project
        # ------------------------------------------------------------

        uv_patch, valid = project_fisheye_simple_radial(
            pts_cam,
            f=f_patch,
            cx=cx_patch,
            cy=cy_patch,
            k=k,
            img_w=Wt,
            img_h=Ht,
        )

        if not np.any(valid):
            print("No valid projected points")
            return

        px = uv_patch[valid, 0].astype(np.int32)
        py = uv_patch[valid, 1].astype(np.int32)

        projected_flat = py * Wt + px

        # ------------------------------------------------------------
        # Sparse lookup
        # ------------------------------------------------------------

        lookup_pos = np.searchsorted(
            sparse_indices,
            projected_flat,
        )

        inside = lookup_pos < len(sparse_indices)

        matched = np.zeros_like(inside)

        matched[inside] = (
            sparse_indices[lookup_pos[inside]]
            == projected_flat[inside]
        )

        valid_idx = np.where(valid)[0]

        matched_idx = valid_idx[matched]

        if len(matched_idx) == 0:
            print("No sparse DINO matches")
            return

        # ------------------------------------------------------------
        # Gather matched features
        # ------------------------------------------------------------

        point_features = sparse_features[
            lookup_pos[matched]
        ].astype(np.float32)

        # ------------------------------------------------------------
        # PCA -> RGB
        # ------------------------------------------------------------

        pca = PCA(n_components=3)

        colors = pca.fit_transform(point_features)

        colors -= colors.min(axis=0, keepdims=True)

        colors /= (
            colors.max(axis=0, keepdims=True)
            + 1e-8
        )

        # ------------------------------------------------------------
        # Build Open3D cloud
        # ------------------------------------------------------------

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(
            pts_xyz[matched_idx]
        )

        pcd.colors = o3d.utility.Vector3dVector(
            colors
        )

        # ------------------------------------------------------------
        # Visualize
        # ------------------------------------------------------------

        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Projected DINO PCA",
        )
