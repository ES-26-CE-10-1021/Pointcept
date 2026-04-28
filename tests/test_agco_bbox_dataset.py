"""Unit tests for AgcoBBoxV1 dataset and AgcoBBoxConfig."""

import json
import os
import tempfile
import warnings

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tmp_dataset_tree(
    tmp_root,
    roots=("scene0",),
    sensors=("lslidar",),
    timestamps=("1700000000.000",),
    boxes_by_root=None,
    R_level_matrix=None,
    num_points=1000,
    write_gravity=True,
):
    """Populate a minimal on-disk tree compatible with AgcoBBoxV1.

    boxes_by_root: dict[root_name, list[dict(position, rotationQuaternion,
                   scaling, label)]]. Defaults to one centered box per root.
    """
    os.makedirs(tmp_root, exist_ok=True)
    meta_dir = os.path.join(tmp_root, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "agco_train.txt"), "w") as f:
        f.write("\n".join(roots) + "\n")
    with open(os.path.join(meta_dir, "agco_val.txt"), "w") as f:
        f.write("\n".join(roots) + "\n")

    R_level_matrix = (
        np.eye(3) if R_level_matrix is None else np.asarray(R_level_matrix)
    )

    for r in roots:
        abs_root = os.path.join(tmp_root, r)
        os.makedirs(abs_root, exist_ok=True)
        if write_gravity:
            np.savez(
                os.path.join(abs_root, "gravity_align.npz"),
                R_level=R_level_matrix.astype(np.float64),
            )
        for s in sensors:
            coord_dir = os.path.join(abs_root, s, "pointcloud_raw", "coord")
            bbox_dir = os.path.join(abs_root, s, "annotations")
            os.makedirs(coord_dir, exist_ok=True)
            os.makedirs(bbox_dir, exist_ok=True)
            for ts in timestamps:
                pts = np.random.RandomState(0).randn(num_points, 3).astype(
                    np.float32
                ) * 5.0
                np.save(os.path.join(coord_dir, f"{ts}.npy"), pts)
                anns = (
                    boxes_by_root[r]
                    if boxes_by_root and r in boxes_by_root
                    else [
                        dict(
                            translation=[0.0, 0.0, 0.0],
                            rotation=[0.0, 0.0, 0.0, 1.0],
                            dimensions=[1.0, 1.0, 1.0],
                            label=1,
                            inliers=100,
                            is_visible=True,
                        )
                    ]
                )
                with open(os.path.join(bbox_dir, f"{ts}.json"), "w") as f:
                    json.dump({"annotations": anns}, f)
    return tmp_root, meta_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_agco_bbox_config_angle_roundtrip():
    from pointcept.models.detection_3detr.dataset_config import AgcoBBoxConfig

    cfg = AgcoBBoxConfig(num_angle_bin=12)
    bin_size = 2 * np.pi / cfg.num_angle_bin
    for yaw_gt in [-np.pi + 1e-3, -np.pi / 2, 0.0, np.pi / 4, np.pi / 2, np.pi - 1e-3]:
        cls_id, res = cfg.angle2class(yaw_gt)
        assert 0 <= cls_id < cfg.num_angle_bin
        # Batch decode (numpy arrays expected).
        recovered = cfg.class2anglebatch(
            np.array([cls_id]), np.array([res]), to_label_format=True
        )[0]
        # Recovered should equal yaw_gt modulo 2π, wrapped to (-π, π].
        diff = (recovered - yaw_gt + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 1e-6, (yaw_gt, recovered, diff)
        # Residual must be within half a bin.
        assert abs(res) <= bin_size / 2 + 1e-9


def test_agco_dataset_schema_and_shapes():
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(tmp)
        ds = AgcoBBoxV1(
            root_dir=tmp_root,
            meta_data_dir=meta_dir,
            split="train",
            sensors=["lslidar"],
            num_points=500,
            augment=False,
        )
        assert len(ds) == 1
        sample = ds[0]

        expected = {
            "point_clouds": (500, 3),
            "point_cloud_dims_min": (3,),
            "point_cloud_dims_max": (3,),
            "gt_box_centers": (64, 3),
            "gt_box_centers_normalized": (64, 3),
            "gt_box_sizes": (64, 3),
            "gt_box_sizes_normalized": (64, 3),
            "gt_box_angles": (64,),
            "gt_angle_class_label": (64,),
            "gt_angle_residual_label": (64,),
            "gt_box_sem_cls_label": (64,),
            "gt_box_present": (64,),
            "gt_box_corners": (64, 8, 3),
        }
        for key, shape in expected.items():
            assert key in sample, f"missing key {key}"
            assert sample[key].shape == shape, (key, sample[key].shape, shape)

        assert sample["gt_box_present"][0] == 1.0
        assert sample["gt_box_present"][1] == 0.0
        assert sample["gt_box_sem_cls_label"][0] == 1


def test_agco_dataset_gravity_roundtrip():
    """Constructing a box with known yaw in the levelled frame, then un-levelling
    it on disk (simulating what the annotation tool does), and finally loading
    it through the Dataset should recover the original yaw."""
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    yaw_gt = 0.7
    # Non-trivial R_level = small tilt about x+y.
    R_level = Rot.from_euler("xyz", [0.08, -0.05, 0.0])
    R_level_mat = R_level.as_matrix()

    # Levelled-frame quaternion (pure yaw)
    q_level = Rot.from_euler("xyz", [0.0, 0.0, yaw_gt]).as_quat()
    # What the annotation tool stores on disk (un-levelled)
    q_raw = (R_level.inv() * Rot.from_quat(q_level)).as_quat()

    boxes = {
        "scene0": [
            dict(
                translation=[0.0, 0.0, 0.0],
                rotation=list(q_raw),
                dimensions=[2.0, 1.0, 1.5],
                label=2,
                inliers=100,
                is_visible=True,
            )
        ]
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(
            tmp, boxes_by_root=boxes, R_level_matrix=R_level_mat
        )
        ds = AgcoBBoxV1(
            root_dir=tmp_root,
            meta_data_dir=meta_dir,
            split="train",
            sensors=["lslidar"],
            num_points=500,
            augment=False,
            apply_r_level_to_points=True,
            apply_r_level_to_boxes=True,
        )
        sample = ds[0]
        recovered_yaw = float(sample["gt_box_angles"][0])
        diff = (recovered_yaw - yaw_gt + np.pi) % (2 * np.pi) - np.pi
        assert abs(diff) < 1e-6, (yaw_gt, recovered_yaw, diff)
        assert sample["gt_box_sem_cls_label"][0] == 2


def test_agco_dataset_requires_gravity_by_default():
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(tmp, write_gravity=False)
        with pytest.raises(FileNotFoundError):
            AgcoBBoxV1(
                root_dir=tmp_root,
                meta_data_dir=meta_dir,
                split="train",
                sensors=["lslidar"],
            )


def test_agco_dataset_gravity_permissive_fallback():
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(tmp, write_gravity=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = AgcoBBoxV1(
                root_dir=tmp_root,
                meta_data_dir=meta_dir,
                split="train",
                sensors=["lslidar"],
                require_gravity_align=False,
            )
        assert any("gravity_align" in str(w.message) for w in caught)
        assert len(ds) == 1


def _write_calibration_yml(root_dir, sensors_matrices: dict):
    """Write a minimal calibration.yml to root_dir.

    sensors_matrices: {sensor_name: 4x4 list-of-lists}
    """
    import yaml

    data = {
        "T_rtk_sensors": {
            s: {"matrix": m} for s, m in sensors_matrices.items()
        }
    }
    path = os.path.join(root_dir, "calibration.yml")
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _identity_4x4():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _translation_4x4(tx, ty, tz):
    """Pure translation 4×4 matrix."""
    return [
        [1.0, 0.0, 0.0, tx],
        [0.0, 1.0, 0.0, ty],
        [0.0, 0.0, 1.0, tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


def test_t_rtk_map_populated_at_init():
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(tmp)
        abs_root = os.path.join(tmp_root, "scene0")
        _write_calibration_yml(abs_root, {"lslidar": _identity_4x4()})

        ds = AgcoBBoxV1(
            root_dir=tmp_root,
            meta_data_dir=meta_dir,
            split="train",
            sensors=["lslidar"],
            num_points=500,
            apply_t_rtk=True,
        )
        # Map must have an entry for (ridx=0, sensor="lslidar")
        key = (0, "lslidar")
        assert key in ds._t_rtk_map
        rot, trans = ds._t_rtk_map[key]
        np.testing.assert_allclose(trans, [0.0, 0.0, 0.0], atol=1e-6)


def test_t_rtk_translates_points():
    """A pure translation T_rtk must shift all point cloud XYZ by (tx, ty, tz)."""
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    tx, ty, tz = 3.0, -1.5, 0.5

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(tmp, num_points=100)
        abs_root = os.path.join(tmp_root, "scene0")
        _write_calibration_yml(abs_root, {
            "lslidar": _translation_4x4(tx, ty, tz)
        })

        ds_no_rtk = AgcoBBoxV1(
            root_dir=tmp_root,
            meta_data_dir=meta_dir,
            split="train",
            sensors=["lslidar"],
            num_points=100,
            apply_t_rtk=False,
        )
        ds_rtk = AgcoBBoxV1(
            root_dir=tmp_root,
            meta_data_dir=meta_dir,
            split="train",
            sensors=["lslidar"],
            num_points=100,
            apply_t_rtk=True,
        )

        # Use seed=42 so _random_sampling picks the same subset
        np.random.seed(42)
        sample_no = ds_no_rtk[0]
        np.random.seed(42)
        sample_rt = ds_rtk[0]

        diff = sample_rt["point_clouds"][:, :3] - sample_no["point_clouds"][:, :3]
        np.testing.assert_allclose(diff[:, 0], tx, atol=1e-4)
        np.testing.assert_allclose(diff[:, 1], ty, atol=1e-4)
        np.testing.assert_allclose(diff[:, 2], tz, atol=1e-4)


def test_agco_dataset_multi_sensor_enumeration():
    from pointcept.datasets.agco_bbox import AgcoBBoxV1

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root, meta_dir = _make_tmp_dataset_tree(
            tmp,
            sensors=["lslidar", "ouster"],
            timestamps=["100.000", "200.000", "300.000"],
        )
        ds = AgcoBBoxV1(
            root_dir=tmp_root,
            meta_data_dir=meta_dir,
            split="train",
            sensors=["lslidar", "ouster"],
            num_points=500,
        )
        # 1 root * 2 sensors * 3 timestamps = 6 samples.
        assert len(ds) == 6
        sensor_counts = {s: 0 for s in ("lslidar", "ouster")}
        for ridx, sensor, ts in ds.samples:
            sensor_counts[sensor] += 1
        assert sensor_counts == {"lslidar": 3, "ouster": 3}
