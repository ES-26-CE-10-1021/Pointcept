#!/usr/bin/env python
"""
Raw visualization of AGCO point clouds + annotations without dataset class.

Loads raw point cloud and annotation files directly, applies minimal transforms,
and visualizes in Open3D. No gravity alignment, no subsampling — just raw data.

This is useful for debugging alignment issues between raw files.

Usage:
    python tools/visualize_raw_agco.py \\
        --root /path/to/annotation_root \\
        --sensor lslidar \\
        --timestamp 1773324821.300000000 \\
        [--num-samples 10] \\
        [--apply-r-level]
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RigidTransform as Trans


def extract_t_rtk_transform(data: dict, sensor: str) -> dict:
    T = np.asarray(data["T_rtk_sensors"][sensor]["matrix"], dtype=np.float32)
    return dict(
        rotation = Rot.from_matrix(T[:3, :3]),
        translation = T[:3,3],
        transformation = Trans.from_matrix(T),
    )


def quaternion_to_rotation_matrix(quat):
    """Convert xyzw quaternion to rotation matrix."""
    return Rot.from_quat(quat).as_matrix()


def create_box_lineset(center, size, rotation_mat, color=None):
    """Create an Open3D LineSet for an oriented bounding box."""
    if color is None:
        color = [0.0, 1.0, 0.0]

    half_size = size / 2.0
    corners_local = np.array([
        [-half_size[0], -half_size[1], -half_size[2]],
        [half_size[0], -half_size[1], -half_size[2]],
        [half_size[0], half_size[1], -half_size[2]],
        [-half_size[0], half_size[1], -half_size[2]],
        [-half_size[0], -half_size[1], half_size[2]],
        [half_size[0], -half_size[1], half_size[2]],
        [half_size[0], half_size[1], half_size[2]],
        [-half_size[0], half_size[1], half_size[2]],
    ])

    corners_world = (rotation_mat @ corners_local.T).T + center

    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def visualize_raw(pts, boxes, timestamp, apply_r_level=False, apply_r_level_inv=False, R_level=None, apply_t_rtk=False, T_rtk=None):
    """Visualize raw point cloud + annotations.

    Args:
        pts: (N, 3+) point cloud
        boxes: list of dicts with 'translation'/'position', 'rotation'/'rotationQuaternion',
               'dimensions'/'scaling', 'label'
        timestamp: for window title
        apply_r_level: whether to apply R_level (raw-world → levelled)
        apply_r_level_inv: whether to apply R_level.inv() (levelled → raw-world)
        R_level: scipy Rotation object or None
        apply_t_rtk: whether to apply T_rtk transform
        T_rtk: dict with 'transformation' (scipy RigidTransform) or None
    """
    colors = [
        [1.0, 0.0, 0.0],  # 0 — red
        [0.0, 1.0, 0.0],  # 1 — green
        [0.0, 0.0, 1.0],  # 2 — blue
        [1.0, 1.0, 0.0],  # 3 — yellow
    ]

    # Transform points if requested
    pts_vis = pts.copy()
    if apply_t_rtk and T_rtk is not None:
        pts_vis[:, :3] = T_rtk["transformation"].apply(pts_vis[:, :3])
    if apply_r_level and R_level is not None:
        pts_vis[:, :3] = R_level.apply(pts_vis[:, :3])
    if apply_r_level_inv and R_level is not None:
        pts_vis[:, :3] = R_level.inv().apply(pts_vis[:, :3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_vis[:, :3])

    if pts_vis.shape[1] >= 4:
        intensity = pts_vis[:, 3]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(
            np.column_stack([intensity, intensity, intensity])
        )
    else:
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones((pts_vis.shape[0], 3)) * 0.7
        )

    geometries = [pcd]

    for box in boxes:
        center = np.array(box.get("translation") or box.get("position"))
        size = np.array(box.get("dimensions") or box.get("scaling"))
        quat = box.get("rotation") or box.get("rotationQuaternion")
        label = int(box["label"])

        # Transform box if requested (T_rtk only transforms points, not boxes)
        if apply_r_level and R_level is not None:
            center = R_level.apply(center)
            q_rot = (R_level * Rot.from_quat(quat)).as_quat()
        elif apply_r_level_inv and R_level is not None:
            center = R_level.inv().apply(center)
            q_rot = (R_level.inv() * Rot.from_quat(quat)).as_quat()
        else:
            q_rot = quat

        color = colors[min(label, len(colors) - 1)]
        rot_mat = quaternion_to_rotation_matrix(q_rot)
        lineset = create_box_lineset(center, size, rot_mat, color)
        geometries.append(lineset)

    if apply_t_rtk:
        mode = " (T_rtk applied)"
    elif apply_r_level:
        mode = " (R_level applied: raw-world → levelled)"
    elif apply_r_level_inv:
        mode = " (R_level.inv() applied: levelled → raw-world)"
    else:
        mode = " (raw)"
    title = f"{timestamp} | {len(boxes)} boxes{mode}"
    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1280,
        height=960,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize raw AGCO data (no dataset class)"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to annotation root (e.g., /path/agco2026/static_2/split_2)",
    )
    parser.add_argument(
        "--sensor",
        default="lslidar",
        help="Sensor name (e.g., lslidar, ouster)",
    )
    parser.add_argument(
        "--timestamp",
        help="Specific timestamp to load (e.g., 1773324821.300000000). "
        "If not specified, will iterate through all.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Max samples to visualize (if no timestamp specified)",
    )
    parser.add_argument(
        "--apply-r-level",
        action="store_true",
        help="Apply R_level rotation to boxes (inverse: raw-world → levelled)",
    )
    parser.add_argument(
        "--apply-r-level-inv",
        action="store_true",
        help="Apply R_level.inverse() to boxes (forward: levelled → raw-world)",
    )
    parser.add_argument(
        "--apply-t-rtk",
        action="store_true",
        help="Apply T_rtk transform (sensor calibration)",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=67,
        help="Minimum inliers constraint for bbox visibility",
    )

    args = parser.parse_args()

    # Load R_level if available
    R_level = None
    gravity_path = os.path.join(args.root, "gravity_align.npz")
    if os.path.isfile(gravity_path):
        grav_data = np.load(gravity_path)
        if "R_level" in grav_data.files:
            R_level = Rot.from_matrix(np.asarray(grav_data["R_level"]))
            if args.apply_r_level:
                print(f"Loaded R_level from {gravity_path}")
    else:
        if args.apply_r_level:
            print(f"Warning: gravity_align.npz not found at {gravity_path}")

    # Load T_rtk if needed
    T_rtk = None
    calib_path = os.path.join(args.root, "calibration.yml")
    if os.path.isfile(calib_path) and args.apply_t_rtk:
        import yaml
        with open(calib_path, 'r') as f:
            calibration = yaml.full_load(f)
        T_rtk = extract_t_rtk_transform(calibration, sensor=args.sensor)
    elif args.apply_t_rtk:
        print(f"[WARNING] Couldn't find 'calibration.yml' at '{calib_path}', exiting...")
        exit()

    coord_dir = os.path.join(args.root, args.sensor, "pointcloud_raw", "coord")
    anno_dir = os.path.join(args.root, args.sensor, "annotations")

    if not os.path.isdir(coord_dir):
        print(f"Error: coord_dir not found: {coord_dir}", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(anno_dir):
        print(f"Error: anno_dir not found: {anno_dir}", file=sys.stderr)
        sys.exit(1)

    # Collect all timestamps (prioritize annotations, match with coords)
    if args.timestamp:
        timestamps = [args.timestamp]
    else:
        anno_files = set(
            os.path.splitext(f)[0]
            for f in os.listdir(anno_dir)
            if f.endswith(".json")
        )
        coord_files = set(
            os.path.splitext(f)[0]
            for f in os.listdir(coord_dir)
            if f.endswith(".npy")
        )
        timestamps = sorted(anno_files & coord_files)[:args.num_samples]

    print(f"Visualizing {len(timestamps)} samples from {args.sensor}")
    print(f"Filtering boxes: is_visible=True and inliers>={args.min_inliers}")
    if args.apply_t_rtk:
        print("T_rtk will be applied to points and boxes")
    if args.apply_r_level:
        print("R_level will be applied to boxes (raw-world → levelled)")
    if args.apply_r_level_inv:
        print("R_level.inv() will be applied to boxes (levelled → raw-world)")

    for ts in timestamps:
        coord_path = os.path.join(coord_dir, f"{ts}.npy")
        anno_path = os.path.join(anno_dir, f"{ts}.json")

        if not os.path.isfile(coord_path) or not os.path.isfile(anno_path):
            print(f"Skipping {ts}: missing coord or anno file")
            continue

        print(f"\nLoading {ts}...", end=" ", flush=True)
        try:
            pts = np.load(coord_path).astype(np.float32)
            with open(anno_path) as f:
                data = json.load(f)
            boxes = data.get("annotations", [])
            # Filter boxes by visibility and inliers constraint
            # Include children only if parent is visible
            filtered_boxes = []
            for b in boxes:
                if b.get("is_visible", True) and b.get("inliers", 0) >= args.min_inliers:
                    filtered_boxes.append(b)
                    # Add children if parent passes filter
                    children = b.get("children", [])
                    filtered_boxes.extend(children)
            boxes = filtered_boxes
            print(f"OK ({pts.shape[0]} pts, {len(boxes)} boxes)")
            visualize_raw(pts, boxes, ts, args.apply_r_level, args.apply_r_level_inv, R_level, args.apply_t_rtk, T_rtk)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
