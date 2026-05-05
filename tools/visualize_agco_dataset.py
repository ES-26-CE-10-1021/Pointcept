#!/usr/bin/env python
"""
Interactive Open3D visualization of AgcoBBoxV1 dataset or 3DETR predictions.

Modes:
  - dataset: Load from AgcoBBoxV1 dataset with GT boxes
  - predictions: Load from saved test run (JSON + .npy files) with GT + pred boxes

Controls:
    [→]     : Load next sample
    [←]     : Load previous sample
    [P/Space] : Toggle Play/Pause video mode
    [R]     : Reload current sample (see different augmentations)
    [1-9]   : Scrub dataset (1=Start, 5=Middle, 9=End)
    [Q]     : Quit
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

from pointcept.datasets.agco_bbox import AgcoBBoxV1

_DEFAULT_VIEW = {
    "front":  [-0.8774444612089376, 0.057891256148444932, 0.47617204869176499],
    "lookat": [46.248564381546856, -4.1298021356340309, -11.353385072551928],
    "up":     [0.47679769375413167, -0.003366388927639026, 0.87900661354527321],
    "zoom":   0.42,
}


def quaternion_to_rotation_matrix(quat):
    """Convert xyzw quaternion to rotation matrix."""
    return Rot.from_quat(quat).as_matrix()


def point_in_axis_aligned_box(point, center, size):
    """Check if a point is inside an axis-aligned bounding box (ignores rotation).

    Args:
        point: (3,) array [x, y, z]
        center: (3,) array box center
        size: (3,) array [length, width, height]

    Returns:
        bool: True if point is inside the box
    """
    half_size = size / 2.0
    return (
        abs(point[0] - center[0]) <= half_size[0]
        and abs(point[1] - center[1]) <= half_size[1]
        and abs(point[2] - center[2]) <= half_size[2]
    )


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
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def create_box_lineset_from_corners(corners, color=None):
    """Create an Open3D LineSet from 8 pre-computed world-space corners."""
    if color is None:
        color = [1.0, 1.0, 1.0]
    corners = np.asarray(corners)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


CLASS_COLORS = [
    [0.0, 1.0, 0.0],  # tractor — green
    [0.0, 0.0, 1.0],  # harvester — blue
    [1.0, 1.0, 0.0],  # trailer — yellow
    [1.0, 0.5, 0.0],  # car — orange
    [1.0, 0.0, 0.0],  # hopper — red
]


def camera_to_lidar_np(points):
    """Inverse of 3DETR flip_axis_to_camera_np: [x, y, z] -> [x, z, -y]."""
    pts = np.asarray(points)
    out = pts.copy()
    out[..., 0] = pts[..., 0]
    out[..., 1] = pts[..., 2]
    out[..., 2] = -pts[..., 1]
    return out


def build_geometries_for_sample(sample, color_pts=False):
    """Generate Open3D geometries from a dataset sample without triggering rendering."""
    colors = CLASS_COLORS

    pts = sample["point_clouds"].copy()
    gt_box_centers = sample["gt_box_centers"]
    gt_box_sizes = sample["gt_box_sizes"]
    gt_box_angles = sample["gt_box_angles"]
    gt_present = sample["gt_box_present"]
    gt_sem_cls = sample["gt_box_sem_cls_label"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    # Initialize point colors: grayscale from intensity or default
    if pts.shape[1] == 4:
        intensity = pts[:, 3]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        point_colors = np.column_stack([intensity, intensity, intensity])
    else:
        point_colors = np.ones((pts.shape[0], 3)) * 0.7

    # Color points that are inside bounding boxes (if enabled)
    if color_pts:
        # For each point, check all boxes and assign the first matching box's color
        for j in range(pts.shape[0]):
            for i in range(len(gt_present)):
                if gt_present[i] < 0.5:
                    continue
                if point_in_axis_aligned_box(pts[j, :3], gt_box_centers[i], gt_box_sizes[i]):
                    cls_id = int(gt_sem_cls[i])
                    point_colors[j] = colors[min(cls_id, len(colors) - 1)]
                    break  # Stop checking boxes for this point once a match is found

    pcd.colors = o3d.utility.Vector3dVector(point_colors)

    geometries = [pcd]

    for i in range(len(gt_present)):
        if gt_present[i] < 0.5:
            continue

        center = gt_box_centers[i]
        size = gt_box_sizes[i]
        yaw = gt_box_angles[i]
        cls_id = int(gt_sem_cls[i])

        rot_mat = Rot.from_euler("xyz", [0, 0, yaw]).as_matrix()
        color = colors[min(cls_id, len(colors) - 1)]
        lineset = create_box_lineset(center, size, rot_mat, color)
        geometries.append(lineset)

    return geometries


def build_geometries_for_prediction(record, pts, score_thresh=0.0):
    """Generate Open3D geometries from a saved prediction record."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    if pts.shape[1] == 4:
        intensity = pts[:, 3]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(np.column_stack([intensity] * 3))
    else:
        pcd.colors = o3d.utility.Vector3dVector(np.full((len(pts), 3), 0.7))

    geometries = [pcd]

    # Preferred future-proof path: draw GT from center/size/yaw in point-cloud frame.
    gt_param = record.get("gt_boxes_param", [])
    if gt_param:
        for box in gt_param:
            cls_id = box["class_idx"]
            color = CLASS_COLORS[min(cls_id, len(CLASS_COLORS) - 1)]
            center = np.asarray(box["center"], dtype=np.float64)
            size = np.asarray(box["size"], dtype=np.float64)
            yaw = float(box["yaw"])
            rot_mat = Rot.from_euler("xyz", [0.0, 0.0, yaw]).as_matrix()
            geometries.append(create_box_lineset(center, size, rot_mat, color))
    else:
        # Backward compatibility for old exports that only stored camera/upright corners.
        for box in record.get("gt_boxes", []):
            cls_id = box["class_idx"]
            color = CLASS_COLORS[min(cls_id, len(CLASS_COLORS) - 1)]
            gt_corners = camera_to_lidar_np(np.asarray(box["box_corners"]))
            geometries.append(create_box_lineset_from_corners(gt_corners, color))

    for box in record.get("predictions", []):
        if box["score"] < score_thresh:
            continue
        cls_id = box["class_idx"]
        base = CLASS_COLORS[min(cls_id, len(CLASS_COLORS) - 1)]
        dim_color = [c * 0.5 for c in base]
        pred_corners = camera_to_lidar_np(np.asarray(box["box_corners"]))
        geometries.append(create_box_lineset_from_corners(pred_corners, dim_color))

    return geometries


class PredictionDataset:
    """Wraps a predictions directory to load saved test results."""
    def __init__(self, predictions_dir, score_thresh=0.0):
        self.predictions_dir = predictions_dir
        self.score_thresh = score_thresh
        manifest_path = os.path.join(predictions_dir, "manifest.json")
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        json_path = os.path.join(self.predictions_dir, entry["json_file"])
        pts_path = os.path.join(self.predictions_dir, entry["points_file"])
        with open(json_path) as f:
            record = json.load(f)
        pts = np.load(pts_path)
        return record, pts


class DatasetViewer:
    def __init__(self, dataset, start_idx=0, max_samples=None, color_pts=False,
                 geometry_builder=None):
        self.dataset = dataset
        self.current_idx = start_idx
        self.max_idx = min(start_idx + max_samples, len(dataset)) if max_samples else len(dataset)
        self.active_geometries = []
        self.color_pts = color_pts
        self.geometry_builder = geometry_builder or build_geometries_for_sample

        # --- PLAYBACK SETTINGS ---
        self.playing = False
        self.play_delay_ms = 100  # Adjust this integer to change milliseconds between frames
        self.last_update_time = time.time()
        # -------------------------

        # Initialize the non-blocking visualizer with callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1280, height=960, window_name="AgcoBBoxV1 Interactive Viewer")

        # Register callbacks: 262 = →, 263 = ←, 32 = Space, 80 = 'P', 81 = 'Q', 82 = 'R'
        self.vis.register_key_callback(262, self.next_sample)
        self.vis.register_key_callback(263, self.prev_sample)
        self.vis.register_key_callback(32, self.toggle_play)
        self.vis.register_key_callback(80, self.toggle_play)
        self.vis.register_key_callback(82, self.reload_sample)
        self.vis.register_key_callback(81, self.quit)
        
        def make_scrub_callback(fraction):
            def callback(vis=None):
                self.playing = False  # Auto-pause on manual jump
                target_idx = int(fraction * (self.max_idx - 1))
                
                # Only reload if the index actually changed
                if self.current_idx != target_idx:
                    self.current_idx = target_idx
                    self.load_sample()
                else:
                    print(f"Already at sample {self.current_idx}")
                return False
            return callback

        for i in range(1, 10):
            # i=1 -> fraction=0.0, i=5 -> fraction=0.5, i=9 -> fraction=1.0
            fraction = (i - 1) / 8.0
            self.vis.register_key_callback(48 + i, make_scrub_callback(fraction))

        # Register the animation loop callback
        self.vis.register_animation_callback(self.animation_update)

        self.is_first_frame = True

        print("\n--- Viewer Controls ---")
        print("[→]     : Next frame")
        print("[←]     : Previous frame")
        print("[P/Space]: Toggle Play/Pause")
        print("[R]     : Reload current sample (see different augmentations)")
        print("[1-9]   : Scrub dataset (1=Start, 5=Middle, 9=End)")
        print("[Q]     : Quit")
        print("-----------------------")

        print("\n--- Class Legend ---")
        class_names = ["tractor", "harvester", "trailer", "car", "hopper"]
        color_symbols = ["🟢", "🔵", "🟡", "🟠", "🔴"]
        for i, (name, symbol) in enumerate(zip(class_names, color_symbols)):
            print(f"{symbol} {i}: {name}")
        print("-----------------------\n")

        # Load the initial sample
        self.load_sample()

    def load_sample(self):
        print(f"[{self.current_idx + 1}/{self.max_idx}] Loading sample {self.current_idx}...", end=" ", flush=True)
        
        try:
            sample = self.dataset[self.current_idx]

            # Dispatch: PredictionDataset returns (record, pts), AgcoBBoxV1 returns dict
            if isinstance(sample, tuple):
                record, pts = sample
                new_geometries = self.geometry_builder(record, pts, score_thresh=self.dataset.score_thresh)
                n_pred = sum(1 for b in record.get("predictions", []) if b["score"] >= self.dataset.score_thresh)
                n_gt = len(record.get("gt_boxes", []))
                print(f"OK (scan {record['scan_idx']}: {pts.shape[0]} pts, {n_pred} preds, {n_gt} GT)")
                diag = record.get("diagnostics", {})
                if diag:
                    print(
                        "  diag: center_delta mean/max = "
                        f"{diag.get('mean_l2_center_delta_corners_vs_gt_centers', float('nan')):.3f}/"
                        f"{diag.get('max_l2_center_delta_corners_vs_gt_centers', float('nan')):.3f}"
                    )
                cfg_diag = record.get("dataset_config_diagnostics", {})
                if cfg_diag:
                    print(
                        "  cfg: "
                        f"split={cfg_diag.get('split')} sensors={cfg_diag.get('sensors')} "
                        f"num_points={cfg_diag.get('num_points')} min_inliers={cfg_diag.get('min_inliers')} "
                        f"apply_t_rtk={cfg_diag.get('apply_t_rtk')} apply_r_global={cfg_diag.get('apply_r_global')} "
                        f"apply_r_level_to_points={cfg_diag.get('apply_r_level_to_points')} "
                        f"apply_r_level_to_boxes={cfg_diag.get('apply_r_level_to_boxes')}"
                    )
            else:
                new_geometries = self.geometry_builder(sample, color_pts=self.color_pts)
                num_boxes = int(np.sum(sample["gt_box_present"]))
                pts_shape = sample["point_clouds"].shape
                print(f"OK ({pts_shape[0]} pts, {num_boxes} boxes)")

            # 1. Remove old geometries
            for geom in self.active_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.active_geometries.clear()

            # 2. Build new geometries already done above
            # new_geometries = ...

            # 3. Add to visualizer
            for geom in new_geometries:
                # Only reset the camera angle on the very first frame
                self.vis.add_geometry(geom, reset_bounding_box=self.is_first_frame)
                self.active_geometries.append(geom)

            if self.is_first_frame:
                vc = self.vis.get_view_control()
                vc.set_front(_DEFAULT_VIEW["front"])
                vc.set_lookat(_DEFAULT_VIEW["lookat"])
                vc.set_up(_DEFAULT_VIEW["up"])
                vc.set_zoom(_DEFAULT_VIEW["zoom"])

            self.is_first_frame = False

        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()

    def next_sample(self, vis=None):
        if self.current_idx < self.max_idx - 1:
            self.current_idx += 1
            self.load_sample()
        else:
            print("Reached the end of the specified samples.")
            self.playing = False
        return False  

    def prev_sample(self, vis=None):
        self.playing = False
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_sample()
        else:
            print("Already at the first sample.")
        return False

    def reload_sample(self, vis=None):
        """Reload the current sample to see different augmentations."""
        self.playing = False
        self.load_sample()
        return False

    def manual_next(self, vis=None):
        """Triggered only by the right arrow key."""
        self.playing = False
        return self.next_sample()

    def manual_prev(self, vis=None):
        """Triggered only by the left arrow key."""
        self.playing = False
        return self.prev_sample()

    def toggle_play(self, vis=None):
        self.playing = not self.playing
        if self.playing:
            print(f"Playing at {self.play_delay_ms}ms per frame...")
            self.last_update_time = time.time()  # Reset the clock when play starts
        else:
            print("Paused.")
        return False

    def animation_update(self, vis):
        """Called by Open3D on every render loop iteration."""
        if self.playing:
            current_time = time.time()
            elapsed_ms = (current_time - self.last_update_time) * 1000.0

            if elapsed_ms >= self.play_delay_ms:
                self.next_sample()
                self.last_update_time = current_time
                
                # Returning True tells Open3D the geometry was modified and requires a re-render
                return True 
                
        return False

    def quit(self, vis=None):
        print("Exiting viewer.")
        self.vis.destroy_window()
        sys.exit(0)
        return False

    def run(self):
        self.vis.run()
        self.vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize AgcoBBoxV1 dataset or 3DETR test predictions in Open3D"
    )
    parser.add_argument("--mode", choices=["dataset", "predictions"], default="dataset",
                        help="Visualization mode")
    parser.add_argument("--predictions-dir", help="Path to predictions/ directory (--mode predictions)")
    parser.add_argument("--score-thresh", type=float, default=0.05,
                        help="Min confidence score to display (predictions mode)")
    parser.add_argument("--root-dir", help="Path to data_root (dataset mode)")
    parser.add_argument("--meta-dir", help="Path to meta_data_dir (dataset mode)")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--split-prefix", default="agco")
    parser.add_argument("--sensors", nargs="+", default=["lslidar"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--use-intensity", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument("--apply-t-rtk", action="store_true")
    parser.add_argument("--apply-r-global", action="store_true",
                        help="Apply per-timestamp global rotation to point cloud XYZ")
    parser.add_argument("--sensor", default="lslidar")
    parser.add_argument("--min-inliers", type=int, default=200)
    parser.add_argument("--apply-gravity-boxes", action="store_true")
    parser.add_argument("--apply-gravity-pts", action="store_true")
    parser.add_argument("--color-pts", action="store_true", help="Color points inside bounding boxes")

    args = parser.parse_args()

    if args.mode == "predictions":
        if not args.predictions_dir:
            parser.error("--predictions-dir is required with --mode predictions")
        dataset = PredictionDataset(args.predictions_dir, score_thresh=args.score_thresh)
        viewer = DatasetViewer(
            dataset=dataset,
            start_idx=args.start_idx,
            max_samples=args.num_samples,
            geometry_builder=build_geometries_for_prediction,
        )
        print(f"Predictions: {len(dataset)} scans (score_thresh={args.score_thresh})\n")
        viewer.run()
        return

    if args.mode != "dataset":
        parser.error(f"Unknown mode: {args.mode}")

    if not args.root_dir or not args.meta_dir:
        parser.error("--root-dir and --meta-dir are required with --mode dataset")

    transform = [dict(type="PointSubsampleDetection", num_points=args.num_points)]
    if args.augment:
        transform = [
            # dict(type="RandomFlipDetection", p_x=0.5, p_y=0.5),
            # dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            # dict(type="RandomScaleDetection", scale=(0.9, 1.1), apply_to_sizes=True),
            dict(type="SphericalCropDetection", max_dist=60.0, min_dist=0.0),
            dict(type="FovCropDetection", azimuth_deg=(-60.0, 60.0), crop_points=True),
            dict(type="RandomFlipDetection", p_x=0.0, p_y=0.5),
            dict(type="RandomRotateZDetection", angle_deg=(-5.0, 5.0)),
            # dict(type="RandomJitterDetection", sigma=0.005, clip=0.02),
            # dict(type="RandomCuboidDetection", min_points=30000),
            dict(type="PointSubsampleDetection", num_points=args.num_points),
        ]

    try:
        dataset = AgcoBBoxV1(
            root_dir=args.root_dir,
            meta_data_dir=args.meta_dir,
            split=args.split,
            split_prefix=args.split_prefix,
            sensors=tuple(args.sensors),
            num_points=args.num_points,
            use_intensity=args.use_intensity,
            transform=transform,
            min_inliers=args.min_inliers,
            apply_t_rtk=args.apply_t_rtk,
            require_calibration=args.apply_t_rtk,
            apply_r_global=args.apply_r_global,
            apply_r_level_to_boxes=args.apply_gravity_boxes,
            apply_r_level_to_points=args.apply_gravity_pts,
            require_gravity_align=args.apply_gravity_boxes or args.apply_gravity_pts,
            loop=1,
            residual_rpy_warn_deg=5,
        )
    except Exception as e:
        print(f"Error: Failed to initialize dataset: {e}", file=sys.stderr)
        sys.exit(1)

    if len(dataset) == 0:
        print("Error: Dataset is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset: {len(dataset)} samples")

    # Delegate viewing to the stateful Viewer class
    viewer = DatasetViewer(
        dataset=dataset,
        start_idx=args.start_idx,
        max_samples=args.num_samples,
        color_pts=args.color_pts
    )
    viewer.run()

if __name__ == "__main__":
    main()
