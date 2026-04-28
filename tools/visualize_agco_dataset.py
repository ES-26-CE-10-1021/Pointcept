#!/usr/bin/env python
"""
Interactive Open3D visualization of AgcoBBoxV1 dataset.

Loads samples directly from the AgcoBBoxV1 dataset class to test its output
schema and transformations. 

Controls:
    [Space] : Load next sample
    [B]     : Load previous sample
    [P]     : Toggle Play/Pause video mode
    [1-9]   : Scrub dataset (1=Start, 5=Middle, 9=End)
    [Q]     : Quit
"""

import argparse
import sys
import time

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

from pointcept.datasets.agco_bbox import AgcoBBoxV1


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
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # vertical
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    return line_set


def build_geometries_for_sample(sample):
    """Generate Open3D geometries from a dataset sample without triggering rendering."""
    colors = [
        [1.0, 0.0, 0.0],  # hopper — red
        [0.0, 1.0, 0.0],  # tractor — green
        [0.0, 0.0, 1.0],  # harvester — blue
        [1.0, 1.0, 0.0],  # trailer — yellow
    ]

    pts = sample["point_clouds"].copy()
    gt_box_centers = sample["gt_box_centers"]
    gt_box_sizes = sample["gt_box_sizes"]
    gt_box_angles = sample["gt_box_angles"]
    gt_present = sample["gt_box_present"]
    gt_sem_cls = sample["gt_box_sem_cls_label"]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])

    if pts.shape[1] == 4:
        intensity = pts[:, 3]
        intensity = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-6)
        pcd.colors = o3d.utility.Vector3dVector(
            np.column_stack([intensity, intensity, intensity])
        )
    else:
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones((pts.shape[0], 3)) * 0.7
        )

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


class DatasetViewer:
    def __init__(self, dataset, start_idx=0, max_samples=None):
        self.dataset = dataset
        self.current_idx = start_idx
        self.max_idx = min(start_idx + max_samples, len(dataset)) if max_samples else len(dataset)
        self.active_geometries = []

        # --- PLAYBACK SETTINGS ---
        self.playing = False
        self.play_delay_ms = 100  # Adjust this integer to change milliseconds between frames
        self.last_update_time = time.time()
        # -------------------------

        # Initialize the non-blocking visualizer with callbacks
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(width=1280, height=960, window_name="AgcoBBoxV1 Interactive Viewer")

        # Register callbacks: 32 = Space, 66 = 'B', 80 = 'P', 81 = 'Q'
        self.vis.register_key_callback(32, self.next_sample)
        self.vis.register_key_callback(66, self.prev_sample)
        self.vis.register_key_callback(80, self.toggle_play)
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
        print("[Space] : Next frame")
        print("[B]     : Previous frame")
        print("[P]     : Toggle Play/Pause")
        print("[1-9]   : Scrub dataset (1=Start, 5=Middle, 9=End)")
        print("[Q]     : Quit")
        print("-----------------------\n")

        # Load the initial sample
        self.load_sample()

    def load_sample(self):
        print(f"[{self.current_idx + 1}/{self.max_idx}] Loading sample {self.current_idx}...", end=" ", flush=True)
        try:
            sample = self.dataset[self.current_idx]
            num_boxes = int(np.sum(sample["gt_box_present"]))
            pts_shape = sample["point_clouds"].shape
            print(f"OK ({pts_shape[0]} pts, {num_boxes} boxes)")
            
            # 1. Remove old geometries
            for geom in self.active_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.active_geometries.clear()

            # 2. Build new geometries
            new_geometries = build_geometries_for_sample(sample)

            # 3. Add to visualizer
            for geom in new_geometries:
                # Only reset the camera angle on the very first frame
                self.vis.add_geometry(geom, reset_bounding_box=self.is_first_frame)
                self.active_geometries.append(geom)

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
    
    def manual_next(self, vis=None):
        """Triggered only by the Spacebar."""
        self.playing = False
        return self.next_sample()

    def manual_prev(self, vis=None):
        """Triggered only by the 'B' key."""
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
        description="Visualize AgcoBBoxV1 dataset output in Open3D"
    )
    parser.add_argument("--root-dir", required=True, help="Path to data_root")
    parser.add_argument("--meta-dir", required=True, help="Path to meta_data_dir")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--split-prefix", default="agco")
    parser.add_argument("--sensors", nargs="+", default=["lslidar"])
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--use-intensity", action="store_true")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--require-gravity", action="store_true", default=True)
    parser.add_argument("--apply-r-level-to-points", action="store_true", default=False)
    parser.add_argument("--num-points", type=int, default=40000)
    parser.add_argument("--apply-t-rtk", action="store_true")
    parser.add_argument("--sensor", default="lslidar")

    args = parser.parse_args()

    try:
        dataset = AgcoBBoxV1(
            root_dir=args.root_dir,
            meta_data_dir=args.meta_dir,
            split=args.split,
            split_prefix=args.split_prefix,
            sensors=tuple(args.sensors),
            num_points=args.num_points,
            use_intensity=args.use_intensity,
            augment=args.augment,
            require_gravity_align=args.require_gravity,
            apply_t_rtk=args.apply_t_rtk,
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
        max_samples=args.num_samples
    )
    viewer.run()

if __name__ == "__main__":
    main()