#!/usr/bin/env python
"""
Visualize fitted ground plane for AGCO point clouds vs bounding boxes.

Fits a plane to the point cloud using PCA (find Z direction), and separately
fits a plane to box centers. Displays both planes and their normal vectors
for easy visual comparison of pitch/roll differences.

Usage:
    python tools/visualize_agco_plane_fit.py \\
        --root /path/to/annotation_root \\
        --sensor lslidar \\
        [--num-samples 5] \\
        [--apply-r-level-inv]
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import RigidTransform as Trans


def fit_ground_plane_pca(points):
    """Fit plane to points using PCA (points should be roughly on ground).

    Returns:
        center: center of points
        normal: normal vector (Z direction) of plane
        plane_points: 4 corner points of a square plane for visualization
    """
    center = points.mean(axis=0)
    centered = points - center

    # PCA
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Normal is the smallest eigenvector (direction of least variance)
    normal = eigenvectors[:, 2]
    normal = normal / np.linalg.norm(normal)

    # Ensure normal points upward (positive Z)
    if normal[2] < 0:
        normal = -normal

    # Create orthogonal vectors for the plane
    if abs(normal[2]) > 0.9:
        u = np.array([1.0, 0.0, 0.0])
    else:
        u = np.array([0.0, 0.0, 1.0])

    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)

    size = 20.0
    plane_points = np.array([
        center + size * u + size * v,
        center - size * u + size * v,
        center - size * u - size * v,
        center + size * u - size * v,
    ])

    return center, normal, plane_points


def normal_to_pitch_roll(normal):
    """Extract pitch and roll from a normal vector (assuming Z points up).

    pitch: rotation about Y (nose up/down)
    roll: rotation about X (wing up/down)
    """
    pitch = np.arctan2(-normal[0], normal[2])
    roll = np.arctan2(normal[1], normal[2])
    return np.degrees(pitch), np.degrees(roll)


def create_plane_mesh(plane_points, color):
    """Create a triangle mesh for a plane."""
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(plane_points)
    mesh.triangles = o3d.utility.Vector3iVector([
        [0, 1, 2],
        [0, 2, 3],
    ])
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def create_normal_lineset(center, normal, length=5.0, color=(1, 0, 0)):
    """Create a line segment showing the normal vector."""
    end = center + length * normal
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector([center, end])
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([color])
    return line_set


def extract_t_rtk_transform(data: dict, sensor: str) -> dict:
    T = np.asarray(data["T_rtk_sensors"][sensor]["matrix"], dtype=np.float32)
    return dict(
        rotation = Rot.from_matrix(T[:3, :3]),
        translation = T[:3,3],
        transformation = Trans.from_matrix(T),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ground plane fit for points vs bounding boxes"
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Path to annotation root",
    )
    parser.add_argument(
        "--sensor",
        default="lslidar",
        help="Sensor name",
    )
    parser.add_argument(
        "--timestamp",
        help="Specific timestamp to analyze",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Max samples to visualize (if no timestamp specified)",
    )
    parser.add_argument(
        "--apply-r-level-inv",
        action="store_true",
        help="Apply R_level.inv() to boxes",
    )
    parser.add_argument(
        "--apply-r-level",
        action="store_true",
        help="Apply R_level to boxes",
    )
    parser.add_argument(
        "--apply-t-rtk",
        action="store_true",
        help="Apply R_level to boxes",
    )

    args = parser.parse_args()

    # Load R_level if needed
    R_level = None
    gravity_path = os.path.join(args.root, "gravity_align.npz")
    if os.path.isfile(gravity_path):
        grav_data = np.load(gravity_path)
        if "R_level" in grav_data.files:
            R_level = Rot.from_matrix(np.asarray(grav_data["R_level"]))
    
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

    if not os.path.isdir(coord_dir) or not os.path.isdir(anno_dir):
        print("Error: missing coord or annotations directory", file=sys.stderr)
        sys.exit(1)

    # Collect timestamps (only those with both coord and annotation files)
    if args.timestamp:
        timestamps = [args.timestamp]
    else:
        coord_files = set(
            os.path.splitext(f)[0]
            for f in os.listdir(coord_dir)
            if f.endswith(".npy")
        )
        anno_files = set(
            os.path.splitext(f)[0]
            for f in os.listdir(anno_dir)
            if f.endswith(".json")
        )
        timestamps = sorted(coord_files & anno_files)[:args.num_samples]

    print(f"Analyzing plane fits for {len(timestamps)} samples\n")

    for ts in timestamps:
        coord_path = os.path.join(coord_dir, f"{ts}.npy")
        anno_path = os.path.join(anno_dir, f"{ts}.json")

        try:
            pts = np.load(coord_path).astype(np.float32)[:, :3]
            
            if args.apply_t_rtk and T_rtk is not None:
                # center = T_rtk["rotation"].inv().apply(center)
                #center = center @ T_rtk["translation"]
                pts = T_rtk["transformation"].apply(pts)
                print("Applied transform to point cloud!")
            
            with open(anno_path) as f:
                data = json.load(f)
            boxes = data.get("annotations", [])

            print(f"{ts}:")

            # Fit plane to points
            pt_center, pt_normal, pt_plane_pts = fit_ground_plane_pca(pts)
            pt_pitch, pt_roll = normal_to_pitch_roll(pt_normal)
            print(f"  Point cloud normal: {pt_normal}")
            print(f"  Point cloud pitch/roll: {pt_pitch:.2f}° / {pt_roll:.2f}°")

            if not boxes:
                print("  No boxes in this sample\n")
                continue

            # Fit plane to box centers
            box_centers = []
            for b in boxes:
                center = np.array(b.get("translation") or b.get("position"))
                if args.apply_r_level_inv and R_level is not None:
                    center = R_level.inv().apply(center)
                if args.apply_r_level and R_level is not None:
                    center = R_level.apply(center)
                box_centers.append(center)
            box_centers = np.array(box_centers)

            box_center, box_normal, box_plane_pts = fit_ground_plane_pca(box_centers)
            box_pitch, box_roll = normal_to_pitch_roll(box_normal)
            print(f"  Box normal: {box_normal}")
            print(f"  Box pitch/roll: {box_pitch:.2f}° / {box_roll:.2f}°")

            # Compute differences
            dot_prod = np.clip(np.dot(pt_normal, box_normal), -1, 1)
            angle_rad = np.arccos(dot_prod)
            angle_deg = np.degrees(angle_rad)
            print(f"\n  → Angle between normals: {angle_deg:.2f}°")
            print(f"  → Pitch difference: {pt_pitch - box_pitch:.2f}°")
            print(f"  → Roll difference: {pt_roll - box_roll:.2f}°\n")

            # Visualization
            geometries = []

            # Points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts)
            pcd.colors = o3d.utility.Vector3dVector(
                np.ones((pts.shape[0], 3)) * 0.7
            )
            geometries.append(pcd)

            # Planes
            pt_plane = create_plane_mesh(pt_plane_pts, [0.2, 0.8, 0.2])  # Green
            box_plane = create_plane_mesh(box_plane_pts, [0.8, 0.2, 0.2])  # Red
            geometries.append(pt_plane)
            geometries.append(box_plane)

            # Normal vectors
            pt_normal_line = create_normal_lineset(pt_center, pt_normal, length=10.0, color=(0, 1, 0))
            box_normal_line = create_normal_lineset(box_center, box_normal, length=10.0, color=(1, 0, 0))
            geometries.append(pt_normal_line)
            geometries.append(box_normal_line)

            # Legend
            title = f"{ts} | Green=PointCloud, Red=Boxes | Angle={angle_deg:.1f}° | Pitch_diff={pt_pitch - box_pitch:.1f}°"
            o3d.visualization.draw_geometries(
                geometries,
                window_name=title,
                width=1280,
                height=960,
            )

        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
