"""
Visualize VoteNet-style ScanNet detection data with Open3D.

Usage examples:
    python tools/visualize_scannet_detection.py --data_dir /path/to/scannet_train_detection_data --scene scene0280_00
    python tools/visualize_scannet_detection.py --data_dir /path/to/data --scene scene0280_00 --color sem --show_boxes
    python tools/visualize_scannet_detection.py --data_dir /path/to/data --scene scene0280_00 --color ins
"""

import argparse
import os
import numpy as np
import open3d as o3d


# 18 detection class names (NYU-40 IDs: 3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39)
DETECTION_CLASS_NAMES = [
    "cabinet", "bed", "chair", "sofa", "table", "door", "window",
    "bookshelf", "picture", "counter", "desk", "curtain",
    "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture",
]
DETECTION_NYU40_IDS = np.array(
    [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
)


def _label_colors(labels, n_colors=256):
    """Map integer labels to RGB colours using a fixed palette."""
    np.random.seed(42)
    palette = np.random.rand(n_colors, 3)
    palette[0] = [0.7, 0.7, 0.7]  # label 0 → grey
    labels = labels.astype(int) % n_colors
    return palette[labels]


def _box_lineset(center, size, color=(1, 0, 0)):
    """Return an Open3D LineSet for one axis-aligned box (cx cy cz dx dy dz)."""
    cx, cy, cz = center
    dx, dy, dz = size[0] / 2, size[1] / 2, size[2] / 2
    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # bottom
        [4, 5], [5, 6], [6, 7], [7, 4],  # top
        [0, 4], [1, 5], [2, 6], [3, 7],  # verticals
    ]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(edges)
    ls.colors = o3d.utility.Vector3dVector([color] * len(edges))
    return ls


# Distinct colours for the 18 detection classes
_BOX_PALETTE = np.array([
    [1.00, 0.20, 0.20], [0.20, 0.80, 0.20], [0.20, 0.20, 1.00],
    [1.00, 0.80, 0.00], [0.80, 0.00, 0.80], [0.00, 0.80, 0.80],
    [1.00, 0.50, 0.00], [0.50, 1.00, 0.00], [0.00, 0.50, 1.00],
    [1.00, 0.00, 0.50], [0.50, 0.00, 1.00], [0.00, 1.00, 0.50],
    [0.80, 0.40, 0.20], [0.20, 0.80, 0.40], [0.40, 0.20, 0.80],
    [0.80, 0.80, 0.20], [0.20, 0.40, 0.80], [0.80, 0.20, 0.40],
])


def main():
    parser = argparse.ArgumentParser(
        description="Visualize ScanNet VoteNet detection data with Open3D."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to scannet_train_detection_data/ directory.",
    )
    parser.add_argument(
        "--scene", required=True,
        help="Scene name, e.g. scene0280_00.",
    )
    parser.add_argument(
        "--color", choices=["rgb", "sem", "ins"], default="rgb",
        help=(
            "How to colour the point cloud: "
            "'rgb' (raw colour), 'sem' (semantic class), 'ins' (instance)."
        ),
    )
    parser.add_argument(
        "--show_boxes", action="store_true",
        help="Overlay GT bounding boxes.",
    )
    parser.add_argument(
        "--subsample", type=int, default=None,
        help="Randomly subsample to this many points for faster rendering (default: all).",
    )
    args = parser.parse_args()

    prefix = os.path.join(args.data_dir, args.scene)

    # Load files
    vert = np.load(prefix + "_vert.npy")          # (N, 6) xyzrgb
    bbox = np.load(prefix + "_bbox.npy")           # (M, 7) cx cy cz dx dy dz cls
    ins_label = np.load(prefix + "_ins_label.npy") # (N,)
    sem_label = np.load(prefix + "_sem_label.npy") # (N,)

    print(f"Scene : {args.scene}")
    print(f"Points: {vert.shape[0]}")
    print(f"Boxes : {bbox.shape[0]}")

    xyz = vert[:, 0:3].astype(np.float64)
    rgb = vert[:, 3:6].astype(np.float64) / 255.0

    # Optional subsampling
    if args.subsample is not None and args.subsample < xyz.shape[0]:
        idx = np.random.choice(xyz.shape[0], args.subsample, replace=False)
        xyz = xyz[idx]
        rgb = rgb[idx]
        ins_label = ins_label[idx]
        sem_label = sem_label[idx]
        print(f"Subsampled to {args.subsample} points.")

    # Build point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    if args.color == "rgb":
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    elif args.color == "sem":
        pcd.colors = o3d.utility.Vector3dVector(_label_colors(sem_label))
    else:  # ins
        pcd.colors = o3d.utility.Vector3dVector(_label_colors(ins_label))

    geometries = [pcd]

    # Bounding boxes
    if args.show_boxes:
        nyu40_to_det = {int(nid): i for i, nid in enumerate(DETECTION_NYU40_IDS)}
        for i, box in enumerate(bbox):
            center, size, cls_nyu40 = box[0:3], box[3:6], int(box[6])
            cls_idx = nyu40_to_det.get(cls_nyu40, 0)
            color = _BOX_PALETTE[cls_idx % len(_BOX_PALETTE)].tolist()
            name = DETECTION_CLASS_NAMES[cls_idx] if cls_nyu40 in nyu40_to_det else "unknown"
            print(f"  box {i:2d}: {name:20s}  center={center}  size={size}")
            geometries.append(_box_lineset(center, size, color))

    print("\nControls: drag to rotate, scroll to zoom, Ctrl+drag to pan, Q to quit.")
    title = f"{args.scene}  |  color={args.color}  |  boxes={'on' if args.show_boxes else 'off'}"
    o3d.visualization.draw_geometries(geometries, window_name=title)


if __name__ == "__main__":
    main()
