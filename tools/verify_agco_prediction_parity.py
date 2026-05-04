#!/usr/bin/env python3
"""Quick parity checks for AGCO prediction exports.

Validates that GT boxes saved in `gt_boxes_param` are frame-consistent with points.
"""

import argparse
import json
import os

import numpy as np


def camera_to_lidar_np(points):
    pts = np.asarray(points)
    out = pts.copy()
    out[..., 0] = pts[..., 0]
    out[..., 1] = pts[..., 2]
    out[..., 2] = -pts[..., 1]
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate AGCO prediction export parity")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=200)
    args = parser.parse_args()

    manifest_path = os.path.join(args.predictions_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    center_errs = []
    inside_rates = []
    checked = 0

    for entry in manifest[: args.max_samples]:
        rec_path = os.path.join(args.predictions_dir, entry["json_file"])
        pts_path = os.path.join(args.predictions_dir, entry["points_file"])
        with open(rec_path, "r") as f:
            rec = json.load(f)
        pts = np.load(pts_path)[:, :3]

        gt_param = rec.get("gt_boxes_param", [])
        if not gt_param:
            continue

        pmin = pts.min(axis=0)
        pmax = pts.max(axis=0)

        in_bounds = 0
        local_errs = []
        for i, box in enumerate(gt_param):
            ctr = np.asarray(box["center"], dtype=np.float64)
            if np.all(ctr >= pmin) and np.all(ctr <= pmax):
                in_bounds += 1
            if i < len(rec.get("gt_boxes", [])):
                corners = np.asarray(rec["gt_boxes"][i]["box_corners"], dtype=np.float64)
                ctr_from_corners = camera_to_lidar_np(corners).mean(axis=0)
                local_errs.append(float(np.linalg.norm(ctr - ctr_from_corners)))

        if local_errs:
            center_errs.extend(local_errs)
        if gt_param:
            inside_rates.append(in_bounds / len(gt_param))
        checked += 1

    if checked == 0:
        print("No samples with gt_boxes_param found.")
        return

    print(f"Checked samples: {checked}")
    if center_errs:
        print(
            "Center consistency (param vs transformed corners): "
            f"mean={np.mean(center_errs):.6f}, max={np.max(center_errs):.6f}"
        )
    print(
        "GT-center in-bounds rate: "
        f"mean={np.mean(inside_rates):.4f}, min={np.min(inside_rates):.4f}"
    )


if __name__ == "__main__":
    main()
