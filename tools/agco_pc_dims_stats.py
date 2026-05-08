#!/usr/bin/env python
"""Per-sensor extent statistics for AgcoBBoxV1.

Builds the dataset described by a Pointcept config (default: data.train), iterates
through every sample, and reports per-sensor percentiles to help pick fixed
normalization bounds for the dataset's ``fixed_pc_dims`` kwarg.

Two bound-source modes:

  ``--bounds-source points`` (default)
      Percentiles of the post-transform ``point_cloud_dims_min`` /
      ``point_cloud_dims_max`` returned by ``__getitem__`` (i.e. the actual
      point envelope of each scene). Conservative: bounds cover all points.

  ``--bounds-source box_centers``
      Percentiles of the *cloud of GT box centers* across all samples (using
      ``gt_box_present`` to mask padded entries). Tighter ("box-of-interest")
      bounds — boxes never appear far outside their distribution, so using
      these as normalization gives the size head more dynamic range. Apply
      ``--box-headroom`` to pad outward for the box half-extents.

Usage:
    python tools/agco_pc_dims_stats.py -c configs/agco/det-3detr-v3m2-1-3cls-agco.py
    python tools/agco_pc_dims_stats.py -c <cfg> --bounds-source box_centers --box-headroom 3.0
"""

import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from pointcept.datasets import build_dataset
from pointcept.utils.config import Config


def _format_row(name, arr, percentiles):
    pct_vals = np.percentile(arr, percentiles, axis=0)  # (P, 3)
    parts = [f"{name:>10s}"]
    for axis_idx, axis_name in enumerate("xyz"):
        col = "  ".join(f"p{p:02d}={pct_vals[i, axis_idx]:+8.2f}"
                        for i, p in enumerate(percentiles))
        parts.append(f"  {axis_name}: {col}")
    return "\n".join(parts)


def _floor_to(arr, step=0.5):
    return np.floor(arr / step) * step


def _ceil_to(arr, step=0.5):
    return np.ceil(arr / step) * step


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config-file", required=True)
    parser.add_argument("--split", default="train",
                        choices=["train", "val", "test"])
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Stop after N samples (random order). Default: all.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--percentiles", type=int, nargs="+",
                        default=[1, 5, 50, 95, 99])
    parser.add_argument("--bounds-source", choices=["points", "box_centers"],
                        default="points",
                        help="Use point envelope (default) or GT box centers "
                             "to derive suggested bounds.")
    parser.add_argument("--box-headroom", type=float, default=0.0,
                        help="When --bounds-source=box_centers, pad each "
                             "axis bound outward by this many metres to "
                             "absorb box half-extents. E.g. 3.0 covers a "
                             "vehicle up to ~6 m on that axis.")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_file)
    if args.split not in cfg.data:
        raise SystemExit(f"Config has no data.{args.split} block.")

    ds_cfg = cfg.data[args.split]
    print(f"Building dataset: type={ds_cfg.get('type')} split={ds_cfg.get('split')}")
    ds = build_dataset(ds_cfg)
    n = len(ds)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(n)
    if args.max_samples is not None and args.max_samples < n:
        rng.shuffle(indices)
        indices = indices[:args.max_samples]

    # Collect per-sensor stats.
    mins_by_sensor = defaultdict(list)        # per-sample point min  (N_s, 3)
    maxs_by_sensor = defaultdict(list)        # per-sample point max  (N_s, 3)
    box_centers_by_sensor = defaultdict(list) # all GT centers        (N_box, 3)
    box_count_by_sensor = defaultdict(int)
    sample_count_by_sensor = defaultdict(int)
    for i in tqdm(indices, desc=f"{args.split} ({len(indices)} samples)"):
        sample = ds[int(i)]
        sensor = sample["frame_meta"]["sensor"]
        sample_count_by_sensor[sensor] += 1
        mins_by_sensor[sensor].append(sample["point_cloud_dims_min"])
        maxs_by_sensor[sensor].append(sample["point_cloud_dims_max"])
        present = sample["gt_box_present"].astype(bool)
        if present.any():
            centers = sample["gt_box_centers"][present]  # (k, 3)
            box_centers_by_sensor[sensor].append(centers)
            box_count_by_sensor[sensor] += int(present.sum())

    print()
    for sensor in sorted(mins_by_sensor.keys()):
        mins = np.stack(mins_by_sensor[sensor], axis=0)   # (N, 3)
        maxs = np.stack(maxs_by_sensor[sensor], axis=0)   # (N, 3)
        extent = maxs - mins
        n_samples = sample_count_by_sensor[sensor]
        n_boxes = box_count_by_sensor[sensor]
        print("=" * 88)
        print(f"sensor: {sensor}    samples={n_samples}    gt_boxes={n_boxes}")
        print("-" * 88)
        print("POINT pc_min percentiles (per axis):")
        print(_format_row("pc_min", mins, args.percentiles))
        print("POINT pc_max percentiles (per axis):")
        print(_format_row("pc_max", maxs, args.percentiles))
        print("POINT extent (max - min) percentiles (per axis):")
        print(_format_row("extent", extent, args.percentiles))

        if box_centers_by_sensor[sensor]:
            centers_all = np.concatenate(box_centers_by_sensor[sensor], axis=0)  # (N_box, 3)
            print()
            print("BOX-CENTER percentiles (per axis, across all GT boxes):")
            print(_format_row("box_ctr", centers_all, args.percentiles))
        else:
            centers_all = None
            print()
            print("BOX-CENTER percentiles: (no GT boxes for this sensor)")

        # Suggested bounds.
        print()
        if args.bounds_source == "points" or centers_all is None:
            floor = _floor_to(np.percentile(mins, 1, axis=0))
            ceil_ = _ceil_to(np.percentile(maxs, 99, axis=0))
            label = "points (p01 floor / p99 ceil, 0.5 m round)"
        else:
            base_min = np.percentile(centers_all, 1, axis=0) - args.box_headroom
            base_max = np.percentile(centers_all, 99, axis=0) + args.box_headroom
            floor = _floor_to(base_min)
            ceil_ = _ceil_to(base_max)
            label = (
                f"box_centers (p01−{args.box_headroom}m / p99+{args.box_headroom}m, "
                f"0.5 m round)"
            )
        print(f"Suggested fixed bounds [{label}]:")
        print(f"  min={floor.tolist()}")
        print(f"  max={ceil_.tolist()}")
        print()


if __name__ == "__main__":
    main()
