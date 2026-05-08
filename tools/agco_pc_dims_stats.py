#!/usr/bin/env python
"""Per-sensor point-cloud extent statistics for AgcoBBoxV1.

Builds the dataset described by a Pointcept config (default: data.train), iterates
through every sample, and reports per-sensor percentiles for the post-transform
``point_cloud_dims_min`` / ``point_cloud_dims_max`` returned by ``__getitem__``.

Use the output to pick fixed normalization constants per sensor.

Usage:
    python tools/agco_pc_dims_stats.py \
        --config-file configs/agco/det-3detr-v3m1-1-3cls-agco.py \
        --split train

    # Subset for a quick smoke check:
    python tools/agco_pc_dims_stats.py -c <cfg> --split val --max-samples 200
"""

import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from pointcept.datasets import build_dataset
from pointcept.utils.config import Config


def _format_row(name, axis_arr, percentiles):
    pct_vals = np.percentile(axis_arr, percentiles, axis=0)  # (P, 3)
    parts = [f"{name:>10s}"]
    for axis_idx, axis_name in enumerate("xyz"):
        col = "  ".join(f"p{p:02d}={pct_vals[i, axis_idx]:+8.2f}"
                        for i, p in enumerate(percentiles))
        parts.append(f"  {axis_name}: {col}")
    return "\n".join(parts)


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

    # Collect post-transform extents per sensor.
    mins_by_sensor = defaultdict(list)
    maxs_by_sensor = defaultdict(list)
    for i in tqdm(indices, desc=f"{args.split} ({len(indices)} samples)"):
        sample = ds[int(i)]
        sensor = sample["frame_meta"]["sensor"]
        mins_by_sensor[sensor].append(sample["point_cloud_dims_min"])
        maxs_by_sensor[sensor].append(sample["point_cloud_dims_max"])

    print()
    for sensor in sorted(mins_by_sensor.keys()):
        mins = np.stack(mins_by_sensor[sensor], axis=0)   # (N, 3)
        maxs = np.stack(maxs_by_sensor[sensor], axis=0)   # (N, 3)
        extent = maxs - mins
        print("=" * 88)
        print(f"sensor: {sensor}    n={len(mins)}")
        print("-" * 88)
        print("MIN percentiles (per axis):")
        print(_format_row("pc_min", mins, args.percentiles))
        print("MAX percentiles (per axis):")
        print(_format_row("pc_max", maxs, args.percentiles))
        print("EXTENT (max - min) percentiles (per axis):")
        print(_format_row("extent", extent, args.percentiles))
        print()
        # Suggested fixed bounds: 1st-percentile floor & 99th-percentile ceiling,
        # rounded outwards to the nearest 0.5 m for headroom.
        floor = np.floor(np.percentile(mins, 1, axis=0) * 2) / 2
        ceil_ = np.ceil(np.percentile(maxs, 99, axis=0) * 2) / 2
        print(f"Suggested fixed bounds (p01 floor / p99 ceil, rounded to 0.5 m):")
        print(f"  min={floor.tolist()}")
        print(f"  max={ceil_.tolist()}")
        print()


if __name__ == "__main__":
    main()
