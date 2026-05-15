"""Audit: every PTv3-using detection config must end each split's
transform list with ``GridSampleDetection(grid_size=<matches pre-encoder>)``
followed by ``PointSubsampleDetection``.

The PTv3 family of pre-encoders (``PTv3PreEncoder``, ``PTv3UNetPreEncoder``,
``PTv3m3PreEncoder``) operates per-voxel. The training contract is "one
feature per voxel" — without an upstream ``GridSampleDetection`` matching
the encoder's ``grid_size``, multiple input points share a voxel and
``Point.sparsify()``'s internal scatter/reduce silently decides which
features (and, for multi-task, which per-point semseg labels) survive.

This test scans every config under ``configs/agco/`` and
``configs/scannet/``, locates any PTv3-flavoured module in the ``model``
sub-tree, and verifies the train / val / test transform tails.
"""

import glob
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pointcept.utils.config import Config  # noqa: E402

PTV3_TYPES = {"PTv3PreEncoder", "PTv3UNetPreEncoder", "PTv3m3PreEncoder"}
CONFIG_GLOBS = [
    os.path.join(REPO_ROOT, "configs", "agco", "*.py"),
    os.path.join(REPO_ROOT, "configs", "scannet", "*.py"),
]


def _walk_find_ptv3(node):
    """Yield (type_str, grid_size) for every PTv3-family entry in node."""
    if isinstance(node, dict):
        t = node.get("type")
        if isinstance(t, str) and t in PTV3_TYPES:
            yield t, node.get("grid_size")
        for v in node.values():
            yield from _walk_find_ptv3(v)
    elif isinstance(node, (list, tuple)):
        for v in node:
            yield from _walk_find_ptv3(v)


def _transform_types(transform):
    if not isinstance(transform, (list, tuple)):
        return []
    return [t.get("type") for t in transform if isinstance(t, dict)]


def _discover_configs():
    files = []
    for pat in CONFIG_GLOBS:
        files.extend(sorted(glob.glob(pat)))
    return files


@pytest.mark.parametrize("config_path", _discover_configs())
def test_ptv3_grid_sample_contract(config_path):
    try:
        cfg = Config.fromfile(config_path)
    except Exception as e:
        pytest.skip(f"Could not load config: {e}")

    model = cfg.get("model")
    data = cfg.get("data")
    if model is None or data is None:
        pytest.skip("Config has no model/data section")

    ptv3_hits = list(_walk_find_ptv3(model))
    if not ptv3_hits:
        pytest.skip("Config does not use a PTv3 pre-encoder")

    grid_sizes = {gs for _, gs in ptv3_hits if gs is not None}
    assert len(grid_sizes) == 1, (
        f"{os.path.basename(config_path)}: PTv3 modules disagree on grid_size: "
        f"{ptv3_hits}"
    )
    expected_grid = next(iter(grid_sizes))

    violations = []
    for split in ("train", "val", "test"):
        split_cfg = data.get(split)
        if split_cfg is None:
            continue
        transform = split_cfg.get("transform")
        if transform is None:
            violations.append(f"{split}: no transform= list")
            continue
        types = _transform_types(transform)
        if types[-2:] != ["GridSampleDetection", "PointSubsampleDetection"]:
            violations.append(
                f"{split}: tail is {types[-2:]!r}, expected "
                f"['GridSampleDetection', 'PointSubsampleDetection']"
            )
            continue

        grid_entry = transform[-2]
        actual_grid = grid_entry.get("grid_size")
        if actual_grid != expected_grid:
            violations.append(
                f"{split}: GridSampleDetection grid_size={actual_grid!r} "
                f"does not match pre-encoder grid_size={expected_grid!r}"
            )

    assert not violations, (
        f"{os.path.basename(config_path)} violates the PTv3 input contract:\n  "
        + "\n  ".join(violations)
    )
