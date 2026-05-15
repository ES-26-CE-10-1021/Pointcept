"""
GIoU sanity tests for the patched box_util.py.

Covers:
  - helper_computeIntersection: geometric correctness and near-degenerate stability
  - generalized_box3d_iou: range, identical boxes, separated boxes, near-parallel edges
  - Path parity: rotated path agrees with axis-aligned path at yaw=0
  - Mixed-yaw pairs: rotated prediction vs yaw-zero GT stays bounded
  - Negative yaw support
  - Both tensor (needs_grad=True) and cython (needs_grad=False) paths

Run with:
    conda run -n pointcept python -m pytest tests/test_giou_sanity.py -v -s
"""

import math
import os
import sys

import pytest
import torch

_3DETR = os.path.join(os.path.dirname(__file__), "..", "third_party", "3detr")
if _3DETR not in sys.path:
    sys.path.insert(0, _3DETR)

from utils.box_util import (
    generalized_box3d_iou,
    get_3d_box_batch_tensor,
    helper_computeIntersection,
)


# ── helpers ───────────────────────────────────────────────────────────────────


def _corners(centers, sizes, angles):
    """Return (B, K, 8, 3) corners from (B,K,3), (B,K,3), (B,K) tensors."""
    return get_3d_box_batch_tensor(sizes, angles, centers)


def _pair_giou(c1, s1, a1, c2, s2, a2, rotated=True, needs_grad=True):
    """GIoU scalar for a single box pair (B=1, K1=1, K2=1).

    needs_grad=True forces the tensor/Python polygon-clipping path.
    needs_grad=False uses the cython path when box_intersection is compiled.
    """
    corn1 = get_3d_box_batch_tensor(
        torch.tensor([s1], dtype=torch.float32).view(1, 1, 3),
        torch.tensor([[a1]], dtype=torch.float32),   # (1, 1)
        torch.tensor([c1], dtype=torch.float32).view(1, 1, 3),
    )
    corn2 = get_3d_box_batch_tensor(
        torch.tensor([s2], dtype=torch.float32).view(1, 1, 3),
        torch.tensor([[a2]], dtype=torch.float32),   # (1, 1)
        torch.tensor([c2], dtype=torch.float32).view(1, 1, 3),
    )
    nums_k2 = torch.tensor([1], dtype=torch.int64)
    g = generalized_box3d_iou(corn1, corn2, nums_k2, rotated_boxes=rotated, needs_grad=needs_grad)
    return g[0, 0, 0].item()


# ── helper_computeIntersection ────────────────────────────────────────────────


class TestHelperComputeIntersection:
    def test_basic_crossing(self):
        """Horizontal segment crosses vertical clip edge at (0, 0.5)."""
        cp1 = torch.tensor([0.0, 0.0])
        cp2 = torch.tensor([0.0, 1.0])
        s = torch.tensor([-1.0, 0.5])
        e = torch.tensor([1.0, 0.5])
        pt = helper_computeIntersection(cp1, cp2, s, e)
        assert abs(pt[0].item()) < 1e-5
        assert abs(pt[1].item() - 0.5) < 1e-5

    def test_result_on_segment(self):
        """Result must lie on [s, e] regardless of clip orientation."""
        cp1 = torch.tensor([1.0, 0.0])
        cp2 = torch.tensor([0.0, 1.0])  # diagonal clip edge
        s = torch.tensor([0.0, 0.0])
        e = torch.tensor([2.0, 2.0])
        pt = helper_computeIntersection(cp1, cp2, s, e)
        # t = (pt - s) / (e - s) must be in [0, 1]
        t = (pt[0] - s[0]) / (e[0] - s[0])
        assert 0.0 <= t.item() <= 1.0 + 1e-6

    def test_near_parallel_stays_bounded(self):
        """Near-parallel edges must not produce a far-away point."""
        cp1 = torch.tensor([0.0, 0.0])
        cp2 = torch.tensor([1.0, 1e-7])  # nearly horizontal clip edge
        s = torch.tensor([-5.0, 0.1])
        e = torch.tensor([5.0, -0.1])
        pt = helper_computeIntersection(cp1, cp2, s, e)
        # result must lie on the segment (within its bounding box)
        x_lo = min(s[0].item(), e[0].item()) - 1e-4
        x_hi = max(s[0].item(), e[0].item()) + 1e-4
        assert x_lo <= pt[0].item() <= x_hi

    def test_degenerate_both_on_boundary(self):
        """Both s and e on the clip boundary → denom ≈ 0 → return s."""
        cp1 = torch.tensor([0.0, 0.0])
        cp2 = torch.tensor([1.0, 0.0])
        s = torch.tensor([0.5, 0.0])
        e = torch.tensor([1.5, 0.0])
        pt = helper_computeIntersection(cp1, cp2, s, e)
        assert torch.allclose(pt, s, atol=1e-6)


# ── generalized_box3d_iou ─────────────────────────────────────────────────────


class TestGIoURange:
    def test_identical_axis_aligned(self):
        """Identical axis-aligned boxes → GIoU = 1 exactly (AABB == box)."""
        g = _pair_giou([0, 0, 0], [4, 2, 3], 0.0, [0, 0, 0], [4, 2, 3], 0.0, rotated=False)
        assert abs(g - 1.0) < 1e-4, f"expected GIoU=1, got {g}"

    def test_identical_rotated_45(self):
        """Identical 45°-rotated boxes → GIoU > 0 (IoU=1, but AABB > box → GIoU < 1).

        For a box with l=4, w=2, h=3 at 45°:
          enclosing_vol (AABB) = ((l+w)/√2)² * h = (6/√2)² * 3 = 18*3 = 54
          vol = l*w*h = 24
          GIoU = vol/enclosing_vol = 24/54 ≈ 0.444  (< 1 by design for rotated boxes)
        """
        angle = math.pi / 4
        g = _pair_giou([0, 0, 0], [4, 2, 3], angle, [0, 0, 0], [4, 2, 3], angle, rotated=True)
        assert math.isfinite(g), f"GIoU is non-finite"
        assert -1.0 - 1e-4 <= g <= 1.0 + 1e-4, f"GIoU={g} out of range"
        assert g > 0.0, f"GIoU={g}: identical overlapping boxes must have positive GIoU"
        # Verify expected value: vol/enclosing = 24/54 = 4/9
        assert abs(g - 4.0 / 9.0) < 0.01, f"GIoU={g}, expected ≈4/9≈0.444"

    def test_non_overlapping(self):
        """Fully separated boxes → GIoU ∈ [-1, 0)."""
        g = _pair_giou([0, 0, 0], [2, 2, 2], 0.0, [100, 0, 100], [2, 2, 2], 0.0, rotated=False)
        assert -1.0 <= g < 0.0, f"expected GIoU ∈ [-1,0), got {g}"

    def test_partial_overlap(self):
        """Partially overlapping boxes → GIoU ∈ (0, 1)."""
        g = _pair_giou([0, 0, 0], [4, 4, 4], 0.0, [2, 0, 0], [4, 4, 4], 0.0, rotated=False)
        assert 0.0 < g < 1.0, f"expected GIoU ∈ (0,1), got {g}"


class TestGIoUBounds:
    def test_random_boxes_always_in_range(self):
        """GIoU ∈ [-1, 1] for many random box pairs (both rotated and not)."""
        torch.manual_seed(42)
        B, K1, K2 = 4, 16, 8
        centers1 = torch.randn(B, K1, 3) * 5
        sizes1 = torch.rand(B, K1, 3) * 4 + 0.5
        angles1 = torch.rand(B, K1) * math.pi
        centers2 = torch.randn(B, K2, 3) * 5
        sizes2 = torch.rand(B, K2, 3) * 4 + 0.5
        angles2 = torch.rand(B, K2) * math.pi

        corn1 = _corners(centers1, sizes1, angles1)
        corn2 = _corners(centers2, sizes2, angles2)
        nums_k2 = torch.full((B,), K2, dtype=torch.int64)

        for rotated in (False, True):
            g = generalized_box3d_iou(corn1, corn2, nums_k2, rotated_boxes=rotated)
            assert g.isfinite().all(), f"non-finite GIoU (rotated={rotated})"
            assert (g >= -1.0 - 1e-4).all(), f"GIoU < -1 (rotated={rotated}), min={g.min()}"
            assert (g <= 1.0 + 1e-4).all(), f"GIoU > 1 (rotated={rotated}), max={g.max()}"


class TestNearParallelEdgeRegression:
    """Root-cause regression: nearly-parallel clip edges must not blow up GIoU."""

    @pytest.mark.parametrize("base_angle_deg", [0.0, 30.0, 45.0, 60.0, 89.9])
    def test_tiny_relative_rotation(self, base_angle_deg):
        """Two boxes differing by 0.01° → GIoU must be finite, bounded, and positive.

        The old code produced GIoU ≈ -9.7e10 for near-parallel clip edges.
        After the signed-distance fix, GIoU equals approximately vol/enclosing_AABB > 0.
        Note: GIoU = 1 only for axis-aligned identical boxes; for rotated boxes
        GIoU = vol/enclosing_AABB < 1 even with full overlap (e.g., ≈0.48 at 30°/60°,
        ≈0.44 at 45°, ≈0.99 at 89.9°).
        """
        a1 = math.radians(base_angle_deg)
        a2 = math.radians(base_angle_deg + 0.01)
        g = _pair_giou([0, 0, 0], [8, 4, 3], a1, [0, 0, 0], [8, 4, 3], a2, rotated=True)
        assert math.isfinite(g), f"GIoU is non-finite at base={base_angle_deg}°"
        assert -1.0 - 1e-4 <= g <= 1.0 + 1e-4, (
            f"GIoU={g} out of range at base={base_angle_deg}°"
        )
        # Nearly identical fully-overlapping boxes must have positive GIoU.
        # (Before the fix, near-parallel edges produced GIoU ≈ -9.7e10.)
        assert g > 0.0, (
            f"GIoU={g} ≤ 0 for nearly-identical overlapping boxes at base={base_angle_deg}°"
        )

    def test_batch_of_near_parallel_pairs(self):  # noqa: E501 (keep as a block)
        """Batched near-parallel pairs: no non-finite or out-of-range GIoU."""
        angles = torch.linspace(0, math.pi, 32)
        eps = 1e-3
        centers = torch.zeros(1, 32, 3)
        sizes = torch.ones(1, 32, 3) * torch.tensor([6.0, 3.0, 2.5])
        a1 = angles.reshape(1, 32)
        a2 = (angles + eps).reshape(1, 32)

        corn1 = _corners(centers, sizes, a1)
        corn2 = _corners(centers, sizes, a2)
        nums_k2 = torch.tensor([32], dtype=torch.int64)
        g = generalized_box3d_iou(corn1, corn2, nums_k2, rotated_boxes=True)
        assert g.isfinite().all(), f"non-finite values: {g[~g.isfinite()]}"
        assert (g >= -1.0 - 1e-4).all(), f"min GIoU={g.min()}"
        assert (g <= 1.0 + 1e-4).all(), f"max GIoU={g.max()}"


# ── Path-correctness and coverage tests ──────────────────────────────────────


class TestAxisAlignedPath:
    def test_axis_aligned_path_identical_boxes_giou_one(self):
        """Axis-aligned path: identical boxes → GIoU = 1 exactly."""
        g = _pair_giou(
            [0, 0, 0], [4, 2, 3], 0.0,
            [0, 0, 0], [4, 2, 3], 0.0,
            rotated=False,
            needs_grad=True,
        )
        assert abs(g - 1.0) < 1e-4, f"expected GIoU=1, got {g}"

    def test_rotated_path_matches_axis_aligned_path_for_zero_yaw(self):
        """Rotated path must agree with axis-aligned path when yaw=0 for both boxes.

        Regression guard: the AABB pre-check fix and polygon-clipping path must
        give the same result as the simple wh-intersection for axis-aligned boxes.
        """
        args = (
            [0, 0, 0], [4, 2, 3], 0.0,
            [1, 0, 0.5], [4, 2, 3], 0.0,
        )
        g_axis = _pair_giou(*args, rotated=False, needs_grad=True)
        g_rot = _pair_giou(*args, rotated=True, needs_grad=True)
        assert abs(g_axis - g_rot) < 1e-4, (
            f"rotated path ({g_rot:.6f}) disagrees with axis-aligned path ({g_axis:.6f})"
        )


class TestMixedYawAndNegativeYaw:
    def test_rotated_prediction_against_zero_yaw_target_is_bounded(self):
        """Rotated prediction vs yaw-zero GT: finite and bounded.

        Relevant to the criterion's `rotated_boxes=torch.any(gt_box_angles > 0)`
        decision: if GT is axis-aligned but a predicted box has yaw, the rotated
        path must still produce a valid GIoU.
        """
        g = _pair_giou(
            [0, 0, 0], [4, 2, 3], math.radians(15.0),
            [0, 0, 0], [4, 2, 3], 0.0,
            rotated=True,
            needs_grad=True,
        )
        assert math.isfinite(g), f"GIoU is non-finite"
        assert -1.0 - 1e-4 <= g <= 1.0 + 1e-4, f"GIoU={g} out of range"
        assert g > 0.0, f"GIoU={g}: overlapping boxes (15° vs 0°) must be positive"

    def test_negative_yaw_is_supported(self):
        """Negative yaw angles must not accidentally disable the rotated path."""
        g = _pair_giou(
            [0, 0, 0], [4, 2, 3], math.radians(-30.0),
            [0, 0, 0], [4, 2, 3], math.radians(-30.0),
            rotated=True,
            needs_grad=True,
        )
        assert math.isfinite(g), f"GIoU is non-finite"
        assert -1.0 - 1e-4 <= g <= 1.0 + 1e-4, f"GIoU={g} out of range"
        assert g > 0.0, f"GIoU={g}: identical −30° boxes must have positive GIoU"


class TestTensorAndCythonBounds:
    """Both execution paths (tensor/Python and cython) must stay within [-1, 1]."""

    @pytest.mark.parametrize("needs_grad", [True, False])
    def test_giou_bounds_tensor_and_cython(self, needs_grad):
        """Single overlapping rotated pair: finite and in [-1, 1] on both paths."""
        g = _pair_giou(
            [0, 0, 0], [4, 2, 3], math.radians(20.0),
            [0.5, 0, 0.5], [4, 2, 3], math.radians(25.0),
            rotated=True,
            needs_grad=needs_grad,
        )
        assert math.isfinite(g), f"GIoU is non-finite (needs_grad={needs_grad})"
        assert -1.0 - 1e-4 <= g <= 1.0 + 1e-4, (
            f"GIoU={g} out of range (needs_grad={needs_grad})"
        )

    @pytest.mark.parametrize("needs_grad", [True, False])
    def test_near_parallel_bounds_both_paths(self, needs_grad):
        """Near-parallel pair must be bounded on both tensor and cython paths."""
        g = _pair_giou(
            [0, 0, 0], [8, 4, 3], math.radians(45.0),
            [0, 0, 0], [8, 4, 3], math.radians(45.01),
            rotated=True,
            needs_grad=needs_grad,
        )
        assert math.isfinite(g), f"GIoU is non-finite (needs_grad={needs_grad})"
        assert -1.0 - 1e-4 <= g <= 1.0 + 1e-4, (
            f"GIoU={g} out of range (needs_grad={needs_grad})"
        )
        assert g > 0.0, (
            f"GIoU={g}: near-parallel overlapping boxes must be positive "
            f"(needs_grad={needs_grad})"
        )

    def test_tensor_and_cython_roughly_agree_for_rotated_pair(self):
        """Tensor and cython paths must agree to within 1e-3 for a rotated pair.

        Both share the same polygon-clipping algorithm; any larger discrepancy
        indicates a divergence between the two implementations.
        """
        args = (
            [0, 0, 0], [4, 2, 3], math.radians(20.0),
            [0.5, 0, 0.5], [4, 2, 3], math.radians(25.0),
        )
        g_tensor = _pair_giou(*args, rotated=True, needs_grad=True)
        g_cython = _pair_giou(*args, rotated=True, needs_grad=False)
        assert abs(g_tensor - g_cython) < 1e-3, (
            f"tensor path ({g_tensor:.6f}) disagrees with cython path ({g_cython:.6f})"
        )


# ── Criterion path-selection tests ────────────────────────────────────────────


class TestCriterionRotatedBoxesDecision:
    """SetCriterion3DETR must pick the rotated path from config, not from GT angles.

    The old logic `torch.any(gt_box_angles > 0)` silently falls back to the
    axis-aligned path when a batch happens to have all-zero GT yaw (common in
    early training or fully flat scenes).  The fix uses `num_angle_bin > 1`.
    """

    def _make_criterion(self, num_angle_bin):
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "3detr"))
        from pointcept.models.detection_3detr.criterion import SetCriterion3DETR
        return SetCriterion3DETR(num_semcls=3, num_angle_bin=num_angle_bin)

    def test_angle_bin_gt_1_enables_rotated_path(self):
        """num_angle_bin=12 (AGCO/SUNRGBD) → _use_rotated_boxes() is True."""
        c = self._make_criterion(num_angle_bin=12)
        assert c._use_rotated_boxes() is True

    def test_angle_bin_1_disables_rotated_path(self):
        """num_angle_bin=1 (ScanNet) → _use_rotated_boxes() is False."""
        c = self._make_criterion(num_angle_bin=1)
        assert c._use_rotated_boxes() is False

    def test_rotated_path_chosen_even_when_all_gt_angles_zero(self):
        """Regression: yaw-enabled criterion must use rotated path when GT yaw = 0.

        Under the old `torch.any(gt_box_angles > 0)` logic, a batch where all
        GT angles are 0 (e.g., early training, flat terrain) would silently
        choose rotated_boxes=False, producing wrong GIoU for rotated predictions.
        This documents the exact failure mode.
        """
        c = self._make_criterion(num_angle_bin=12)

        gt_angles = torch.zeros(2, 16)
        old_logic = torch.any(gt_angles > 0).item()

        assert old_logic is False           # old code would have picked axis-aligned
        assert c._use_rotated_boxes() is True  # new code correctly picks rotated
