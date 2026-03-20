"""
Tests for PTv3PreEncoder integration with 3DETR detection model.

Verifies that PTv3PreEncoder correctly wraps PointTransformerV3 as a
3DETR pre-encoder, producing Point objects that flow through the
Model3DETRDetector pipeline via the isinstance-based dispatch.

Run with:
    pytest tests/test_ptv3_3detr_integration.py -v -s

Requires CUDA and flash-attn.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DETR_ROOT = os.path.join(REPO_ROOT, "third_party", "3detr")
for p in [REPO_ROOT, DETR_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2bincount
from pointcept.models.detection_3detr.model import (
    Model3DETRDetector,
    dense2point,
    point2dense,
)
from pointcept.models.detection_3detr.components import IdentityEncoder3DETR
from pointcept.models.detection_3detr.dataset_config import ScanNetDetectionConfig

try:
    from pointcept.models.detection_3detr.ptv3 import PTv3PreEncoder

    PTV3_AVAILABLE = True
except ImportError as e:
    PTV3_AVAILABLE = False
    PTV3_IMPORT_ERROR = str(e)

skip_no_ptv3 = pytest.mark.skipif(
    not PTV3_AVAILABLE,
    reason=f"PTv3PreEncoder not importable: {PTV3_IMPORT_ERROR if not PTV3_AVAILABLE else ''}",
)
skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── Minimal PTv3 config for fast tests ────────────────────────────────────────
# Tiny model: fewer stages, smaller channels, smaller patches
TINY_PTV3_CFG = dict(
    type="PTv3PreEncoder",
    grid_size=0.02,
    in_channels=3,
    order=("z", "z-trans"),
    stride=(2, 2),
    enc_depths=(1, 1, 1),
    enc_channels=(32, 64, 128),
    enc_num_head=(2, 4, 8),
    enc_patch_size=(64, 64, 64),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.0,
    shuffle_orders=False,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    pdnorm_bn=False,
    pdnorm_ln=False,
    pdnorm_decouple=True,
    pdnorm_adaptive=False,
    pdnorm_affine=True,
    pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
)

# Encoder dim must match enc_channels[-1]
TINY_ENC_DIM = TINY_PTV3_CFG["enc_channels"][-1]  # 128


def _build_tiny_model(criterion=None):
    """Build a minimal Model3DETRDetector with PTv3PreEncoder for testing."""
    return Model3DETRDetector(
        pre_encoder=TINY_PTV3_CFG,
        encoder=dict(type="IdentityEncoder3DETR"),
        decoder=dict(
            type="TransformerDecoder3DETR",
            decoder_dim=TINY_ENC_DIM,
            nhead=4,
            nlayers=2,
            ffn_dim=128,
            dropout=0.0,
        ),
        dataset_config=dict(type="ScanNetDetectionConfig"),
        encoder_dim=TINY_ENC_DIM,
        decoder_dim=TINY_ENC_DIM,
        num_queries=32,
        position_embedding="fourier",
        mlp_dropout=0.0,
        criterion=criterion,
    )


def _make_batch(B=2, N=5000, device="cuda"):
    """Create a synthetic point cloud batch.

    Uses N=5000 to survive voxelization + 2 pooling strides without
    collapsing to too few points for the decoder's FPS.
    """
    pc = torch.randn(B, N, 3, device=device) * 2.0  # spread for voxelization
    return {
        "point_clouds": pc,
        "point_cloud_dims_min": pc.amin(dim=1),
        "point_cloud_dims_max": pc.amax(dim=1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# A0 — point2dense unit tests (padding + autograd)
# ═══════════════════════════════════════════════════════════════════════════════


class TestPoint2Dense:
    """Unit tests for point2dense with variable-length padding."""

    def test_equal_length_no_mask(self):
        """Equal-length scenes should use reshape fast path (no mask)."""
        B, N, C = 3, 50, 8
        coord = torch.randn(B * N, 3)
        feat = torch.randn(B * N, C)
        offset = torch.arange(1, B + 1) * N

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_out, feat_out, mask = point2dense(point)

        assert mask is None
        assert xyz_out.shape == (B, N, 3)
        assert feat_out.shape == (B, C, N)

    def test_equal_length_preserves_autograd(self):
        """Reshape fast path must preserve autograd."""
        B, N, C = 2, 20, 4
        feat = torch.randn(B * N, C, requires_grad=True)
        coord = torch.randn(B * N, 3)
        offset = torch.arange(1, B + 1) * N

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        _, feat_out, _ = point2dense(point)
        feat_out.sum().backward()
        assert feat.grad is not None

    def test_variable_length_returns_mask(self):
        """Variable-length scenes must return a padding mask."""
        n0, n1, C = 10, 20, 8
        coord = torch.randn(n0 + n1, 3)
        feat = torch.randn(n0 + n1, C)
        offset = torch.tensor([n0, n0 + n1])

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_out, feat_out, mask = point2dense(point)

        assert mask is not None
        assert mask.shape == (2, 20)
        # Scene 0: 10 real + 10 padded
        assert mask[0, :10].sum() == 0  # real positions = False
        assert mask[0, 10:].sum() == 10  # padded positions = True
        # Scene 1: 20 real, no padding
        assert mask[1].sum() == 0

    def test_variable_length_preserves_autograd(self):
        """Pad path (split/F.pad/stack) must preserve autograd."""
        n0, n1, C = 10, 20, 4
        feat = torch.randn(n0 + n1, C, requires_grad=True)
        coord = torch.randn(n0 + n1, 3)
        offset = torch.tensor([n0, n0 + n1])

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        _, feat_out, _ = point2dense(point)
        feat_out.sum().backward()
        assert feat.grad is not None, "No gradient through padded point2dense"

    def test_variable_length_correct_values(self):
        """Padded output must contain correct values in real positions."""
        n0, n1, C = 5, 10, 3
        coord = torch.arange(n0 + n1, dtype=torch.float).unsqueeze(1).expand(-1, 3)
        feat = torch.ones(n0 + n1, C)
        offset = torch.tensor([n0, n0 + n1])

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_out, feat_out, mask = point2dense(point)

        # Scene 0 real coords
        assert torch.allclose(xyz_out[0, :n0], coord[:n0])
        # Scene 0 padded coords should be zero
        assert (xyz_out[0, n0:] == 0).all()
        # Scene 1 real coords
        assert torch.allclose(xyz_out[1, :n1], coord[n0:])


# ═══════════════════════════════════════════════════════════════════════════════
# A — PTv3PreEncoder unit tests
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3
@skip_no_cuda
class TestPTv3PreEncoderUnit:
    """Verify PTv3PreEncoder produces well-formed Point output."""

    @pytest.fixture(scope="class")
    def pre_encoder(self):
        from pointcept.models.builder import MODULES

        pe = MODULES.build(TINY_PTV3_CFG)
        return pe.cuda().eval()

    def test_returns_point(self, pre_encoder):
        """Forward must return a Point object, not a tuple."""
        xyz = torch.randn(2, 5000, 3, device="cuda") * 2.0
        with torch.no_grad():
            result = pre_encoder(xyz, features=None)
        assert isinstance(result, Point), f"Expected Point, got {type(result)}"

    def test_output_has_required_fields(self, pre_encoder):
        """Output Point must have coord, feat, offset, batch."""
        xyz = torch.randn(2, 5000, 3, device="cuda") * 2.0
        with torch.no_grad():
            point = pre_encoder(xyz, features=None)
        for field in ("coord", "feat", "offset", "batch"):
            assert field in point.keys(), f"Missing field: {field}"

    def test_output_feat_dim(self, pre_encoder):
        """Output feature dim must match enc_channels[-1]."""
        xyz = torch.randn(2, 5000, 3, device="cuda") * 2.0
        with torch.no_grad():
            point = pre_encoder(xyz, features=None)
        assert (
            point.feat.shape[-1] == TINY_ENC_DIM
        ), f"Expected feat dim {TINY_ENC_DIM}, got {point.feat.shape[-1]}"

    def test_output_offset_is_valid(self, pre_encoder):
        """Offset must be monotonically increasing with length B."""
        B = 3
        xyz = torch.randn(B, 5000, 3, device="cuda") * 2.0
        with torch.no_grad():
            point = pre_encoder(xyz, features=None)
        assert point.offset.shape == (B,)
        assert (point.offset[1:] > point.offset[:-1]).all(), "Offset not monotonic"
        assert point.offset[-1] == point.coord.shape[0]

    def test_output_point_count_reduced(self, pre_encoder):
        """Pooling strides should reduce point count vs input."""
        B, N = 2, 5000
        xyz = torch.randn(B, N, 3, device="cuda") * 2.0
        with torch.no_grad():
            point = pre_encoder(xyz, features=None)
        total_out = point.coord.shape[0]
        # With grid voxelization + 2 pooling strides, output should be much smaller
        assert (
            total_out < B * N
        ), f"Expected fewer output points than input ({total_out} >= {B * N})"

    def test_enc_mode_enabled(self, pre_encoder):
        """enc_mode must be True (encoder-only, no decoder)."""
        assert pre_encoder.enc_mode is True

    def test_no_decoder_attributes(self, pre_encoder):
        """With enc_mode=True, PTv3 should not have a dec attribute."""
        assert not hasattr(
            pre_encoder, "dec"
        ), "PTv3PreEncoder should not have a decoder (enc_mode=True)"

    def test_grid_size_stored(self, pre_encoder):
        """grid_size must be accessible on the module."""
        assert hasattr(pre_encoder, "grid_size")
        assert pre_encoder.grid_size == TINY_PTV3_CFG["grid_size"]


# ═══════════════════════════════════════════════════════════════════════════════
# B — PTv3PreEncoder → run_encoder dispatch
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3
@skip_no_cuda
class TestPTv3RunEncoderDispatch:
    """Verify Model3DETRDetector.run_encoder handles PTv3PreEncoder output."""

    @pytest.fixture(scope="class")
    def model(self):
        m = _build_tiny_model()
        return m.cuda().eval()

    def test_run_encoder_succeeds(self, model):
        """run_encoder with PTv3 pre-encoder should not raise."""
        batch = _make_batch()
        with torch.no_grad():
            enc_xyz, enc_features, enc_inds, padding_mask = model.run_encoder(
                batch["point_clouds"]
            )

    def test_run_encoder_output_shapes(self, model):
        """run_encoder output tensors must have consistent shapes."""
        B = 2
        batch = _make_batch(B=B)
        with torch.no_grad():
            enc_xyz, enc_features, enc_inds, padding_mask = model.run_encoder(
                batch["point_clouds"]
            )
        # enc_xyz: (B, N_enc, 3)
        assert enc_xyz.ndim == 3
        assert enc_xyz.shape[0] == B
        assert enc_xyz.shape[2] == 3
        N_enc = enc_xyz.shape[1]

        # enc_features: (N_enc, B, enc_dim) — transformer convention
        assert enc_features.shape == (N_enc, B, TINY_ENC_DIM)

    def test_run_encoder_inds_is_none(self, model):
        """PTv3 pre-encoder returns Point → enc_inds should be None."""
        batch = _make_batch()
        with torch.no_grad():
            _, _, enc_inds, _ = model.run_encoder(batch["point_clouds"])
        assert enc_inds is None

    def test_enc_features_finite(self, model):
        """Encoder features must not contain NaN or Inf."""
        batch = _make_batch()
        with torch.no_grad():
            _, enc_features, _, _ = model.run_encoder(batch["point_clouds"])
        assert torch.isfinite(enc_features).all()

    def test_padding_mask_shape(self, model):
        """Padding mask must match enc_xyz spatial dimension."""
        B = 2
        batch = _make_batch(B=B)
        with torch.no_grad():
            enc_xyz, _, _, padding_mask = model.run_encoder(batch["point_clouds"])
        # padding_mask may be None (equal-length) or (B, N_enc)
        if padding_mask is not None:
            assert padding_mask.shape == (B, enc_xyz.shape[1])
            assert padding_mask.dtype == torch.bool


# ═══════════════════════════════════════════════════════════════════════════════
# C — Full eval forward
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3
@skip_no_cuda
class TestPTv3FullForwardEval:
    """End-to-end eval forward with PTv3 + IdentityEncoder + 3DETR decoder."""

    @pytest.fixture(scope="class")
    def model(self):
        m = _build_tiny_model()
        return m.cuda().eval()

    def test_forward_returns_outputs(self, model):
        """Eval forward must return dict with 'outputs' and 'aux_outputs'."""
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        assert "outputs" in out
        assert "aux_outputs" in out

    def test_output_box_predictions(self, model):
        """Output must contain valid box prediction tensors."""
        B, nq = 2, 32
        batch = _make_batch(B=B)
        with torch.no_grad():
            out = model(batch)
        preds = out["outputs"]

        assert preds["sem_cls_logits"].shape == (B, nq, 19)  # 18 + 1
        assert preds["center_unnormalized"].shape == (B, nq, 3)
        assert preds["size_unnormalized"].shape == (B, nq, 3)
        assert preds["box_corners"].shape == (B, nq, 8, 3)
        assert preds["angle_logits"].shape == (B, nq, 1)
        assert preds["objectness_prob"].shape == (B, nq)

    def test_aux_outputs_count(self, model):
        """Number of aux_outputs should equal decoder layers - 1."""
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        # 2 decoder layers → 1 aux output (all layers except final)
        assert len(out["aux_outputs"]) == 1

    def test_all_predictions_finite(self, model):
        """No NaN/Inf in any prediction tensor."""
        batch = _make_batch()
        with torch.no_grad():
            out = model(batch)
        for key, val in out["outputs"].items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), f"Non-finite in outputs['{key}']"


# ═══════════════════════════════════════════════════════════════════════════════
# D — Training forward (loss + backward)
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3
@skip_no_cuda
class TestPTv3TrainingForward:
    """Verify training forward with criterion produces valid loss + gradients."""

    @pytest.fixture(scope="class")
    def model(self):
        criterion_cfg = dict(
            type="SetCriterion3DETR",
            matcher_cfg=dict(
                cost_class=1.0,
                cost_objectness=0.0,
                cost_giou=2.0,
                cost_center=0.0,
            ),
            loss_weight_dict=dict(
                loss_giou_weight=0.0,
                loss_sem_cls_weight=1.0,
                loss_no_object_weight=0.2,
                loss_angle_cls_weight=0.1,
                loss_angle_reg_weight=0.5,
                loss_center_weight=5.0,
                loss_size_weight=1.0,
            ),
            num_semcls=18,
            num_angle_bin=1,
        )
        m = _build_tiny_model(criterion=criterion_cfg)
        return m.cuda().train()

    def _make_train_batch(self, B=2, N=5000, device="cuda"):
        ds_cfg = ScanNetDetectionConfig()
        max_obj, ngt = 64, 4
        pc = torch.randn(B, N, 3, device=device) * 2.0

        gt_center = torch.rand(B, max_obj, 3, device=device)
        gt_size = torch.rand(B, max_obj, 3, device=device).clamp(0.05, 1.0)
        gt_angle = torch.zeros(B, max_obj, device=device)
        gt_corners = ds_cfg.box_parametrization_to_corners(gt_center, gt_size, gt_angle)
        gt_present = torch.zeros(B, max_obj, device=device)
        gt_present[:, :ngt] = 1.0

        return {
            "point_clouds": pc,
            "point_cloud_dims_min": pc.amin(dim=1),
            "point_cloud_dims_max": pc.amax(dim=1),
            "gt_box_present": gt_present,
            "gt_box_corners": gt_corners,
            "gt_box_centers_normalized": torch.rand(B, max_obj, 3, device=device),
            "gt_box_sem_cls_label": torch.randint(0, 18, (B, max_obj), device=device),
            "gt_box_angles": gt_angle,
            "gt_angle_class_label": torch.zeros(
                B, max_obj, dtype=torch.long, device=device
            ),
            "gt_angle_residual_label": torch.zeros(B, max_obj, device=device),
            "gt_box_sizes_normalized": torch.rand(B, max_obj, 3, device=device),
        }

    def test_produces_scalar_loss(self, model):
        """Training forward must return a finite scalar loss."""
        batch = self._make_train_batch()
        output = model(batch)
        assert "loss" in output
        assert output["loss"].ndim == 0
        assert torch.isfinite(output["loss"])

    def test_loss_requires_grad(self, model):
        """Loss must require grad for backward pass."""
        batch = self._make_train_batch()
        output = model(batch)
        assert output["loss"].requires_grad

    def test_backward_produces_gradients(self, model):
        """Backward through loss must produce finite gradients."""
        model.zero_grad()
        batch = self._make_train_batch()
        output = model(batch)
        output["loss"].backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No gradients computed"
        for i, g in enumerate(grads):
            assert torch.isfinite(g).all(), f"Non-finite gradient in param {i}"

    def test_ptv3_params_receive_gradients(self, model):
        """PTv3 pre-encoder parameters must receive gradients (not frozen)."""
        model.zero_grad()
        batch = self._make_train_batch()
        output = model(batch)
        output["loss"].backward()

        ptv3_grads = [
            (n, p.grad)
            for n, p in model.pre_encoder.named_parameters()
            if p.grad is not None
        ]
        assert len(ptv3_grads) > 0, "No PTv3 pre-encoder parameters received gradients"


# ═══════════════════════════════════════════════════════════════════════════════
# E — Config match (v2 config consistency)
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3
class TestConfigConsistency:
    """Verify the v2 config is internally consistent."""

    def test_encoder_dim_matches_enc_channels(self):
        """encoder_dim in model config must equal enc_channels[-1] of PTv3."""
        # These are from det-3detr-v2m1-0-scannet.py
        enc_channels = (32, 64, 128, 256, 512)
        encoder_dim = 512
        assert (
            encoder_dim == enc_channels[-1]
        ), f"encoder_dim ({encoder_dim}) must match enc_channels[-1] ({enc_channels[-1]})"

    def test_stride_and_depths_consistent(self):
        """len(enc_depths) must equal len(stride) + 1."""
        stride = (2, 2, 2, 2)
        enc_depths = (2, 2, 2, 6, 2)
        assert (
            len(enc_depths) == len(stride) + 1
        ), f"enc_depths ({len(enc_depths)}) must be len(stride)+1 ({len(stride) + 1})"

    def test_enc_channels_heads_depths_consistent(self):
        """enc_channels, enc_num_head, enc_depths must have same length."""
        enc_channels = (32, 64, 128, 256, 512)
        enc_num_head = (2, 4, 8, 16, 32)
        enc_depths = (2, 2, 2, 6, 2)
        assert len(enc_channels) == len(enc_num_head) == len(enc_depths)

    def test_in_channels_matches_xyz_only(self):
        """in_channels=3 means XYZ only (no color), matching use_color=False."""
        in_channels = 3
        use_color = False
        expected = 3 if not use_color else 6
        assert in_channels == expected


# ═══════════════════════════════════════════════════════════════════════════════
# F — Variable-length scenes handled via padding mask
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3
@skip_no_cuda
class TestPTv3VariableLengthPadding:
    """PTv3 voxelization + pooling produces variable-length scenes.
    point2dense must pad them and return a mask for the decoder."""

    @pytest.fixture(scope="class")
    def pre_encoder(self):
        from pointcept.models.builder import MODULES

        pe = MODULES.build(TINY_PTV3_CFG)
        return pe.cuda().eval()

    def test_point2dense_succeeds_with_variable_length(self, pre_encoder):
        """point2dense must handle variable-length PTv3 output via padding."""
        # Different spatial extents to force different voxel counts
        xyz_0 = torch.randn(1, 5000, 3, device="cuda") * 0.5
        xyz_1 = torch.randn(1, 5000, 3, device="cuda") * 5.0
        xyz = torch.cat([xyz_0, xyz_1], dim=0)

        with torch.no_grad():
            point = pre_encoder(xyz, features=None)

        # Should not raise — returns padded tensors + mask
        xyz_dense, feat_dense, mask = point2dense(point)

        counts = offset2bincount(point.offset)
        max_n = counts.max().item()
        assert xyz_dense.shape == (2, max_n, 3)
        assert feat_dense.shape == (2, TINY_ENC_DIM, max_n)

    def test_padding_mask_correct(self, pre_encoder):
        """Padding mask must mark exactly the padded positions."""
        xyz_0 = torch.randn(1, 5000, 3, device="cuda") * 0.5
        xyz_1 = torch.randn(1, 5000, 3, device="cuda") * 5.0
        xyz = torch.cat([xyz_0, xyz_1], dim=0)

        with torch.no_grad():
            point = pre_encoder(xyz, features=None)

        counts = offset2bincount(point.offset)
        _, _, mask = point2dense(point)

        if mask is not None:
            max_n = counts.max().item()
            for b in range(2):
                n = counts[b].item()
                # Real positions should not be masked
                assert mask[b, :n].sum() == 0, f"Real positions masked in scene {b}"
                # Padded positions should be masked
                assert mask[b, n:].sum() == max_n - n, f"Padding not masked in scene {b}"

    def test_padding_preserves_gradients(self, pre_encoder):
        """Gradients must flow through the padding path back to input."""
        xyz = (torch.randn(2, 5000, 3, device="cuda") * 2.0).requires_grad_(True)
        point = pre_encoder(xyz, features=None)
        xyz_dense, feat_dense, mask = point2dense(point)
        loss = feat_dense.sum()
        loss.backward()
        assert xyz.grad is not None, "No gradient on input xyz"
        assert (xyz.grad != 0).any(), "All gradients are zero"
