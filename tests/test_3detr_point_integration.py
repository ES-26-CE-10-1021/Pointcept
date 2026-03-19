"""
Tests for Point class integration in the 3DETR detection model.

Verifies that the dense↔Point conversion functions and the model's
isinstance-based dispatch (for Point-returning pre-encoders/encoders)
work correctly, preserving numerical equivalence with the original
dense-tensor path.

Run with:
    pytest tests/test_3detr_point_integration.py -v -s
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── determinism ───────────────────────────────────────────────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ── imports ───────────────────────────────────────────────────────────────────
from pointcept.models.utils.structure import Point
from pointcept.models.utils import offset2bincount
from pointcept.models.detection_3detr.model import (
    Model3DETRDetector,
    dense2point,
    point2dense,
)
from pointcept.models.detection_3detr.components import (
    IdentityEncoder3DETR,
)
from pointcept.models.detection_3detr.dataset_config import ScanNetDetectionConfig

# ═══════════════════════════════════════════════════════════════════════════════
# A — dense2point / point2dense round-trip
# ═══════════════════════════════════════════════════════════════════════════════


class TestDensePointConversions:
    """Verify dense↔Point conversion functions are lossless and correct."""

    def test_dense2point_shapes(self):
        """dense2point must produce correct concat+offset shapes."""
        B, N, C = 3, 100, 8
        xyz = torch.randn(B, N, 3)
        features = torch.randn(B, C, N)

        point = dense2point(xyz, features)

        assert point.coord.shape == (B * N, 3)
        assert point.feat.shape == (B * N, C)
        assert point.offset.shape == (B,)
        assert point.offset.tolist() == [N, 2 * N, 3 * N]

    def test_dense2point_no_features(self):
        """When features=None, feat should be a clone of coord."""
        B, N = 2, 50
        xyz = torch.randn(B, N, 3)

        point = dense2point(xyz, features=None)

        assert point.feat.shape == (B * N, 3)
        torch.testing.assert_close(point.feat, point.coord)
        # Must be a clone, not the same tensor
        assert point.feat.data_ptr() != point.coord.data_ptr()

    def test_dense2point_values(self):
        """dense2point must correctly flatten batch dim for coords and features."""
        B, N, C = 2, 5, 4
        xyz = torch.randn(B, N, 3)
        features = torch.randn(B, C, N)

        point = dense2point(xyz, features)

        # First scene coords
        torch.testing.assert_close(point.coord[:N], xyz[0])
        # Second scene coords
        torch.testing.assert_close(point.coord[N:], xyz[1])
        # First scene features: (B, C, N) → permute → (B, N, C) → flatten
        torch.testing.assert_close(point.feat[:N], features[0].T)

    def test_point2dense_shapes(self):
        """point2dense must produce correct dense tensor shapes."""
        B, N, C = 3, 100, 16
        coord = torch.randn(B * N, 3)
        feat = torch.randn(B * N, C)
        offset = torch.arange(1, B + 1) * N

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_out, feat_out = point2dense(point)

        assert xyz_out.shape == (B, N, 3)
        assert feat_out.shape == (B, C, N)

    def test_roundtrip_dense_to_point_to_dense(self):
        """dense → Point → dense must be numerically identical."""
        B, N, C = 4, 200, 32
        xyz = torch.randn(B, N, 3)
        features = torch.randn(B, C, N)

        point = dense2point(xyz, features)
        xyz_rt, feat_rt = point2dense(point)

        torch.testing.assert_close(xyz_rt, xyz)
        torch.testing.assert_close(feat_rt, features)

    def test_roundtrip_point_to_dense_to_point(self):
        """Point → dense → Point must be numerically identical."""
        B, N, C = 2, 150, 8
        coord = torch.randn(B * N, 3)
        feat = torch.randn(B * N, C)
        offset = torch.arange(1, B + 1) * N

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_dense, feat_dense = point2dense(point)
        point_rt = dense2point(xyz_dense, feat_dense)

        torch.testing.assert_close(point_rt.coord, coord)
        torch.testing.assert_close(point_rt.feat, feat)
        assert point_rt.offset.tolist() == offset.tolist()

    def test_point2dense_variable_lengths_raises(self):
        """point2dense must reject variable-length scenes (no padding mask)."""
        # Scene 0: 10 points, Scene 1: 20 points
        n0, n1, C = 10, 20, 4
        coord = torch.randn(n0 + n1, 3)
        feat = torch.randn(n0 + n1, C)
        offset = torch.tensor([n0, n0 + n1])

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        with pytest.raises(ValueError, match="variable-length scenes"):
            point2dense(point)

    def test_point2dense_equal_lengths(self):
        """point2dense must succeed when all scenes have equal length."""
        B, N, C = 3, 50, 8
        coord = torch.randn(B * N, 3)
        feat = torch.randn(B * N, C)
        offset = torch.arange(1, B + 1) * N

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_out, feat_out = point2dense(point)

        assert xyz_out.shape == (B, N, 3)
        assert feat_out.shape == (B, C, N)
        # Verify first scene data is correct
        torch.testing.assert_close(xyz_out[0], coord[:N])

    def test_point2dense_preserves_gradients(self):
        """point2dense must preserve autograd graph for backprop."""
        B, N, C = 2, 50, 8
        coord = torch.randn(B * N, 3, requires_grad=True)
        feat = torch.randn(B * N, C, requires_grad=True)
        offset = torch.arange(1, B + 1) * N

        point = Point(dict(coord=coord, feat=feat, offset=offset))
        xyz_out, feat_out = point2dense(point)

        loss = xyz_out.sum() + feat_out.sum()
        loss.backward()

        assert coord.grad is not None, "No gradient on coord"
        assert feat.grad is not None, "No gradient on feat"
        assert (coord.grad != 0).any(), "Coord gradients are all zero"
        assert (feat.grad != 0).any(), "Feat gradients are all zero"

    def test_batch_field_auto_generated(self):
        """Point init should auto-generate batch from offset."""
        B, N = 3, 10
        offset = torch.arange(1, B + 1) * N
        point = Point(
            dict(coord=torch.randn(B * N, 3), feat=torch.randn(B * N, 3), offset=offset)
        )

        assert "batch" in point.keys()
        assert point.batch.shape == (B * N,)
        # First N points should be batch 0, next N batch 1, etc.
        assert (point.batch[:N] == 0).all()
        assert (point.batch[N : 2 * N] == 1).all()
        assert (point.batch[2 * N :] == 2).all()

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dense2point_device_preservation(self, device):
        """dense2point must keep tensors on the same device as input."""
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        xyz = torch.randn(2, 50, 3, device=device)
        point = dense2point(xyz)
        assert point.coord.device.type == device
        assert point.feat.device.type == device
        assert point.offset.device.type == device


# ═══════════════════════════════════════════════════════════════════════════════
# B — Point-returning pre-encoder dispatch in run_encoder
# ═══════════════════════════════════════════════════════════════════════════════


class _MockPointPreEncoder(nn.Module):
    """Pre-encoder that returns a Point object (simulates PTv3-based component).

    Applies a simple linear projection so weights are non-trivial, then
    returns the result as a Point with concat+offset format.
    """

    def __init__(self, in_dim, out_dim, npoint):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.npoint = npoint

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        # Sub-sample via stride (deterministic, no FPS needed)
        stride = max(1, N // self.npoint)
        inds = torch.arange(0, N, stride, device=xyz.device)[: self.npoint]
        sub_xyz = xyz[:, inds]  # (B, npoint, 3)
        if features is not None:
            sub_feat = features[:, :, inds].permute(0, 2, 1)  # (B, npoint, C)
            inp = torch.cat([sub_xyz, sub_feat], dim=-1)
        else:
            inp = sub_xyz
        out_feat = self.proj(inp)  # (B, npoint, out_dim)
        return dense2point(sub_xyz, out_feat.permute(0, 2, 1))


class _MockTuplePreEncoder(nn.Module):
    """Pre-encoder that returns the traditional (xyz, features, inds) tuple.

    Uses the same linear projection as _MockPointPreEncoder for comparison.
    """

    def __init__(self, in_dim, out_dim, npoint):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.npoint = npoint

    def forward(self, xyz, features=None):
        B, N, _ = xyz.shape
        stride = max(1, N // self.npoint)
        inds = torch.arange(0, N, stride, device=xyz.device)[: self.npoint]
        sub_xyz = xyz[:, inds]
        if features is not None:
            sub_feat = features[:, :, inds].permute(0, 2, 1)
            inp = torch.cat([sub_xyz, sub_feat], dim=-1)
        else:
            inp = sub_xyz
        out_feat = self.proj(inp)  # (B, npoint, out_dim)
        return sub_xyz, out_feat.permute(0, 2, 1), inds.unsqueeze(0).expand(B, -1)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FPS in decoder"
)
class TestPointPreEncoderDispatch:
    """Verify run_encoder handles Point-returning pre-encoders correctly."""

    def _build_model(self, pre_encoder, encoder_dim=64):
        """Build a Model3DETRDetector with a custom pre-encoder."""
        return Model3DETRDetector(
            pre_encoder=None,  # placeholder, replaced below
            encoder=dict(
                type="IdentityEncoder3DETR",
            ),
            decoder=dict(
                type="TransformerDecoder3DETR",
                decoder_dim=encoder_dim,
                nhead=4,
                nlayers=2,
                ffn_dim=64,
                dropout=0.0,
            ),
            dataset_config=dict(type="ScanNetDetectionConfig"),
            encoder_dim=encoder_dim,
            decoder_dim=encoder_dim,
            num_queries=32,
            position_embedding="fourier",
            mlp_dropout=0.0,
            criterion=None,
        )

    def _make_batch(self, B=2, N=200, device="cuda"):
        pc = torch.randn(B, N, 3, device=device)
        return {
            "point_clouds": pc,
            "point_cloud_dims_min": pc.amin(dim=1),
            "point_cloud_dims_max": pc.amax(dim=1),
        }

    def test_point_pre_encoder_produces_valid_encoder_output(self):
        """A Point-returning pre-encoder should produce valid enc_xyz/features."""
        enc_dim, npoint = 64, 50
        model = self._build_model(pre_encoder=None, encoder_dim=enc_dim)
        # Replace pre_encoder with our mock
        model.pre_encoder = _MockPointPreEncoder(
            in_dim=3, out_dim=enc_dim, npoint=npoint
        )
        model = model.cuda().eval()

        batch = self._make_batch()
        with torch.no_grad():
            enc_xyz, enc_features, enc_inds = model.run_encoder(batch["point_clouds"])

        B = 2
        assert enc_xyz.shape == (B, npoint, 3)
        # IdentityEncoder doesn't change feature count
        assert enc_features.shape == (npoint, B, enc_dim)
        # Point pre-encoder → inds should be None
        assert enc_inds is None

    def test_point_vs_tuple_pre_encoder_equivalence(self):
        """Point and tuple pre-encoders with shared weights must give identical results."""
        enc_dim, npoint = 64, 50
        in_dim = 3  # XYZ only

        # Build two models
        model_point = self._build_model(pre_encoder=None, encoder_dim=enc_dim)
        model_tuple = self._build_model(pre_encoder=None, encoder_dim=enc_dim)

        # Create pre-encoders with shared weights
        point_pe = _MockPointPreEncoder(in_dim=in_dim, out_dim=enc_dim, npoint=npoint)
        tuple_pe = _MockTuplePreEncoder(in_dim=in_dim, out_dim=enc_dim, npoint=npoint)
        tuple_pe.proj.load_state_dict(point_pe.proj.state_dict())

        model_point.pre_encoder = point_pe
        model_tuple.pre_encoder = tuple_pe

        # Share all other weights
        model_tuple.load_state_dict(model_point.state_dict(), strict=False)

        model_point = model_point.cuda().eval()
        model_tuple = model_tuple.cuda().eval()

        batch = self._make_batch()
        with torch.no_grad():
            p_xyz, p_feat, p_inds = model_point.run_encoder(batch["point_clouds"])
            t_xyz, t_feat, t_inds = model_tuple.run_encoder(batch["point_clouds"])

        torch.testing.assert_close(p_xyz, t_xyz, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(p_feat, t_feat, atol=1e-5, rtol=1e-4)

    def test_full_forward_with_point_pre_encoder(self):
        """Full eval forward should succeed with a Point-returning pre-encoder."""
        enc_dim, npoint = 64, 50
        model = self._build_model(pre_encoder=None, encoder_dim=enc_dim)
        model.pre_encoder = _MockPointPreEncoder(
            in_dim=3, out_dim=enc_dim, npoint=npoint
        )
        model = model.cuda().eval()

        batch = self._make_batch()
        with torch.no_grad():
            output = model(batch)

        assert "outputs" in output
        assert "aux_outputs" in output
        out = output["outputs"]
        assert out["sem_cls_logits"].shape[0] == 2  # batch size
        assert out["sem_cls_logits"].shape[1] == 32  # num_queries
        assert out["center_unnormalized"].shape == (2, 32, 3)
        assert out["box_corners"].shape == (2, 32, 8, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Point-returning encoder dispatch in run_encoder
# ═══════════════════════════════════════════════════════════════════════════════


class _MockPointEncoder(nn.Module):
    """Encoder that returns a Point object instead of (xyz, features, inds)."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, xyz):
        # features: (N, B, C) → apply norm → convert to Point
        out = self.norm(features)  # (N, B, C)
        N, B, C = out.shape
        # Convert to (B, C, N) for dense2point
        feat_bcn = out.permute(1, 2, 0)  # (B, C, N)
        return dense2point(xyz, feat_bcn)


class _MockTupleEncoder(nn.Module):
    """Encoder that returns (xyz, features, inds) tuple with same computation."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, features, xyz):
        out = self.norm(features)
        return xyz, out, None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FPS in decoder"
)
class TestPointEncoderDispatch:
    """Verify run_encoder handles Point-returning encoders correctly."""

    def _build_model(self, encoder_dim=64):
        return Model3DETRDetector(
            pre_encoder=None,
            encoder=dict(type="IdentityEncoder3DETR"),  # placeholder
            decoder=dict(
                type="TransformerDecoder3DETR",
                decoder_dim=encoder_dim,
                nhead=4,
                nlayers=2,
                ffn_dim=64,
                dropout=0.0,
            ),
            dataset_config=dict(type="ScanNetDetectionConfig"),
            encoder_dim=encoder_dim,
            decoder_dim=encoder_dim,
            num_queries=32,
            position_embedding="fourier",
            mlp_dropout=0.0,
            criterion=None,
        )

    def _make_batch(self, B=2, N=200, device="cuda"):
        pc = torch.randn(B, N, 3, device=device)
        return {
            "point_clouds": pc,
            "point_cloud_dims_min": pc.amin(dim=1),
            "point_cloud_dims_max": pc.amax(dim=1),
        }

    def test_point_vs_tuple_encoder_equivalence(self):
        """Point and tuple encoders with shared weights must give identical output."""
        enc_dim = 64
        model_point = self._build_model(encoder_dim=enc_dim)
        model_tuple = self._build_model(encoder_dim=enc_dim)

        point_enc = _MockPointEncoder(dim=enc_dim)
        tuple_enc = _MockTupleEncoder(dim=enc_dim)
        tuple_enc.load_state_dict(point_enc.state_dict())

        model_point.encoder = point_enc
        model_tuple.encoder = tuple_enc

        # Share all other weights (input_projection, decoder, etc.)
        model_tuple.load_state_dict(model_point.state_dict(), strict=False)

        model_point = model_point.cuda().eval()
        model_tuple = model_tuple.cuda().eval()

        batch = self._make_batch()
        with torch.no_grad():
            p_xyz, p_feat, _ = model_point.run_encoder(batch["point_clouds"])
            t_xyz, t_feat, _ = model_tuple.run_encoder(batch["point_clouds"])

        torch.testing.assert_close(p_xyz, t_xyz, atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(p_feat, t_feat, atol=1e-5, rtol=1e-4)

    def test_full_forward_with_point_encoder(self):
        """Full eval forward should succeed with a Point-returning encoder."""
        enc_dim = 64
        model = self._build_model(encoder_dim=enc_dim)
        model.encoder = _MockPointEncoder(dim=enc_dim)
        model = model.cuda().eval()

        batch = self._make_batch()
        with torch.no_grad():
            output = model(batch)

        assert "outputs" in output
        assert output["outputs"]["sem_cls_logits"].shape[:2] == (2, 32)


# ═══════════════════════════════════════════════════════════════════════════════
# D — Combined Point pre-encoder + Point encoder
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FPS in decoder"
)
class TestPointPreEncoderPlusPointEncoder:
    """End-to-end test: both pre-encoder and encoder return Point objects."""

    def test_full_forward(self):
        """Full eval forward with Point pre-encoder + Point encoder."""
        enc_dim, npoint = 64, 50

        model = Model3DETRDetector(
            pre_encoder=None,
            encoder=dict(type="IdentityEncoder3DETR"),
            decoder=dict(
                type="TransformerDecoder3DETR",
                decoder_dim=enc_dim,
                nhead=4,
                nlayers=2,
                ffn_dim=64,
                dropout=0.0,
            ),
            dataset_config=dict(type="ScanNetDetectionConfig"),
            encoder_dim=enc_dim,
            decoder_dim=enc_dim,
            num_queries=32,
            position_embedding="fourier",
            mlp_dropout=0.0,
            criterion=None,
        )
        model.pre_encoder = _MockPointPreEncoder(
            in_dim=3, out_dim=enc_dim, npoint=npoint
        )
        model.encoder = _MockPointEncoder(dim=enc_dim)
        model = model.cuda().eval()

        B, N = 2, 200
        pc = torch.randn(B, N, 3, device="cuda")
        batch = {
            "point_clouds": pc,
            "point_cloud_dims_min": pc.amin(dim=1),
            "point_cloud_dims_max": pc.amax(dim=1),
        }

        with torch.no_grad():
            output = model(batch)

        assert "outputs" in output
        out = output["outputs"]
        assert out["sem_cls_logits"].shape == (B, 32, 19)  # 18 classes + 1
        assert out["center_unnormalized"].shape == (B, 32, 3)
        assert out["box_corners"].shape == (B, 32, 8, 3)
        # enc_inds should be None (both pre-enc and enc return Point)
        enc_xyz, enc_feat, enc_inds = model.run_encoder(batch["point_clouds"])
        assert enc_inds is None


# ═══════════════════════════════════════════════════════════════════════════════
# E — IdentityEncoder3DETR
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdentityEncoder:
    """Verify IdentityEncoder3DETR is a true passthrough."""

    def test_passthrough(self):
        enc = IdentityEncoder3DETR()
        N, B, C = 100, 2, 64
        features = torch.randn(N, B, C)
        xyz = torch.randn(B, N, 3)

        out_xyz, out_feat, out_inds = enc(features, xyz)

        assert out_xyz is xyz
        assert out_feat is features
        assert out_inds is None

    def test_no_parameters(self):
        """IdentityEncoder should have zero learnable parameters."""
        enc = IdentityEncoder3DETR()
        assert sum(p.numel() for p in enc.parameters()) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# F — offset2bincount helper
# ═══════════════════════════════════════════════════════════════════════════════


class TestOffset2Bincount:
    """Verify the offset→bincount conversion used by point2dense."""

    def test_equal_lengths(self):
        offset = torch.tensor([100, 200, 300])
        counts = offset2bincount(offset)
        assert counts.tolist() == [100, 100, 100]

    def test_variable_lengths(self):
        offset = torch.tensor([10, 30, 35])
        counts = offset2bincount(offset)
        assert counts.tolist() == [10, 20, 5]

    def test_single_scene(self):
        offset = torch.tensor([42])
        counts = offset2bincount(offset)
        assert counts.tolist() == [42]


# ═══════════════════════════════════════════════════════════════════════════════
# G — Training forward with Point pre-encoder (loss computation)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for FPS")
class TestPointPreEncoderTraining:
    """Verify training forward (with criterion) works with Point pre-encoder."""

    def test_training_forward_produces_loss(self):
        """Model with Point pre-encoder + criterion should return a scalar loss."""
        enc_dim, npoint = 64, 50
        ds_cfg = ScanNetDetectionConfig()

        model = Model3DETRDetector(
            pre_encoder=None,
            encoder=dict(type="IdentityEncoder3DETR"),
            decoder=dict(
                type="TransformerDecoder3DETR",
                decoder_dim=enc_dim,
                nhead=4,
                nlayers=2,
                ffn_dim=64,
                dropout=0.0,
            ),
            dataset_config=dict(type="ScanNetDetectionConfig"),
            encoder_dim=enc_dim,
            decoder_dim=enc_dim,
            num_queries=32,
            position_embedding="fourier",
            mlp_dropout=0.0,
            criterion=dict(
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
            ),
        )
        model.pre_encoder = _MockPointPreEncoder(
            in_dim=3, out_dim=enc_dim, npoint=npoint
        )
        model = model.cuda().train()

        B, N, max_obj, ngt = 2, 200, 64, 4
        pc = torch.randn(B, N, 3, device="cuda")

        # Synthetic GT
        gt_center = torch.rand(B, max_obj, 3, device="cuda")
        gt_size = torch.rand(B, max_obj, 3, device="cuda").clamp(0.05, 1.0)
        gt_angle = torch.zeros(B, max_obj, device="cuda")
        gt_corners = ds_cfg.box_parametrization_to_corners(gt_center, gt_size, gt_angle)
        gt_present = torch.zeros(B, max_obj, device="cuda")
        gt_present[:, :ngt] = 1.0

        input_dict = {
            "point_clouds": pc,
            "point_cloud_dims_min": pc.amin(dim=1),
            "point_cloud_dims_max": pc.amax(dim=1),
            "gt_box_present": gt_present,
            "gt_box_corners": gt_corners,
            "gt_box_centers_normalized": torch.rand(B, max_obj, 3, device="cuda"),
            "gt_box_sem_cls_label": torch.randint(0, 18, (B, max_obj), device="cuda"),
            "gt_box_angles": gt_angle,
            "gt_angle_class_label": torch.zeros(
                B, max_obj, dtype=torch.long, device="cuda"
            ),
            "gt_angle_residual_label": torch.zeros(B, max_obj, device="cuda"),
            "gt_box_sizes_normalized": torch.rand(B, max_obj, 3, device="cuda"),
        }

        output = model(input_dict)

        assert "loss" in output
        assert output["loss"].ndim == 0  # scalar
        assert output["loss"].requires_grad
        assert torch.isfinite(output["loss"])

        # Verify backward works
        output["loss"].backward()
        grad_norms = [
            p.grad.norm().item() for p in model.parameters() if p.grad is not None
        ]
        assert len(grad_norms) > 0, "No gradients computed"
        assert all(np.isfinite(g) for g in grad_norms), "Non-finite gradients"

        # Ensure gradients propagate through Point conversion to pre-encoder
        pre_enc_params = [
            p for p in model.pre_encoder.parameters() if p.requires_grad
        ]
        assert len(pre_enc_params) > 0, (
            "Pre-encoder has no trainable parameters to receive gradients"
        )
        for p in pre_enc_params:
            assert p.grad is not None, "Missing gradient on pre-encoder parameter"
            assert torch.isfinite(p.grad).all(), (
                "Non-finite gradient in pre-encoder parameter"
            )
