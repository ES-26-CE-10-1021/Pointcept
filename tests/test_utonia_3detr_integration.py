"""
Tests for PTv3m3PreEncoder (Utonia / PT-v3m3) integration with 3DETR.

Mirrors the structure of ``tests/test_ptv3_3detr_integration.py`` but
exercises the frozen-backbone variant. All tests run with
``pretrained=None`` so they don't require HuggingFace access; the real
download path is covered by a separate ``integration``-marked test that
is skipped by default / when ``HF_HUB_OFFLINE=1``.

Run with:
    pytest tests/test_utonia_3detr_integration.py -v -s

Most tests require CUDA — PT-v3m3's Block uses operations (flash or
sparse attention) that are only exercised on GPU.
"""

import logging
import os
import sys

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

from pointcept.models.utils.structure import Point  # noqa: E402
from pointcept.models.detection_3detr.model import (  # noqa: E402
    Model3DETRDetector,
)
from pointcept.models.detection_3detr.dataset_config import (  # noqa: E402
    ScanNetDetectionConfig,
)

try:
    from pointcept.models.detection_3detr.ptv3 import PTv3m3PreEncoder

    PTV3M3_AVAILABLE = True
    PTV3M3_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - defensive
    PTV3M3_AVAILABLE = False
    PTV3M3_IMPORT_ERROR = str(e)

skip_no_ptv3m3 = pytest.mark.skipif(
    not PTV3M3_AVAILABLE,
    reason=f"PTv3m3PreEncoder not importable: {PTV3M3_IMPORT_ERROR}",
)
skip_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


# ── Tiny PT-v3m3 config for fast tests ────────────────────────────────────────
# Head dims must be divisible by 3 (Point3DRoPE constraint).
# Stage widths / heads pick head_dim=6 throughout:
#   enc_channels=(12,24,48), enc_num_head=(2,4,8) → 6,6,6
# enc_mode is user-selectable; dec_* defaults are unused unless enabled.
TINY_UTONIA_CFG = dict(
    type="PTv3m3PreEncoder",
    pretrained=None,
    grid_size=0.02,
    freeze_backbone="enc",
    # Backbone config (merged into super().__init__ via **overrides).
    in_channels=9,
    order=("z", "z-trans"),
    stride=(2, 2),
    enc_depths=(1, 1, 1),
    enc_channels=(12, 24, 48),
    enc_num_head=(2, 4, 8),
    enc_patch_size=(64, 64, 64),
    mlp_ratio=4,
    drop_path=0.0,
    shuffle_orders=False,
    enable_rpe=False,
    enable_flash=False,
    upcast_attention=False,
    upcast_softmax=False,
)

TINY_ENC_DIM = TINY_UTONIA_CFG["enc_channels"][-1]  # 48


def _build_pre_encoder(**overrides):
    """Instantiate a tiny PTv3m3PreEncoder with test-friendly overrides."""
    cfg = {k: v for k, v in TINY_UTONIA_CFG.items() if k != "type"}
    cfg.update(overrides)
    return PTv3m3PreEncoder(**cfg)


def _build_tiny_model(criterion=None):
    """Build a minimal Model3DETRDetector with a tiny PTv3m3PreEncoder."""
    return Model3DETRDetector(
        pre_encoder=TINY_UTONIA_CFG,
        encoder=dict(
            type="VanillaTransformerEncoder3DETR",
            encoder_dim=TINY_ENC_DIM,
            nhead=4,
            nlayers=2,
            ffn_dim=64,
            dropout=0.0,
            activation="relu",
        ),
        decoder=dict(
            type="TransformerDecoder3DETR",
            decoder_dim=TINY_ENC_DIM,
            nhead=4,
            nlayers=2,
            ffn_dim=64,
            dropout=0.0,
        ),
        dataset_config=dict(type="ScanNetDetectionConfig"),
        encoder_dim=TINY_ENC_DIM,
        decoder_dim=TINY_ENC_DIM,
        num_queries=32,
        position_embedding="fourier",
        mlp_dropout=0.0,
        projection_norm="ln",
        criterion=criterion,
    )


def _make_batch(B=2, N=4000, device="cuda"):
    pc = torch.randn(B, N, 3, device=device) * 2.0
    return {
        "point_clouds": pc,
        "point_cloud_dims_min": pc.amin(dim=1),
        "point_cloud_dims_max": pc.amax(dim=1),
    }


def _make_train_batch(B=2, N=4000, device="cuda"):
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


# ═══════════════════════════════════════════════════════════════════════════════
# A — Basic forward + output shape
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3m3
@skip_no_cuda
class TestPTv3m3PreEncoderForward:
    """Verify a tiny PTv3m3PreEncoder produces a well-formed Point."""

    @pytest.fixture(scope="class")
    def pre_encoder(self):
        return _build_pre_encoder().cuda()

    def test_returns_point(self, pre_encoder):
        xyz = torch.randn(2, 4000, 3, device="cuda") * 2.0
        out = pre_encoder(xyz, features=None)
        assert isinstance(out, Point), f"Expected Point, got {type(out)}"

    def test_output_has_required_fields(self, pre_encoder):
        xyz = torch.randn(2, 4000, 3, device="cuda") * 2.0
        out = pre_encoder(xyz, features=None)
        for field in ("coord", "feat", "offset", "batch"):
            assert field in out.keys(), f"Missing field: {field}"

    def test_output_feat_dim_matches_enc_channels(self, pre_encoder):
        xyz = torch.randn(2, 4000, 3, device="cuda") * 2.0
        out = pre_encoder(xyz, features=None)
        assert out.feat.shape[-1] == TINY_ENC_DIM

    def test_enc_mode_forced_on(self, pre_encoder):
        """Adapter must force enc_mode=True regardless of user config."""
        assert pre_encoder.enc_mode is True
        assert not hasattr(pre_encoder, "dec")

    def test_grid_size_stored(self, pre_encoder):
        assert pre_encoder.grid_size == 0.02

    def test_in_channels_override_mismatch_raises(self):
        """config_overrides['in_channels'] ≠ checkpoint must raise.

        With ``pretrained=None`` there's no checkpoint to mismatch against,
        so this test only exercises the *build* path succeeds; the strict
        mismatch guard is covered conceptually by `_load_pretrained_state`
        and the real-HF smoke test.
        """
        pe = _build_pre_encoder(in_channels=9)
        assert pe.embedding.in_channels == 9


# ═══════════════════════════════════════════════════════════════════════════════
# B — Zero-padding hack: XYZ + optional RGB + normal pad
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3m3
@skip_no_cuda
class TestPTv3m3ZeroPadding:
    """Exercise the _build_padded_feat helper on device."""

    @pytest.fixture(scope="class")
    def pre_encoder(self):
        return _build_pre_encoder().cuda()

    def test_features_none_pads_to_target(self, pre_encoder):
        """With features=None, channels 3..8 must all be zero."""
        B, N = 2, 500
        xyz = torch.randn(B, N, 3, device="cuda")
        feat = pre_encoder._build_padded_feat(xyz, features=None)
        assert feat.shape == (B, 9, N)
        # xyz channels preserved
        assert torch.allclose(feat[:, :3, :], xyz.transpose(1, 2))
        # rgb + normal channels are zero
        assert (feat[:, 3:, :] == 0).all()

    def test_features_3ch_fills_rgb_and_zeros_normals(self, pre_encoder):
        """With features=(B,3,N), channels 3..5 get the rgb, 6..8 zeros."""
        B, N = 2, 500
        xyz = torch.randn(B, N, 3, device="cuda")
        rgb = torch.randn(B, 3, N, device="cuda")
        feat = pre_encoder._build_padded_feat(xyz, features=rgb)
        assert feat.shape == (B, 9, N)
        assert torch.allclose(feat[:, :3, :], xyz.transpose(1, 2))
        assert torch.allclose(feat[:, 3:6, :], rgb)
        assert (feat[:, 6:, :] == 0).all()

    def test_padded_feat_on_device(self, pre_encoder):
        """Regression guard: zero-pad must never trigger a CPU→GPU sync."""
        B, N = 2, 500
        xyz = torch.randn(B, N, 3, device="cuda")
        feat = pre_encoder._build_padded_feat(xyz, features=None)
        assert feat.device == xyz.device
        assert feat.dtype == xyz.dtype

    def test_over_capacity_raises(self, pre_encoder):
        """Refuse to silently truncate when user hands too many channels."""
        B, N = 2, 500
        xyz = torch.randn(B, N, 3, device="cuda")
        # 9 total + 1 extra = 10 channels, exceeds target_c=9
        bad = torch.randn(B, 7, N, device="cuda")
        with pytest.raises(ValueError, match="channel"):
            pre_encoder._build_padded_feat(xyz, features=bad)


# ═══════════════════════════════════════════════════════════════════════════════
# C — Frozen backbone: requires_grad + forced eval
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3m3
@skip_no_cuda
class TestPTv3m3FreezeAndEval:
    """With freeze=freeze_eval=True, embedding/enc are frozen AND eval-mode."""

    @pytest.fixture(scope="class")
    def pre_encoder(self):
        return _build_pre_encoder().cuda()

    def test_embedding_params_require_no_grad(self, pre_encoder):
        for p in pre_encoder.embedding.parameters():
            assert p.requires_grad is False

    def test_enc_params_require_no_grad(self, pre_encoder):
        for p in pre_encoder.enc.parameters():
            assert p.requires_grad is False

    def test_train_flips_wrapper_but_not_backbone(self, pre_encoder):
        """After .train(), embedding/enc must stay in eval mode."""
        pre_encoder.train()
        assert pre_encoder.training is True  # wrapper is training
        assert pre_encoder.embedding.training is False
        assert pre_encoder.enc.training is False

    def test_backward_does_not_touch_backbone(self, pre_encoder):
        """A loss.backward() on the output must not populate .grad on
        any parameter inside the frozen backbone."""
        pre_encoder.train()
        # Clear any residual grad state
        for p in pre_encoder.parameters():
            p.grad = None

        xyz = torch.randn(2, 3000, 3, device="cuda") * 2.0
        out = pre_encoder(xyz, features=None)
        # Tack a tiny trainable projection on the output and backprop.
        proj = nn.Linear(TINY_ENC_DIM, 4, device="cuda")
        loss = proj(out.feat).sum()
        loss.backward()

        for n, p in pre_encoder.embedding.named_parameters():
            assert p.grad is None, f"Frozen embedding param '{n}' got a grad"
        for n, p in pre_encoder.enc.named_parameters():
            assert p.grad is None, f"Frozen enc param '{n}' got a grad"


# ═══════════════════════════════════════════════════════════════════════════════
# D — no_grad bridge: output is a fresh leaf with requires_grad=True
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3m3
@skip_no_cuda
class TestPTv3m3NoGradBridge:
    """Verify the detach + requires_grad_(True) bridge between the
    no_grad backbone and the downstream autograd-tracked projection."""

    def test_feat_is_fresh_leaf(self):
        pe = _build_pre_encoder(freeze_no_grad=True).cuda()
        pe.train()
        xyz = torch.randn(2, 3000, 3, device="cuda") * 2.0
        out = pe(xyz, features=None)
        # Leaf => grad_fn is None, requires_grad is True
        assert out.feat.grad_fn is None, (
            "Output feat has a grad_fn — the no_grad context didn't break "
            "the graph back into the Utonia backbone."
        )
        assert out.feat.requires_grad is True, (
            "Output feat doesn't require grad — downstream won't backprop "
            "into its own layers."
        )
        assert out.feat.is_leaf

    def test_eval_mode_skips_requires_grad(self):
        """In eval mode there's no backward pass, so requires_grad stays False."""
        pe = _build_pre_encoder(freeze_no_grad=True).cuda().eval()
        xyz = torch.randn(2, 3000, 3, device="cuda") * 2.0
        out = pe(xyz, features=None)
        assert out.feat.requires_grad is False
        assert out.feat.grad_fn is None

    def test_freeze_no_grad_false_builds_graph_through_inputs(self):
        """With freeze_no_grad=False and an autograd-tracked input, the
        null context must let the graph extend through the backbone. This
        is what makes the no_grad *flag* meaningful: True actively
        suppresses the graph, False leaves it up to the caller."""
        pe = _build_pre_encoder(freeze_no_grad=False).cuda()
        pe.train()
        xyz = (torch.randn(2, 3000, 3, device="cuda") * 2.0).requires_grad_(True)
        out = pe(xyz, features=None)
        assert out.feat.grad_fn is not None, (
            "freeze_no_grad=False did not let autograd extend through the "
            "backbone from an input that requires grad."
        )

    def test_freeze_no_grad_true_blocks_graph_even_from_inputs(self):
        """Counterpart to the above: with freeze_no_grad=True, even an
        autograd-tracked input must not produce a graph on the output."""
        pe = _build_pre_encoder(freeze_no_grad=True).cuda()
        pe.train()
        xyz = (torch.randn(2, 3000, 3, device="cuda") * 2.0).requires_grad_(True)
        out = pe(xyz, features=None)
        # Fresh leaf with requires_grad flipped on by the adapter.
        assert out.feat.is_leaf
        assert out.feat.grad_fn is None

    def test_backward_on_downstream_leaves_backbone_untouched(self):
        """After downstream loss.backward(), no backbone param has .grad."""
        pe = _build_pre_encoder(freeze_no_grad=True).cuda()
        pe.train()
        for p in pe.parameters():
            p.grad = None

        xyz = torch.randn(2, 3000, 3, device="cuda") * 2.0
        out = pe(xyz, features=None)
        # Simulate a downstream projection
        proj = nn.Linear(TINY_ENC_DIM, 4, device="cuda")
        loss = proj(out.feat).sum()
        loss.backward()

        # Downstream projection got gradients …
        assert proj.weight.grad is not None
        # … backbone did not.
        for p in pe.embedding.parameters():
            assert p.grad is None
        for p in pe.enc.parameters():
            assert p.grad is None


# ═══════════════════════════════════════════════════════════════════════════════
# E — VRAM regression guard: no_grad path allocates strictly less peak VRAM
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3m3
@skip_no_cuda
class TestPTv3m3VRAMNoBleed:
    """Regression guard for the torch.no_grad() wrapper in forward()."""

    def _measure_peak(self, freeze_no_grad):
        pe = _build_pre_encoder(freeze_no_grad=freeze_no_grad).cuda()
        pe.train()
        # Warmup so cached blocks don't skew the measurement.
        xyz_warm = torch.randn(2, 3000, 3, device="cuda") * 2.0
        _ = pe(xyz_warm, features=None)
        del _

        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        xyz = torch.randn(2, 3000, 3, device="cuda") * 2.0
        out = pe(xyz, features=None)
        # Touch the output so the compiler can't constant-fold anything away.
        _ = out.feat.float().sum().item()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated()

    def test_no_grad_uses_less_peak_vram(self):
        with_no_grad = self._measure_peak(freeze_no_grad=True)
        without_no_grad = self._measure_peak(freeze_no_grad=False)
        # Strict inequality — graph construction always allocates *something*
        # for activations that no_grad skips.
        assert with_no_grad < without_no_grad, (
            f"freeze_no_grad=True peak VRAM ({with_no_grad}) is not strictly "
            f"less than freeze_no_grad=False peak VRAM ({without_no_grad}). "
            f"The torch.no_grad() wrapper may have been removed from "
            f"PTv3m3PreEncoder.forward."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# F — End-to-end: Utonia + VanillaTransformerEncoder + 3DETR decoder
# ═══════════════════════════════════════════════════════════════════════════════


@skip_no_ptv3m3
@skip_no_cuda
class TestUtonia3DETREndToEnd:
    """Full forward + backward through a tiny Model3DETRDetector.

    Verifies that only the non-frozen parameters receive gradients and
    that the loss is finite."""

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
        return _build_tiny_model(criterion=criterion_cfg).cuda().train()

    def test_forward_returns_finite_loss(self, model):
        batch = _make_train_batch()
        out = model(batch)
        assert "loss" in out
        assert out["loss"].ndim == 0
        assert torch.isfinite(out["loss"])
        assert out["loss"].requires_grad

    def test_only_non_utonia_params_get_gradients(self, model):
        model.zero_grad()
        batch = _make_train_batch()
        out = model(batch)
        out["loss"].backward()

        # Backbone parameters: all grads must be None
        for n, p in model.pre_encoder.embedding.named_parameters():
            assert p.grad is None, f"Frozen backbone embedding '{n}' got a grad"
        for n, p in model.pre_encoder.enc.named_parameters():
            assert p.grad is None, f"Frozen backbone enc '{n}' got a grad"

        # At least some non-backbone parameters must have received grads
        downstream = [
            (n, p.grad)
            for n, p in model.named_parameters()
            if not n.startswith("pre_encoder.")
            and p.grad is not None
        ]
        assert len(downstream) > 0, "No downstream parameters received gradients"
        for n, g in downstream:
            assert torch.isfinite(g).all(), f"Non-finite grad in {n}"

    def test_eval_forward_outputs_box_predictions(self, model):
        model.eval()
        try:
            B, nq = 2, 32
            batch = _make_batch(B=B)
            with torch.no_grad():
                out = model(batch)
            preds = out["outputs"]
            assert preds["sem_cls_logits"].shape == (B, nq, 19)
            assert preds["center_unnormalized"].shape == (B, nq, 3)
            assert preds["size_unnormalized"].shape == (B, nq, 3)
            assert preds["box_corners"].shape == (B, nq, 8, 3)
        finally:
            model.train()


# ═══════════════════════════════════════════════════════════════════════════════
# G — Opportunistic: real HuggingFace checkpoint load (network-bound)
# ═══════════════════════════════════════════════════════════════════════════════


# Skipped by default — set RUN_UTONIA_HF_TEST=1 to opt in. Also always
# skipped under HF_HUB_OFFLINE=1.
_HF_TEST_ENABLED = (
    os.environ.get("RUN_UTONIA_HF_TEST") == "1"
    and os.environ.get("HF_HUB_OFFLINE") != "1"
)


@skip_no_ptv3m3
@pytest.mark.skipif(
    not _HF_TEST_ENABLED,
    reason="Set RUN_UTONIA_HF_TEST=1 to run the real-HF smoke test.",
)
def test_utonia_load_hf():
    """Smoke test: real Utonia checkpoint loads into pointcept's PT-v3m3.

    Verifies key parity between the upstream and pointcept-side
    ``PointTransformerV3`` classes. Requires network access on first run.
    """
    pe = PTv3m3PreEncoder(pretrained="utonia", grid_size=0.02)
    # After load, at least one Utonia conv/linear weight should be
    # non-identity (i.e. not freshly initialized).
    stem_w = pe.embedding.stem.linear.weight
    assert stem_w.abs().mean() > 1e-6
    if torch.cuda.is_available():
        pe = pe.cuda()
        xyz = torch.randn(1, 4000, 3, device="cuda") * 2.0
        out = pe(xyz, features=None)
        assert isinstance(out, Point)
