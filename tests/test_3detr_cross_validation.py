"""
Cross-validation tests: Pointcept 3DETR vs native third_party/3detr.

Verifies numerical equivalence between the two implementations across:
  A) Dataset output alignment
  B) Model forward-pass equivalence (via weight transfer)
  C) Criterion/loss equivalence
  D) Config audit (informational, no assertions)

Run with:
    pytest tests/test_3detr_cross_validation.py -v -s
"""

import copy
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

# ── determinism ───────────────────────────────────────────────────────────────
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# ── data paths ────────────────────────────────────────────────────────────────
DATA_ROOT = "/home/andreas/3D-Perception/votenet/scannet/scannet_train_detection_data"
META_ROOT = "/home/andreas/3D-Perception/votenet/scannet/meta_data"
DATA_AVAILABLE = os.path.isdir(DATA_ROOT) and os.path.isdir(META_ROOT)

# ── imports ───────────────────────────────────────────────────────────────────
# Native 3DETR
from datasets.scannet import ScannetDatasetConfig, ScannetDetectionDataset
from models.model_3detr import build_3detr
from criterion import SetCriterion as NativeSetCriterion, Matcher as NativeMatcher

# Pointcept 3DETR
from pointcept.datasets.scannet_detection import ScanNetDetectionDataset
from pointcept.models.detection_3detr.model import Model3DETRDetector
from pointcept.models.detection_3detr.criterion import SetCriterion3DETR
from pointcept.models.detection_3detr.dataset_config import ScanNetDetectionConfig


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def native_to_pointcept(name: str) -> str:
    """Map a native Model3DETR state_dict key to the Pointcept equivalent.

    Pointcept wraps each major component with one extra container level:
      pre_encoder.*  -> pre_encoder.sa_module.*
      encoder.*      -> encoder.encoder.*
      decoder.*      -> decoder.decoder.*
    All other keys (encoder_to_decoder_projection, query_projection,
    mlp_heads, pos_embedding) are identical.
    """
    if name.startswith("pre_encoder."):
        return name.replace("pre_encoder.", "pre_encoder.sa_module.", 1)
    if name.startswith("encoder."):
        return name.replace("encoder.", "encoder.encoder.", 1)
    if name.startswith("decoder."):
        return name.replace("decoder.", "decoder.decoder.", 1)
    return name


class _NativeArgs:
    """Minimal argparse.Namespace replacement for build_3detr()."""
    enc_type = "vanilla"
    enc_dim = 256
    enc_nhead = 4
    enc_ffn_dim = 128
    enc_dropout = 0.1
    enc_activation = "relu"
    enc_nlayers = 3
    dec_dim = 256
    dec_nhead = 4
    dec_ffn_dim = 256
    dec_dropout = 0.1
    dec_nlayers = 8
    mlp_dropout = 0.3
    preenc_npoints = 2048
    pos_embed = "fourier"
    nqueries = 256
    use_color = False


# Pointcept model config matching _NativeArgs exactly
POINTCEPT_MODEL_CFG = dict(
    pre_encoder=dict(
        type="PointnetSAPreEncoder",
        npoint=2048,
        radius=0.2,
        nsample=64,
        mlp_dims=[0, 64, 128, 256],
        normalize_xyz=True,
    ),
    encoder=dict(
        type="VanillaTransformerEncoder3DETR",
        encoder_dim=256,
        nhead=4,
        nlayers=3,
        ffn_dim=128,
        dropout=0.1,
        activation="relu",
    ),
    decoder=dict(
        type="TransformerDecoder3DETR",
        decoder_dim=256,
        nhead=4,
        nlayers=8,
        ffn_dim=256,
        dropout=0.1,
    ),
    dataset_config=dict(type="ScanNetDetectionConfig"),
    encoder_dim=256,
    decoder_dim=256,
    num_queries=256,
    position_embedding="fourier",
    mlp_dropout=0.3,
)


def transfer_weights(src: nn.Module, dst: nn.Module) -> None:
    """Copy all native Model3DETR state_dict entries into Pointcept model.

    Uses state_dict() which includes both parameters AND buffers
    (important for pos_embedding.gauss_B).

    Criterion keys in dst are skipped (the native model has no criterion).
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()

    new_dst = {}
    for native_name, tensor in src_sd.items():
        pc_name = native_to_pointcept(native_name)
        assert pc_name in dst_sd, (
            f"Unmapped native key: '{native_name}' -> '{pc_name}'"
        )
        assert dst_sd[pc_name].shape == tensor.shape, (
            f"Shape mismatch for '{native_name}': "
            f"native {tensor.shape} vs Pointcept {dst_sd[pc_name].shape}"
        )
        new_dst[pc_name] = tensor

    # Verify all non-criterion Pointcept keys are covered
    for k in dst_sd:
        if k not in new_dst and not k.startswith("criterion."):
            raise KeyError(
                f"Pointcept key '{k}' not covered by weight transfer"
            )

    dst.load_state_dict(new_dst, strict=False)


# ═══════════════════════════════════════════════════════════════════════════════
# Test A — Dataset alignment
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not DATA_AVAILABLE, reason="ScanNet detection data not found")
class TestDatasetAlignment:
    """Verify both datasets return identical data for the same scene + seed."""

    # Keys that should be exactly equal
    EXACT_KEYS = [
        "point_clouds",
        "gt_box_corners",
        "gt_box_centers",
        "gt_angle_class_label",
        "gt_angle_residual_label",
        "gt_box_sem_cls_label",
        "gt_box_present",
        "gt_box_sizes",
        "gt_box_angles",
        "point_cloud_dims_min",
        "point_cloud_dims_max",
    ]

    # Keys with known tiny differences due to +1e-6 denominator in Pointcept
    APPROX_KEYS = [
        "gt_box_sizes_normalized",
        "gt_box_centers_normalized",
    ]

    @pytest.fixture(scope="class")
    def datasets(self):
        native_ds = ScannetDetectionDataset(
            dataset_config=ScannetDatasetConfig(),
            split_set="val",
            root_dir=DATA_ROOT,
            meta_data_dir=META_ROOT,
            num_points=40000,
            use_color=False,
            use_height=False,
            augment=False,
        )
        pc_ds = ScanNetDetectionDataset(
            root_dir=DATA_ROOT,
            meta_data_dir=META_ROOT,
            split="val",
            num_points=40000,
            use_color=False,
            use_height=False,
            augment=False,
        )
        return native_ds, pc_ds

    def test_scan_list_matches(self, datasets):
        """Both datasets should have the same scene list in the same order."""
        native_ds, pc_ds = datasets
        assert native_ds.scan_names == pc_ds.scan_names

    @pytest.mark.parametrize("idx", [0, 5, 42])
    def test_scene_data_matches(self, datasets, idx):
        native_ds, pc_ds = datasets

        # Same numpy seed -> same np.random.choice in random_sampling
        np.random.seed(999)
        native_item = native_ds[idx]
        np.random.seed(999)
        pc_item = pc_ds[idx]

        for key in self.EXACT_KEYS:
            np.testing.assert_array_equal(
                native_item[key],
                pc_item[key],
                err_msg=f"Exact mismatch in '{key}' for scene idx={idx}",
            )

        for key in self.APPROX_KEYS:
            np.testing.assert_allclose(
                native_item[key],
                pc_item[key],
                atol=1e-5,
                err_msg=(
                    f"Approx mismatch in '{key}' for scene idx={idx} "
                    f"(expected tiny diff from +1e-6 denominator in Pointcept)"
                ),
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test B — Model weight transfer and forward equivalence
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FPS sampling"
)
class TestModelEquivalence:
    """Weight-transferred models must produce identical outputs."""

    @pytest.fixture(scope="class", autouse=True)
    def _enable_deterministic(self):
        """Enable deterministic algorithms for FPS reproducibility."""
        torch.use_deterministic_algorithms(True)
        yield
        torch.use_deterministic_algorithms(False)

    @pytest.fixture(scope="class")
    def model_pair(self):
        # Build native model
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        native_model, _ = build_3detr(_NativeArgs(), ScannetDatasetConfig())
        native_model = native_model.cuda().eval()

        # Build Pointcept model and transfer weights
        pc_model = Model3DETRDetector(**POINTCEPT_MODEL_CFG)
        transfer_weights(native_model, pc_model)
        pc_model = pc_model.cuda().eval()

        return native_model, pc_model

    def _make_batch(self, device="cuda"):
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        B, N = 2, 40000
        pc = torch.randn(B, N, 3, device=device)
        return {
            "point_clouds": pc,
            "point_cloud_dims_min": pc.amin(dim=1),
            "point_cloud_dims_max": pc.amax(dim=1),
        }

    def test_weight_mapping_covers_all_keys(self, model_pair):
        """Verify every native key maps to a valid Pointcept key."""
        native_model, pc_model = model_pair
        native_keys = set(native_model.state_dict().keys())
        pc_keys = set(pc_model.state_dict().keys())

        mapped = {native_to_pointcept(k) for k in native_keys}
        # All mapped keys should exist in Pointcept (excluding criterion.*)
        pc_keys_no_crit = {k for k in pc_keys if not k.startswith("criterion.")}
        assert mapped == pc_keys_no_crit, (
            f"Missing from Pointcept: {mapped - pc_keys_no_crit}\n"
            f"Extra in Pointcept: {pc_keys_no_crit - mapped}"
        )

    def test_encoder_output(self, model_pair):
        """run_encoder should produce identical (xyz, features, inds)."""
        native_model, pc_model = model_pair
        batch = self._make_batch()

        with torch.no_grad():
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            n_xyz, n_feat, n_inds = native_model.run_encoder(
                batch["point_clouds"]
            )

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            p_xyz, p_feat, p_inds = pc_model.run_encoder(
                batch["point_clouds"]
            )

        torch.testing.assert_close(
            n_xyz, p_xyz, atol=1e-5, rtol=1e-4, msg="enc_xyz mismatch"
        )
        torch.testing.assert_close(
            n_feat, p_feat, atol=1e-5, rtol=1e-4, msg="enc_features mismatch"
        )

    def test_full_forward_eval(self, model_pair):
        """Full eval forward pass should produce identical box predictions."""
        native_model, pc_model = model_pair
        batch = self._make_batch()

        COMPARE_KEYS = [
            "sem_cls_logits",
            "center_normalized",
            "center_unnormalized",
            "size_normalized",
            "size_unnormalized",
            "angle_logits",
            "angle_residual",
            "angle_residual_normalized",
            "angle_continuous",
            "objectness_prob",
            "sem_cls_prob",
            "box_corners",
        ]

        with torch.no_grad():
            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            native_out = native_model(batch)

            torch.manual_seed(42)
            torch.cuda.manual_seed(42)
            pc_out = pc_model(batch)

        # Compare final layer outputs
        for key in COMPARE_KEYS:
            torch.testing.assert_close(
                native_out["outputs"][key],
                pc_out["outputs"][key],
                atol=1e-5,
                rtol=1e-4,
                msg=f"outputs['{key}'] mismatch",
            )

        # Compare all intermediate (aux) layers
        assert len(native_out["aux_outputs"]) == len(pc_out["aux_outputs"]), (
            f"aux_outputs length: {len(native_out['aux_outputs'])} vs "
            f"{len(pc_out['aux_outputs'])}"
        )
        for layer_idx, (n_aux, p_aux) in enumerate(
            zip(native_out["aux_outputs"], pc_out["aux_outputs"])
        ):
            for key in COMPARE_KEYS:
                torch.testing.assert_close(
                    n_aux[key],
                    p_aux[key],
                    atol=1e-5,
                    rtol=1e-4,
                    msg=f"aux_outputs[{layer_idx}]['{key}'] mismatch",
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Test C — Criterion equivalence
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for generalized_box3d_iou"
)
class TestCriterionEquivalence:
    """Both criteria must produce identical losses from identical inputs."""

    LOSS_WEIGHTS = dict(
        loss_giou_weight=1.0,
        loss_sem_cls_weight=1.0,
        loss_no_object_weight=0.25,
        loss_angle_cls_weight=0.1,
        loss_angle_reg_weight=0.5,
        loss_center_weight=5.0,
        loss_size_weight=1.0,
    )
    MATCHER_CFG = dict(
        cost_class=1.0,
        cost_objectness=0.1,
        cost_giou=1.0,
        cost_center=5.0,
    )

    @pytest.fixture(scope="class")
    def criterion_pair(self):
        # Native criterion (note: __init__ mutates loss_weight_dict via del)
        native_crit = NativeSetCriterion(
            NativeMatcher(**self.MATCHER_CFG),
            ScannetDatasetConfig(),
            copy.deepcopy(self.LOSS_WEIGHTS),
        ).cuda().eval()

        # Pointcept criterion
        pc_crit = SetCriterion3DETR(
            matcher_cfg=copy.deepcopy(self.MATCHER_CFG),
            loss_weight_dict=copy.deepcopy(self.LOSS_WEIGHTS),
            num_semcls=18,
            num_angle_bin=1,
        ).cuda().eval()

        return native_crit, pc_crit

    def _make_synthetic_data(self, device="cuda"):
        """Create synthetic predictions + targets for criterion testing."""
        torch.manual_seed(13)
        B, nq, nc, max_obj = 2, 256, 18, 64
        ngt = 8  # GT objects per scene

        ds_cfg = ScanNetDetectionConfig()

        # Predictions
        center_u = torch.rand(B, nq, 3, device=device)
        size_u = torch.rand(B, nq, 3, device=device).clamp(0.05, 1.0)
        angle = torch.zeros(B, nq, device=device)
        box_corners = ds_cfg.box_parametrization_to_corners(
            center_u, size_u, angle
        )

        outputs_final = {
            "sem_cls_logits": torch.randn(B, nq, nc + 1, device=device),
            "center_normalized": torch.rand(B, nq, 3, device=device),
            "center_unnormalized": center_u,
            "size_normalized": torch.rand(B, nq, 3, device=device).clamp(0.01),
            "size_unnormalized": size_u,
            "angle_logits": torch.randn(B, nq, 1, device=device),
            "angle_residual": torch.zeros(B, nq, 1, device=device),
            "angle_residual_normalized": torch.zeros(B, nq, 1, device=device),
            "angle_continuous": angle,
            "objectness_prob": torch.rand(B, nq, device=device),
            "sem_cls_prob": torch.softmax(
                torch.randn(B, nq, nc, device=device), dim=-1
            ),
            "box_corners": box_corners,
        }

        # Targets
        gt_center = torch.rand(B, max_obj, 3, device=device)
        gt_size = torch.rand(B, max_obj, 3, device=device).clamp(0.05, 1.0)
        gt_angle = torch.zeros(B, max_obj, device=device)
        gt_corners = ds_cfg.box_parametrization_to_corners(
            gt_center, gt_size, gt_angle
        )

        gt_present = torch.zeros(B, max_obj, device=device)
        gt_present[:, :ngt] = 1.0

        targets = {
            "gt_box_present": gt_present,
            "gt_box_corners": gt_corners,
            "gt_box_centers_normalized": torch.rand(
                B, max_obj, 3, device=device
            ),
            "gt_box_sem_cls_label": torch.randint(
                0, nc, (B, max_obj), device=device
            ),
            "gt_box_angles": gt_angle,
            "gt_angle_class_label": torch.zeros(
                B, max_obj, dtype=torch.long, device=device
            ),
            "gt_angle_residual_label": torch.zeros(
                B, max_obj, device=device
            ),
            "gt_box_sizes_normalized": torch.rand(
                B, max_obj, 3, device=device
            ),
        }

        preds = {"outputs": outputs_final, "aux_outputs": []}
        return preds, targets

    def test_criterion_losses_match(self, criterion_pair):
        """Total loss and every loss_dict entry should be numerically equal."""
        native_crit, pc_crit = criterion_pair
        preds, targets = self._make_synthetic_data()

        # Deep-copy because both criteria mutate outputs and targets in-place
        n_loss, n_dict = native_crit(
            copy.deepcopy(preds), copy.deepcopy(targets)
        )
        p_loss, p_dict = pc_crit(
            copy.deepcopy(preds), copy.deepcopy(targets)
        )

        torch.testing.assert_close(
            n_loss, p_loss, atol=1e-5, rtol=1e-4, msg="Total loss mismatch"
        )

        # Compare individual loss terms
        for key in n_dict:
            assert key in p_dict, (
                f"Key '{key}' in native loss_dict but not Pointcept"
            )
            torch.testing.assert_close(
                n_dict[key],
                p_dict[key],
                atol=1e-5,
                rtol=1e-4,
                msg=f"loss_dict['{key}'] mismatch",
            )

        for key in p_dict:
            assert key in n_dict, (
                f"Key '{key}' in Pointcept loss_dict but not native"
            )

    def test_criterion_with_aux_outputs(self, criterion_pair):
        """Criterion should also agree when aux_outputs are present."""
        native_crit, pc_crit = criterion_pair
        preds, targets = self._make_synthetic_data()

        # Add a fake intermediate layer output (same structure as final)
        preds["aux_outputs"] = [copy.deepcopy(preds["outputs"])]

        n_loss, n_dict = native_crit(
            copy.deepcopy(preds), copy.deepcopy(targets)
        )
        p_loss, p_dict = pc_crit(
            copy.deepcopy(preds), copy.deepcopy(targets)
        )

        torch.testing.assert_close(
            n_loss, p_loss, atol=1e-5, rtol=1e-4,
            msg="Total loss mismatch (with aux_outputs)",
        )

        for key in n_dict:
            assert key in p_dict, (
                f"Key '{key}' missing from Pointcept loss_dict (aux test)"
            )
            torch.testing.assert_close(
                n_dict[key],
                p_dict[key],
                atol=1e-5,
                rtol=1e-4,
                msg=f"loss_dict['{key}'] mismatch (with aux_outputs)",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Test D — Config audit (informational, no assertions)
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigAudit:
    """Print a comparison table of native 3DETR defaults vs Pointcept config.

    This test always passes — it is purely informational.
    """

    def test_hyperparameter_audit(self, capsys):
        rows = [
            ("Parameter", "Native 3DETR Default", "Pointcept Config"),
            ("-" * 30, "-" * 22, "-" * 22),
            ("loss_giou_weight", "0 (DISABLED)", "1.0"),
            ("loss_no_object_weight", "0.2", "0.25"),
            ("matcher_giou_cost", "2.0", "1.0"),
            ("matcher_cls_cost", "1.0", "1.0"),
            ("matcher_center_cost", "0.0 (disabled)", "5.0"),
            ("matcher_objectness_cost", "0.0 (disabled)", "0.1"),
            ("max_epoch", "720", "90"),
            ("lr_scheduler", "cosine+warmup", "OneCycleLR"),
            ("batch_size_per_gpu", "8", "4 (16/4 GPUs)"),
            ("box_sizes_normalized", "/ mult_factor", "/ (mult_factor + 1e-6)"),
            ("box_centers_normalized", "/ src_diff", "/ (src_diff + 1e-6)"),
        ]
        widths = [max(len(r[i]) for r in rows) for i in range(3)]
        with capsys.disabled():
            print("\n\n=== Config Audit: Native 3DETR vs Pointcept ===\n")
            for row in rows:
                line = " | ".join(v.ljust(widths[i]) for i, v in enumerate(row))
                print(line)
            print()
