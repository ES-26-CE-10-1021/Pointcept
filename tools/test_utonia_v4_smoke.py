"""
Smoke test for the reworked ``PTv3m3PreEncoder`` and the v4m1 / v4m2
Utonia configs. Runs entirely offline (no HuggingFace download) with
small backbone hyperparameters so it completes in seconds on one GPU.

Coverage:
  1. Instantiate PTv3m3PreEncoder in encoder-only + FPS and full-U-Net
     + FPS modes, both with ``freeze_backbone="full"``.
  2. Forward shape check — tuple (xyz, features, inds) with the
     expected shapes and feature width.
  3. Freeze correctness — after .train(), every backbone parameter has
     ``requires_grad=False`` and frozen submodules are in eval() mode.
  4. Gradient isolation — a dummy downstream head builds a graph on
     the output leaf and backpropagates; no gradient reaches the
     backbone parameters.
  5. Config loader parse — all five touched configs load and expose
     the expected ``pre_encoder`` fields.

Usage:
    conda run -n pointcept python tools/test_utonia_v4_smoke.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Ensure the repo root is on sys.path when the script is invoked directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pointcept.models.detection_3detr.ptv3 import PTv3m3PreEncoder  # noqa: E402
from pointcept.utils.config import Config  # noqa: E402


# Tiny backbone hyperparameters — small enough to run fast, big enough
# to exercise every code path (5 enc stages, 4 dec stages).
TINY_ENC_KWARGS = dict(
    in_channels=9,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(1, 1, 1, 1, 1),
    enc_channels=(8, 16, 32, 64, 64),
    enc_num_head=(1, 1, 1, 1, 1),
    enc_patch_size=(64, 64, 64, 64, 64),
    mlp_ratio=2,
    qkv_bias=True,
    drop_path=0.0,
    shuffle_orders=False,
    pre_norm=True,
    enable_flash=False,
    rope_base=10,
)

TINY_DEC_KWARGS = dict(
    dec_depths=(1, 1, 1, 1),
    dec_channels=(8, 16, 32, 64),
    dec_num_head=(1, 1, 1, 1),
    dec_patch_size=(64, 64, 64, 64),
)


def _report(section, passed, detail=""):
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {section}" + (f" — {detail}" if detail else ""))
    return bool(passed)


def _make_inputs(device, B=2, N=512):
    # Random point cloud in a unit cube.
    xyz = torch.randn(B, N, 3, device=device) * 0.5
    # Two feature channels (e.g. stand-in RGB); backbone expects 9 total
    # with xyz + these 2 + 4 zero-padded normals/extras.
    feats = torch.randn(B, 2, N, device=device)
    return xyz, feats


def _shape_check(out, expected_B, expected_npoint, expected_C, label):
    if not (isinstance(out, tuple) and len(out) == 3):
        return _report(
            f"{label}: forward shape", False,
            f"expected 3-tuple, got {type(out).__name__}",
        )
    xyz, features, inds = out
    ok_xyz = xyz.shape == (expected_B, expected_npoint, 3)
    ok_feat = features.shape == (expected_B, expected_C, expected_npoint)
    ok_inds = inds.shape == (expected_B, expected_npoint)
    detail = (
        f"xyz={tuple(xyz.shape)} feat={tuple(features.shape)} "
        f"inds={tuple(inds.shape)} (expected C={expected_C})"
    )
    return _report(f"{label}: forward shape", ok_xyz and ok_feat and ok_inds, detail)


def _freeze_check(model, label, expect_dec_frozen):
    model.train()
    trainable = sum(p.requires_grad for p in model.parameters())
    ok_params = trainable == 0
    ok_emb_eval = not model.embedding.training
    ok_enc_eval = not model.enc.training
    detail = (
        f"trainable_params={trainable} "
        f"embedding.training={model.embedding.training} "
        f"enc.training={model.enc.training}"
    )
    ok = ok_params and ok_emb_eval and ok_enc_eval
    if expect_dec_frozen:
        dec_trainable = sum(p.requires_grad for p in model.dec.parameters())
        ok_dec = dec_trainable == 0 and not model.dec.training
        detail += f" dec.training={model.dec.training} dec_trainable={dec_trainable}"
        ok = ok and ok_dec
    return _report(f"{label}: freeze state", ok, detail)


def _grad_isolation_check(model, xyz, feats, label):
    model.train()
    out_xyz, out_feat, _ = model(xyz, feats)
    C = out_feat.shape[1]
    head = nn.Linear(C, 1).to(out_feat.device)
    # (B, npoint, C) → (B, npoint, 1) → scalar loss.
    pred = head(out_feat.transpose(1, 2))
    loss = pred.sum()
    loss.backward()
    backbone_grads = [
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    ]
    leaked = any(backbone_grads)
    head_has_grad = head.weight.grad is not None
    return _report(
        f"{label}: gradient isolation",
        (not leaked) and head_has_grad,
        f"backbone_leak={leaked} head_grad_present={head_has_grad}",
    )


def test_encoder_only_fps(device):
    print("\n── encoder-only + FPS ──")
    model = PTv3m3PreEncoder(
        pretrained=None,
        enc_mode=True,
        npoint=128,
        freeze_backbone="full",
        **TINY_ENC_KWARGS,
    ).to(device)
    ok = _report("encoder-only + FPS: instantiate", True)
    xyz, feats = _make_inputs(device)
    out = model(xyz, feats)
    ok &= _shape_check(out, 2, 128, TINY_ENC_KWARGS["enc_channels"][-1],
                       "encoder-only + FPS")
    ok &= _freeze_check(model, "encoder-only + FPS", expect_dec_frozen=False)
    ok &= _grad_isolation_check(model, xyz, feats, "encoder-only + FPS")
    return ok


def test_unet_fps(device):
    print("\n── full U-Net + FPS ──")
    model = PTv3m3PreEncoder(
        pretrained=None,
        enc_mode=False,
        npoint=128,
        freeze_backbone="full",
        **TINY_ENC_KWARGS,
        **TINY_DEC_KWARGS,
    ).to(device)
    ok = _report("full U-Net + FPS: instantiate", True)
    xyz, feats = _make_inputs(device)
    out = model(xyz, feats)
    ok &= _shape_check(out, 2, 128, TINY_DEC_KWARGS["dec_channels"][0],
                       "full U-Net + FPS")
    ok &= _freeze_check(model, "full U-Net + FPS", expect_dec_frozen=True)
    ok &= _grad_isolation_check(model, xyz, feats, "full U-Net + FPS")
    return ok


def test_config_loader():
    print("\n── config loader ──")
    configs = [
        ("det-3detr-utonia-v1m1-0-scannet.py", dict(enc_mode=True, freeze_backbone="full"), None),
        ("det-3detr-utonia-v2m1-0-scannet.py", dict(enc_mode=True, freeze_backbone="full"), None),
        ("det-3detr-utonia-v3m1-0-scannet.py", dict(enc_mode=True, freeze_backbone="full"), None),
        ("det-3detr-utonia-v4m1-0-scannet.py", dict(enc_mode=True, freeze_backbone="full"), 2048),
        ("det-3detr-utonia-v4m2-0-scannet.py", dict(enc_mode=False, freeze_backbone="full"), 2048),
    ]
    ok = True
    for name, expected_pre_encoder, expected_npoint in configs:
        path = REPO_ROOT / "configs" / "scannet" / name
        try:
            cfg = Config.fromfile(str(path))
            pre = cfg.model["pre_encoder"]
            mismatch = [
                k for k, v in expected_pre_encoder.items() if pre.get(k) != v
            ]
            if expected_npoint is not None and pre.get("npoint") != expected_npoint:
                mismatch.append("npoint")
            # Old freeze kwargs must be absent.
            for stale in ("freeze", "freeze_eval", "freeze_no_grad"):
                if stale in pre:
                    mismatch.append(f"stale:{stale}")
            ok &= _report(
                f"config {name}", not mismatch,
                "mismatched: " + ",".join(mismatch) if mismatch else "ok",
            )
        except Exception as e:
            ok &= _report(f"config {name}", False, f"{type(e).__name__}: {e}")
    return ok


def main():
    if not torch.cuda.is_available():
        print("[SKIP] no CUDA device — PTv3m3 requires GPU (spconv).")
        sys.exit(2)
    device = torch.device("cuda")

    results = []
    results.append(test_encoder_only_fps(device))
    results.append(test_unet_fps(device))
    results.append(test_config_loader())

    print()
    if all(results):
        print("ALL SMOKE TESTS PASSED")
        sys.exit(0)
    else:
        print("SOME SMOKE TESTS FAILED — see above")
        sys.exit(1)


if __name__ == "__main__":
    main()
