#!/usr/bin/env python
# scripts/smoke_test.py
# Verify the pixi env is functional on the current node. Run via:
#   pixi run smoke-test
#
# Two check categories:
#   - hard: required for training (failure -> nonzero exit)
#   - soft: known to fail in headless / cluster environments (failure -> warning)

import sys


HARD_FAIL = 0
SOFT_WARN = 0


def check(name: str, fn, *, soft: bool = False) -> None:
    global HARD_FAIL, SOFT_WARN
    try:
        fn()
        print(f"  [OK]    {name}")
    except Exception as exc:  # noqa: BLE001
        label = "[WARN]" if soft else "[FAIL]"
        print(f"  {label}  {name}: {exc}")
        if soft:
            SOFT_WARN += 1
        else:
            HARD_FAIL += 1


def main() -> int:
    # ---------------- torch ----------------
    def torch_basic():
        import torch
        print(f"          torch {torch.__version__}, cuda runtime {torch.version.cuda}")

    check("torch import", torch_basic)

    import torch
    if torch.cuda.is_available():
        def cuda_info():
            cap = torch.cuda.get_device_capability(0)
            print(f"          device: {torch.cuda.get_device_name(0)}")
            print(f"          capability: sm_{cap[0]}{cap[1]}")
            if cap[0] < 10:
                raise RuntimeError(
                    f"Expected sm_100 (B200); got sm_{cap[0]}{cap[1]}. "
                    "Wrong wheel resolved, or wrong node."
                )

        check("cuda capability >= sm_100", cuda_info)
    else:
        print("  [SKIP]  cuda checks (no GPU visible — CPU-only session)")

    # ---------------- core extensions (must work) ----------------
    check("spconv",          lambda: __import__("spconv.pytorch"))
    check("torch_cluster",   lambda: __import__("torch_cluster"))
    check("torch_scatter",   lambda: __import__("torch_scatter"))
    check("torch_sparse",    lambda: __import__("torch_sparse"))
    check("torch_geometric", lambda: __import__("torch_geometric"))
    check("pointops",        lambda: __import__("pointops"))
    check("pointgroup_ops",  lambda: __import__("pointgroup_ops"))
    check("peft",            lambda: __import__("peft"))
    check("timm",            lambda: __import__("timm"))

    # ---------------- soft checks ----------------

    # FA4 is Blackwell-native but its API lives under flash_attn.cute, not the
    # FA2-compatible flash_attn namespace that PTv3 / Pointcept / 3DETR import
    # from. This check confirms FA4 itself loaded; it does NOT mean upstream
    # code is using it — that requires patching call sites.
    def fa4_cute():
        from flash_attn.cute import flash_attn_func  # noqa: F401
    check("flash_attn.cute (FA4)", fa4_cute, soft=True)

    # ---------------- end-to-end GPU exercise ----------------
    if torch.cuda.is_available():
        def spconv_forward():
            import spconv.pytorch as spconv
            coords = torch.zeros((4, 4), dtype=torch.int32, device="cuda")
            coords[:, 1:] = torch.randint(0, 16, (4, 3), device="cuda", dtype=torch.int32)
            feats = torch.randn((4, 8), device="cuda")
            x = spconv.SparseConvTensor(feats, coords, (16, 16, 16), batch_size=1)
            conv = spconv.SubMConv3d(8, 16, kernel_size=3).cuda()
            _ = conv(x)

        check("spconv forward (NVRTC JIT on first call)", spconv_forward)

    # ---------------- summary ----------------
    print()
    if HARD_FAIL:
        print(f"{HARD_FAIL} required check(s) FAILED, {SOFT_WARN} optional warning(s).")
        return 1
    if SOFT_WARN:
        print(f"All required checks passed; {SOFT_WARN} optional warning(s) above.")
    else:
        print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
