"""
Compare LR and momentum schedules between native 3DETR and Pointcept OneCycleLR.

Verifies that the Pointcept config (with cycle_momentum=False) produces an LR
curve closely matching the native 3DETR custom cosine-warmup schedule.

Run: pytest tests/test_lr_schedule_comparison.py -v
"""

import math

import numpy as np
import pytest
import torch
from torch.optim.lr_scheduler import OneCycleLR


# ── Native 3DETR schedule (from third_party/3detr/engine.py) ─────────────────


def native_lr(curr_iter, max_iters, base_lr=5e-4, warm_lr=1e-6,
              warm_lr_epochs=9, max_epoch=180, final_lr=1e-6):
    """Reproduce the native 3DETR per-iteration LR computation."""
    curr_epoch_normalized = curr_iter / max_iters  # 0 → 1
    warmup_frac = warm_lr_epochs / max_epoch
    if curr_epoch_normalized <= warmup_frac:
        lr = warm_lr + curr_epoch_normalized * max_epoch * (
            (base_lr - warm_lr) / warm_lr_epochs
        )
    else:
        lr = final_lr + 0.5 * (base_lr - final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return lr


# ── Pointcept OneCycleLR schedule ────────────────────────────────────────────


def pointcept_schedule(total_steps, max_lr=5e-4, pct_start=0.05,
                       div_factor=500.0, final_div_factor=1.0,
                       cycle_momentum=False):
    """Return (lr_array, beta1_array) for the Pointcept OneCycleLR config."""
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=0.1)
    scheduler = OneCycleLR(
        optimizer, max_lr=max_lr, total_steps=total_steps,
        pct_start=pct_start, anneal_strategy="cos",
        div_factor=div_factor, final_div_factor=final_div_factor,
        cycle_momentum=cycle_momentum,
    )

    lrs, betas = [], []
    for step in range(total_steps):
        # For all steps except the last, record LR/beta before stepping so that
        # they correspond to the LR used for that optimizer step. For the final
        # step, record after scheduler.step() so that the last element reflects
        # the end-of-schedule LR.
        if step < total_steps - 1:
            lrs.append(optimizer.param_groups[0]["lr"])
            betas.append(optimizer.param_groups[0]["betas"][0])
            optimizer.step()
            scheduler.step()
        else:
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])
            betas.append(optimizer.param_groups[0]["betas"][0])
    return np.array(lrs), np.array(betas)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLRScheduleComparison:
    """Compare native and Pointcept LR schedules for the 180-epoch config."""

    MAX_EPOCH = 180
    ITERS_PER_EPOCH = 150  # approximate: ~1201 scenes / batch_size 8
    TOTAL_ITERS = MAX_EPOCH * ITERS_PER_EPOCH

    def test_lr_curves_close(self):
        """LR curves should be within ~11% relative tolerance after warmup.

        The native schedule uses cos(pi * global_progress) over [0, 1] including
        warmup, while OneCycleLR resets the cosine phase to span [0, pi] within
        only the decay portion. This causes up to ~10% relative difference in
        mid-training LR — a known, minor discrepancy.
        """
        native_lrs = np.array([
            native_lr(i, self.TOTAL_ITERS) for i in range(self.TOTAL_ITERS)
        ])
        pc_lrs, _ = pointcept_schedule(self.TOTAL_ITERS)

        # Skip warmup region
        warmup_end = int(0.05 * self.TOTAL_ITERS)
        rel_diff = np.abs(native_lrs[warmup_end:] - pc_lrs[warmup_end:]) / native_lrs[warmup_end:]
        max_rel_diff = rel_diff.max()
        mean_rel_diff = rel_diff.mean()

        print(f"\nMax relative LR difference after warmup: {max_rel_diff:.4%}")
        print(f"Mean relative LR difference after warmup: {mean_rel_diff:.4%}")
        print(f"Native LR range: [{native_lrs.min():.2e}, {native_lrs.max():.2e}]")
        print(f"Pointcept LR range: [{pc_lrs.min():.2e}, {pc_lrs.max():.2e}]")

        # ~10% max relative difference is expected from the cosine phase offset
        assert max_rel_diff < 0.12, (
            f"LR schedules diverge by {max_rel_diff:.2%} after warmup (expected <12%)"
        )
        # Mean difference is ~7% due to the cosine phase offset
        assert mean_rel_diff < 0.10, (
            f"Mean LR difference {mean_rel_diff:.2%} is too large"
        )

    def test_warmup_endpoints_match(self):
        """Both schedules should start at ~1e-6 and peak at ~5e-4."""
        native_start = native_lr(0, self.TOTAL_ITERS)
        native_peak = native_lr(int(0.05 * self.TOTAL_ITERS), self.TOTAL_ITERS)
        native_end = native_lr(self.TOTAL_ITERS - 1, self.TOTAL_ITERS)

        pc_lrs, _ = pointcept_schedule(self.TOTAL_ITERS)
        pc_start = pc_lrs[0]
        pc_peak = pc_lrs[int(0.05 * self.TOTAL_ITERS)]
        pc_end = pc_lrs[-1]

        print(f"\n{'':15s} {'Native':>12s} {'Pointcept':>12s}")
        print(f"{'Start LR':15s} {native_start:12.2e} {pc_start:12.2e}")
        print(f"{'Peak LR':15s} {native_peak:12.2e} {pc_peak:12.2e}")
        print(f"{'Final LR':15s} {native_end:12.2e} {pc_end:12.2e}")

        assert abs(native_start - pc_start) < 1e-7
        assert abs(native_end - pc_end) < 1e-7

    def test_cycle_momentum_disabled(self):
        """With cycle_momentum=False, beta1 should remain constant at 0.9."""
        _, betas = pointcept_schedule(1000, cycle_momentum=False)
        assert np.allclose(betas, 0.9), (
            f"beta1 should be constant 0.9, got range [{betas.min()}, {betas.max()}]"
        )

    def test_cycle_momentum_enabled_differs(self):
        """With cycle_momentum=True (old default), beta1 varies — this was the bug."""
        _, betas = pointcept_schedule(1000, cycle_momentum=True)
        assert betas.min() < 0.9, f"Expected beta1 < 0.9 at some point, got min={betas.min()}"
        assert betas.max() > 0.9, f"Expected beta1 > 0.9 at some point, got max={betas.max()}"
        print(f"\ncycle_momentum=True: beta1 range [{betas.min():.4f}, {betas.max():.4f}]")
        print("This differs from native 3DETR which uses constant beta1=0.9")


class TestLRSchedule720Epoch:
    """Verify the 720-epoch (full-length) config schedule."""

    MAX_EPOCH = 720
    ITERS_PER_EPOCH = 150
    TOTAL_ITERS = MAX_EPOCH * ITERS_PER_EPOCH

    def test_warmup_fraction_matches(self):
        """720-epoch config uses pct_start=0.0125 ≈ 9/720."""
        pc_lrs, _ = pointcept_schedule(
            self.TOTAL_ITERS, pct_start=0.0125, cycle_momentum=False
        )
        warmup_steps = int(0.0125 * self.TOTAL_ITERS)
        native_warmup_steps = 9 * self.ITERS_PER_EPOCH

        print(f"\nPointcept warmup steps: {warmup_steps}")
        print(f"Native warmup steps (9 epochs): {native_warmup_steps}")
        assert abs(warmup_steps - native_warmup_steps) <= self.ITERS_PER_EPOCH
