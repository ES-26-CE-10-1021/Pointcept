"""
3DETR on AGCO — v0m1-0-debug-v2: 40-epoch learning-focused debug variant.

Inherits v0m1-0-agco-debug and additionally:
  - epoch=40, eval_epoch=10 (long enough for loss curves to develop)
  - gravity leveling ON for points and boxes (require_gravity_align=True)
  - min_inliers=200 (denser GT supervision)
  - matcher costs widened: cost_objectness=1.0, cost_center=5.0
    (still 3DETR-style; not the SUN-like criterion used by v1m3/v2m3/v3m3).

Usage:
    sh scripts/train.sh -d agco -c det-3detr-v0m1-0-agco-debug-v2 -n debug_v2 -g 1
"""

_base_ = ["./det-3detr-v0m1-0-agco-debug.py"]

epoch = 40
eval_epoch = 10

# Denser GT supervision for debugging
min_inliers = 200

# Stronger matching signal for localization/assignment
model = dict(
    criterion=dict(
        matcher_cfg=dict(
            cost_class=1.0,
            cost_objectness=1.0,
            cost_giou=2.0,
            cost_center=5.0,
        ),
    ),
)

# Keep train/val/test frame policy consistent with gravity leveling on
# to reduce inter-scene orientation variance.
data = dict(
    train=dict(
        min_inliers=min_inliers,
        apply_r_level_to_points=True,
        apply_r_level_to_boxes=True,
        require_gravity_align=True,
    ),
    val=dict(
        min_inliers=min_inliers,
        apply_r_level_to_points=True,
        apply_r_level_to_boxes=True,
        require_gravity_align=True,
    ),
    test=dict(
        min_inliers=min_inliers,
        apply_r_level_to_points=True,
        apply_r_level_to_boxes=True,
        require_gravity_align=True,
    ),
)
