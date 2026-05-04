"""
AGCO 3DETR debug config v2.

Adds the main learning-focused diagnostics tweaks on top of the debug config:
- gravity leveling enabled for both points and boxes
- matcher costs include center/objectness terms
- reduced min_inliers for denser supervision
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
