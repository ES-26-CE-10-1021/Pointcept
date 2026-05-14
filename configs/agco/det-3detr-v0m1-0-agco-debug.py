"""
3DETR on AGCO — v0m1-0-debug: Deterministic 2-epoch smoke variant.

Inherits the full v0m1-0-agco config; overrides only:
  - epoch=2, eval_epoch=1 (short schedule)
  - deterministic point sampling for train/val/test (seed=123)
  - debug_roundtrip_check=True (frame-roundtrip diagnostics)
  - save_predictions=True in the tester

Usage:
    sh scripts/train.sh -d agco -c det-3detr-v0m1-0-agco-debug -n debug -g 1
"""

_base_ = ["./det-3detr-v0m1-0-agco.py"]

# Short smoke schedule
epoch = 2
eval_epoch = 1

# Enable prediction export from test runs
test = dict(type="ObjDetTester", verbose=True, save_predictions=True)

# Deterministic/debug dataset options
# These are merged into the base config's data.{train,val,test} dicts.
data = dict(
    train=dict(
        deterministic_debug=True,
        deterministic_seed=123,
        debug_roundtrip_check=True,
    ),
    val=dict(
        deterministic_debug=True,
        deterministic_seed=123,
        debug_roundtrip_check=True,
    ),
    test=dict(
        deterministic_debug=True,
        deterministic_seed=123,
        debug_roundtrip_check=True,
    ),
)
