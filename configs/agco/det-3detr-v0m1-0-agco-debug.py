"""
AGCO 3DETR debug config.

Small, deterministic smoke/debug variant of det-3detr-v0m1-0-agco:
- short schedule (2 epochs)
- deterministic point sampling for train/val/test
- roundtrip frame diagnostics enabled
- prediction export enabled in tester
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
