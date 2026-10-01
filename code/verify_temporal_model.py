"""Optional self-check for the temporal-only model. Standalone: nothing imports it.

    python verify_temporal_model.py

Confirms that best_accuracy_temporal is the baseline plus exactly the temporal
half of the existing channel-temporal attention, and nothing else.
"""

from __future__ import annotations

import torch

from model import build_best_accuracy_model, count_trainable_parameters
from model_attention import (
    TemporalAttention1D,
    build_best_accuracy_channel_temporal_model,
)
from model_se import build_best_accuracy_se_model
from model_temporal import build_best_accuracy_temporal_model
from reproducibility import set_global_seed

OK, BAD = "  OK  ", " FAIL "
failures = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global failures
    if not condition:
        failures += 1
    print(f"[{OK if condition else BAD}] {label}" + (f"  --  {detail}" if detail else ""))


print("=" * 72)
print("TEMPORAL-ONLY MODEL -- VERIFICATION")
print("=" * 72)

set_global_seed(42)
baseline = build_best_accuracy_model()
set_global_seed(42)
temporal = build_best_accuracy_temporal_model()
set_global_seed(42)
chan_temp = build_best_accuracy_channel_temporal_model(reduction_ratio=2)

n_base = count_trainable_parameters(baseline)
n_temp = count_trainable_parameters(temporal)
n_ct = count_trainable_parameters(chan_temp)

check("parameter count is 8806", n_temp == 8806, f"got {n_temp}")
check("baseline is 8722", n_base == 8722, f"got {n_base}")
check("overhead is 2 * 7 * 6 blocks = 84", n_temp - n_base == 84)

# --- the attention module must be the SAME one, not a copy -------------------
mod_temp = temporal.features[0].temporal
mod_ct = chan_temp.features[0].attention.temporal
check(
    "same TemporalAttention1D class as channel-temporal",
    type(mod_temp) is type(mod_ct) is TemporalAttention1D,
)
check(
    "same convolution shape (1, 2, 7)",
    mod_temp.conv.weight.shape == mod_ct.conv.weight.shape == (1, 2, 7),
    f"{tuple(mod_temp.conv.weight.shape)}",
)
check("temporal conv has no bias", mod_temp.conv.bias is None)

mod_temp.load_state_dict(mod_ct.state_dict())
probe = torch.randn(16, 32, 39)
with torch.no_grad():
    same_output = torch.equal(mod_temp(probe), mod_ct(probe))
check("identical output to channel-temporal's temporal half on equal weights", same_output)

# --- the rest of the network must be untouched -------------------------------
base_sd, temp_sd = baseline.state_dict(), temporal.state_dict()
check(
    "every baseline tensor present, unchanged in shape",
    all(k in temp_sd and temp_sd[k].shape == base_sd[k].shape for k in base_sd),
    f"{len(base_sd)} tensors",
)
extra = sorted(set(temp_sd) - set(base_sd))
check(
    "the only new tensors are the six temporal convolutions",
    len(extra) == 6 and all(k.endswith("temporal.conv.weight") for k in extra),
    ", ".join(extra),
)

# Inserting any attention module consumes RNG draws, so layers built afterwards
# start from different values at the same seed. This is already true of SE, ECA
# and channel-temporal in the published sweep; assert the new model is no different.
set_global_seed(42)
se_sd = build_best_accuracy_se_model(reduction_ratio=2).state_dict()
matches = lambda sd: sum(torch.equal(base_sd[k], sd[k]) for k in base_sd if k in sd)
check(
    "same seed behaviour as the existing SE variant",
    matches(temp_sd) == matches(se_sd),
    f"temporal {matches(temp_sd)}/{len(base_sd)}, SE {matches(se_sd)}/{len(base_sd)} "
    "tensors equal to baseline (RNG ordering, affects every variant alike)",
)

# --- forward pass ------------------------------------------------------------
temporal.eval()
with torch.no_grad():
    y = temporal(torch.randn(8, 4, 1250))
check("forward (8, 4, 1250) -> (8, 6)", tuple(y.shape) == (8, 6), f"{tuple(y.shape)}")
check("output is finite", bool(torch.isfinite(y).all()))

set_global_seed(123)
a = build_best_accuracy_temporal_model().state_dict()
set_global_seed(123)
b = build_best_accuracy_temporal_model().state_dict()
check("same seed -> identical initial weights", all(torch.equal(a[k], b[k]) for k in a))

print("-" * 72)
print(f"baseline           {n_base:>6,} params")
print(f"temporal only      {n_temp:>6,} params   (+{n_temp - n_base})")
print(f"channel-temporal   {n_ct:>6,} params   (+{n_ct - n_base}, r=2)")
print("-" * 72)
print("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
print("=" * 72)
raise SystemExit(1 if failures else 0)
