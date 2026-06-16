"""Regression tests for the weight-load allocation-dtype fix.

`get_submodule` implementations build modules on the meta device and call
`to_empty(device)` to materialise storage. `to_empty` allocates each param
in its *current* dtype, so the dtype cast must happen on the meta module
(metadata-only, no allocation) BEFORE `to_empty` — otherwise params are first
materialised in the meta default (float32, 2x bf16) and only then down-cast,
doubling the load-time VRAM peak (the TP=4 OOM that motivated this change).

These tests pin the contract with a tiny module so they run CPU-only:
  - cast-before-to_empty allocates directly in the target dtype,
  - the engine-manager's None hint leaves params at the default (float32),
  - the inverted order still ends at the right dtype but is the pattern we
    explicitly avoid (documented here so the intent is greppable).
"""
import torch
from torch import nn


def _build_meta_module() -> nn.Module:
    with torch.device("meta"):
        return nn.Sequential(
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Linear(16, 8),
        )


def test_meta_cast_before_to_empty_allocates_in_dtype():
    m = _build_meta_module()
    # The fix's pattern: cast on meta (no allocation), then materialise.
    m.to(torch.bfloat16)
    m.to_empty(device="cpu")
    for name, p in m.named_parameters():
        assert p.dtype == torch.bfloat16, f"{name} is {p.dtype}, expected bf16"
    assert not any(p.is_meta for p in m.parameters())


def test_none_dtype_leaves_default_float32():
    m = _build_meta_module()
    # autocast_dtype=None path: skip the cast, params stay at the meta default.
    m.to_empty(device="cpu")
    for name, p in m.named_parameters():
        assert p.dtype == torch.float32, f"{name} is {p.dtype}, expected fp32"


def test_to_on_meta_is_in_place_and_zero_alloc():
    # `.to(dtype)` on a meta module mutates in place (returns self) and
    # allocates nothing — the property the fix relies on to cast safely
    # before to_empty.
    m = _build_meta_module()
    ret = m.to(torch.bfloat16)
    assert ret is m
    assert all(p.is_meta for p in m.parameters())
    assert all(p.dtype == torch.bfloat16 for p in m.parameters())
