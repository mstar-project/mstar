"""Host-side contract of per-token-group fp8 quant, runnable without a GPU."""
from __future__ import annotations

import pytest
import torch
import triton

from mstar.utils import quant_fp8
from mstar.utils.quant_fp8 import FP8_DTYPE


class _Recorder:
    """Stand-in for a ``@triton.jit`` kernel: ``kernel[grid](*args, **kw)``."""

    def __init__(self):
        self.calls: list[dict] = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append({"grid": grid, "args": args, "kwargs": kwargs})

        return launch


def test_per_token_group_quant_fp8_host_contract(monkeypatch):
    rec = _Recorder()
    monkeypatch.setattr(quant_fp8, "per_token_group_quant_fp8_kernel", rec)
    # The conftest triton stub has no next_power_of_2; real triton does.
    monkeypatch.setattr(triton, "next_power_of_2", lambda n: 1 << (n - 1).bit_length(), raising=False)

    x = torch.randn(6, 384, dtype=torch.bfloat16)
    x_q, x_s = quant_fp8.per_token_group_quant_fp8(x, 128)

    assert x_q.dtype == FP8_DTYPE and x_q.shape == x.shape
    assert x_s.dtype == torch.float32 and x_s.shape == (6, 3)
    (call,) = rec.calls
    groups = quant_fp8.GROUPS_PER_PROGRAM
    assert call["grid"] == (-(-6 * 3 // groups),)  # 18 groups, GROUPS per program
    assert call["args"][3] == 128 and call["args"][4] == 6 * 3
    assert call["kwargs"] == {"BLOCK": 128, "GROUPS": groups}
    assert call["args"][6] == torch.finfo(FP8_DTYPE).min and call["args"][7] == torch.finfo(FP8_DTYPE).max

    with pytest.raises(AssertionError, match="multiple of group_size"):
        quant_fp8.per_token_group_quant_fp8(torch.randn(2, 100, dtype=torch.bfloat16), 128)
