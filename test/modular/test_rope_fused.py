"""PositionConfig(fused=True): RoPE as torch ops over a cached cos/sin table, against
FlashInfer's kernel (the default path)."""
from __future__ import annotations

import pytest
import torch

from mstar.engine.resources.position.config import PositionConfig
from mstar.engine.resources.position.manager import RopeManager

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _manager(fused: bool, device: str, **cfg) -> RopeManager:
    return RopeManager(
        config=PositionConfig(kv_cache="kv", fused=fused, **cfg),
        device=torch.device(device), max_seq_len=4096, head_dim=64,
    )


@pytest.mark.parametrize("cfg", [
    {"interleave": True},
    {"low_freq_factor": 1.0, "high_freq_factor": 4.0, "old_context_len": 8192},
])
def test_fused_refuses_what_it_does_not_implement(cfg):
    with pytest.raises(NotImplementedError):
        _manager(True, "cpu", **cfg)


@cuda
@pytest.mark.parametrize("compiled", [False, True])
def test_fused_matches_flashinfer(compiled):
    torch.manual_seed(0)
    ref, fused = _manager(False, "cuda", rope_theta=10000.0), _manager(True, "cuda", rope_theta=10000.0)
    n = 37
    pos = torch.randint(0, 4096, (n,), device="cuda", dtype=torch.int32)
    for m in (ref, fused):
        m._current_pos_ids["main"] = pos
    q = torch.randn(n, 12, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n, 4, 64, device="cuda", dtype=torch.bfloat16)
    want_q, want_k = ref.apply_qk(q.clone(), k.clone(), label="main")
    apply = fused.apply_qk
    if compiled:
        apply = torch.compile(lambda a, b: fused.apply_qk(a, b, label="main"), fullgraph=True)
        got_q, got_k = apply(q.clone(), k.clone())
    else:
        got_q, got_k = apply(q.clone(), k.clone(), label="main")
    assert got_q.dtype == q.dtype and got_k.shape == k.shape
    # both compute in float32 and round to bf16; sin/cos differ in the last fp32 bits
    torch.testing.assert_close(got_q, want_q, rtol=1.6e-2, atol=1.6e-2)
    torch.testing.assert_close(got_k, want_k, rtol=1.6e-2, atol=1.6e-2)
    # and they round the same way almost everywhere
    assert (got_q != want_q).float().mean() < 0.05
