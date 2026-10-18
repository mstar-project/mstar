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


def test_fused_refuses_a_rope_dtype_it_does_not_compute_in():
    with pytest.raises(NotImplementedError):
        _manager(True, "cpu", rope_dtype=torch.bfloat16)


@cuda
def test_fused_refuses_overrides_its_table_ignores():
    """The table is the config's: a call may repeat its parameters, not change them."""
    fused = _manager(True, "cuda", rope_theta=10000.0)
    fused._current_pos_ids["main"] = torch.zeros(2, dtype=torch.int32, device="cuda")
    q, k = torch.randn(2, 2, 64, device="cuda"), torch.randn(2, 2, 64, device="cuda")
    # what the shared attention layer passes: Llama 3.1's factors at scale 1, a no-op
    fused.apply_qk(q, k, label="main", rope_theta=10000.0, rope_scale=1.0,
                   low_freq_factor=1.0, high_freq_factor=4.0, old_context_len=8192)
    with pytest.raises(ValueError, match="rope_theta"):
        fused.apply_qk(q, k, label="main", rope_theta=1e6)
    with pytest.raises(ValueError, match="rotary_dim"):
        fused.apply_qk(q, k, label="main", rotary_dim=32)
    with pytest.raises(ValueError, match="low_freq_factor"):
        fused.apply_qk(q, k, label="main", rope_scale=8.0, low_freq_factor=1.0)


@pytest.mark.parametrize("fused", [True, False])
def test_fused_admit_refuses_positions_past_its_table(fused):
    """The fused path indexes its table unchecked, so admit refuses a step that would
    run past it; FlashInfer computes any position."""
    from mstar.engine.resources.position.config import PositionStep
    from mstar.engine.resources.step import Segment, StepContext

    m = _manager(fused, "cpu")
    m._counters["r"] = {"main": 4090}
    ctx = StepContext(request_ids=("r",), graph_walk="w", slot=0, capture=False)
    assert m.admit(PositionStep(segments=(Segment("r", "main", 6),)), ctx).ok
    outcome = m.admit(PositionStep(segments=(Segment("r", "main", 7),)), ctx)
    assert outcome.ok != fused
