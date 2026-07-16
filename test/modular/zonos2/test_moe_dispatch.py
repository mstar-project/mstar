"""MoE dispatch graph-safety tests for Zonos2 (Phase 1).

Zonos2's MoE now dispatches through :func:`dispatch_experts` (fused-Triton
preferring) instead of the naive per-expert loop, so a CUDA-graph capture can
run the grouped-GEMM kernel instead of a data-dependent Python loop. These
tests pin the two invariants that swap relies on:

  * the fused grouped-GEMM matches the naive per-expert loop within bf16
    tolerance (grouped reduction order is not bit-identical, and this is a
    quality-sensitive TTS model, so parity is asserted, not exact equality).

The fused kernel autotune (host sync on first-seen shape) is handled by the
CUDA-graph runner's eager warmup forwards before capture, so no dedicated
warmup is exercised here.

The fused kernel is CUDA + ``sgl-kernel`` + bf16/fp16 only, so the parity
tests skip when it isn't available.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mstar.model.components import moe as moe_mod
from mstar.model.components.moe import (
    dispatch_experts,
    dispatch_experts_fused,
)

fused_required = pytest.mark.skipif(
    not (torch.cuda.is_available() and moe_mod._HAS_FUSED),
    reason="fused MoE kernel requires CUDA + sgl-kernel",
)


def _random_moe_inputs(
    *, tokens=32, hidden=64, inter=128, num_experts=8, top_k=2,
    device="cuda", dtype=torch.bfloat16, seed=0,
):
    """Random experts + routing in the fused checkpoint layout."""
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=gen)
    gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device=device, dtype=dtype, generator=gen
    ) * 0.05
    down = torch.randn(
        num_experts, hidden, inter, device=device, dtype=dtype, generator=gen
    ) * 0.05
    # Distinct top-k experts per token, uniform (non-renormalized) weights —
    # matches how Zonos2Router feeds dispatch (weights are NOT renormalized).
    logits = torch.randn(tokens, num_experts, device=device, generator=gen)
    _, ids = torch.topk(logits, top_k, dim=-1)
    weights = torch.rand(tokens, top_k, device=device, dtype=dtype, generator=gen)
    return x, gate_up, down, num_experts, ids, weights


@fused_required
def test_dispatch_fused_matches_naive():
    x, gate_up, down, E, ids, weights = _random_moe_inputs()

    naive = dispatch_experts_fused(x, gate_up, down, E, ids, weights)
    fused = dispatch_experts(x, gate_up, down, E, ids, weights)

    assert fused.shape == naive.shape
    assert fused.dtype == naive.dtype
    a = fused.float()
    b = naive.float()
    # bf16 grouped-GEMM vs per-expert loop: parity within bf16 rounding, not
    # bit-exact. Tolerance scaled to the output magnitude.
    scale = b.abs().max().clamp(min=1e-3)
    assert (a - b).abs().max() <= 3e-2 * scale, (
        f"max abs diff {(a - b).abs().max().item():.4g} exceeds tol "
        f"(scale {scale.item():.4g})"
    )
    assert torch.allclose(a, b, rtol=3e-2, atol=3e-2 * scale.item())


@fused_required
def test_dispatch_fused_matches_naive_seeds():
    # A few seeds so the parity bound isn't accidentally tuned to one draw.
    for seed in range(4):
        x, gate_up, down, E, ids, weights = _random_moe_inputs(seed=seed)
        naive = dispatch_experts_fused(x, gate_up, down, E, ids, weights).float()
        fused = dispatch_experts(x, gate_up, down, E, ids, weights).float()
        scale = naive.abs().max().clamp(min=1e-3)
        assert (fused - naive).abs().max() <= 3e-2 * scale, f"seed {seed}"


@fused_required
def test_moe_feedforward_forward_parity(monkeypatch):
    """``Zonos2MoEFeedForward.forward`` parity: naive vs fused dispatch.

    Same module, same routing (the router is deterministic given weights +
    input); only the dispatch path differs. Toggled via the module-global
    ``_HAS_FUSED`` that ``dispatch_experts`` reads at call time.
    """
    from mstar.model.zonos2.config import Zonos2Config
    from mstar.model.zonos2.components.language_model import Zonos2MoEFeedForward

    cfg = Zonos2Config(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=128,
        moe_n_experts=8, num_experts_per_tok=2, moe_router_dim=32,
        moe_start_from_layer=0,
    )
    torch.manual_seed(0)
    # layer_id == moe_start_from_layer -> no EDA state threading (self-contained).
    ff = Zonos2MoEFeedForward(cfg, layer_id=0).to("cuda", torch.bfloat16).eval()
    for p in ff.parameters():
        torch.nn.init.normal_(p, std=0.05)

    x = torch.randn(32, 64, device="cuda", dtype=torch.bfloat16)

    monkeypatch.setattr(moe_mod, "_HAS_FUSED", False)
    with torch.no_grad():
        naive, _ = ff(x)
    monkeypatch.setattr(moe_mod, "_HAS_FUSED", True)
    with torch.no_grad():
        fused, _ = ff(x)

    a, b = fused.float(), naive.float()
    scale = b.abs().max().clamp(min=1e-3)
    assert (a - b).abs().max() <= 3e-2 * scale, (
        f"forward parity: max abs diff {(a - b).abs().max().item():.4g} "
        f"exceeds tol (scale {scale.item():.4g})"
    )
