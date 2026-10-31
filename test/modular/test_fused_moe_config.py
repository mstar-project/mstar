"""Tuned-config lookup and precomputed alignment for the fused MoE."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.utils.fused_moe import kernels

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")


@pytest.fixture
def tuned_table(tmp_path, monkeypatch):
    path = tmp_path / "table.json"
    monkeypatch.setattr(kernels, "config_path", lambda E, N, K: path)
    kernels._tuned_configs.cache_clear()
    yield path
    kernels._tuned_configs.cache_clear()


def test_get_config_falls_back_to_default(tuned_table):
    up, down = kernels.get_config(M=4, E=8, N=64, K=32, top_k=2)
    assert up == down == kernels.get_default_config(4, 8, 64, 32, 2)


def test_get_config_uses_nearest_batch_and_down_overrides(tuned_table):
    tile = {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 3}
    tuned_table.write_text(json.dumps({
        "1": {**tile, "down": {"BLOCK_SIZE_K": 64, "BLOCK_SIZE_M": 64}},
        "64": {**tile, "BLOCK_SIZE_M": 64},
    }))
    up, down = kernels.get_config(M=2, E=8, N=64, K=32, top_k=2)
    assert up == tile
    # The down GEMM shares the up GEMM's alignment, so BLOCK_SIZE_M is not overridable.
    assert down == {**tile, "BLOCK_SIZE_K": 64}
    assert kernels.get_config(M=50, E=8, N=64, K=32, top_k=2)[0]["BLOCK_SIZE_M"] == 64


@CUDA
@pytest.mark.parametrize(
    "table", sorted((kernels.config_path(1, 1, 1, "x").parent).glob("*.json")), ids=lambda p: p.name,
)
def test_fused_experts_traces_with_a_symbolic_batch_on_every_table(table, monkeypatch):
    """An uncaptured step compiles fused_experts with a dynamic batch size, and
    every entry of a shipped table must trace that way (launch options included)."""
    import re

    from mstar.utils.fused_moe import fused_experts

    experts, two_inter, hidden = (int(v) for v in re.match(r"E=(\d+),N=(\d+),K=(\d+)", table.name).groups())
    monkeypatch.setattr(kernels, "config_path", lambda E, N, K: table)
    kernels._tuned_configs.cache_clear()
    torch._dynamo.reset()
    w1 = torch.randn(experts, two_inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, two_inter // 2, device="cuda", dtype=torch.bfloat16) * 0.02
    run = torch.compile(lambda x, w, i: fused_experts(x, w1, w2, w, i), dynamic=True)
    # every tuned batch size; 0 and 1 are specialized, never symbolic
    for num_tokens in sorted({max(int(m), 2) for m in json.loads(table.read_text())}):
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
        weights, ids = torch.rand(num_tokens, experts, device="cuda").topk(8, dim=-1)
        run(x, weights.to(torch.bfloat16), ids)
    kernels._tuned_configs.cache_clear()


@CUDA
@pytest.mark.parametrize("num_tokens", [1, 3, 16, 64])
def test_fused_experts_matches_naive_and_precomputed_alignment(num_tokens):
    from mstar.model.components.moe import dispatch_experts_fused
    from mstar.utils.fused_moe import MoeAlignment, fused_experts, moe_block_m
    from mstar.utils.fused_moe.align import moe_align_block_size

    # The Command A+ expert shape, which has a tuned table on GB200.
    experts, hidden, inter, top_k = 128, 4096, 1024, 8
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16) * 0.02
    w2 = torch.randn(experts, hidden, inter, device="cuda", dtype=torch.bfloat16) * 0.02
    weights, ids = torch.rand(num_tokens, experts, device="cuda").topk(top_k, dim=-1)
    weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)

    out = fused_experts(x, w1, w2, weights, ids)
    naive = dispatch_experts_fused(x, w1, w2, experts, ids, weights)
    torch.testing.assert_close(out, naive, atol=2e-2, rtol=2e-2)

    block_m = moe_block_m(num_tokens, w1, top_k)
    alignment = MoeAlignment(*moe_align_block_size(ids.to(torch.int32), block_m, experts), block_m)
    precomputed = fused_experts(x, w1, w2, weights, ids, alignment=alignment)
    torch.testing.assert_close(precomputed, out, atol=0, rtol=0)

    with pytest.raises(AssertionError, match="BLOCK_SIZE_M"):
        fused_experts(x, w1, w2, weights, ids, alignment=alignment._replace(block_m=block_m * 2))
