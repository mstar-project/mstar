"""Parity tests for the Triton fused MoE dispatch vs the naive path.

Skips automatically when CUDA is unavailable (required for the fused
kernel).  The naive path in ``_dispatch_experts_fused`` runs on CPU, but
we need CUDA bf16 tensors to exercise the Triton kernels, so the
comparison is done entirely on GPU.

Problem shapes mirror the live Qwen3-Omni configs:

* Thinker: ``hidden=2048, moe_intermediate_size=768, num_experts=128,
  top_k=8, norm_topk_prob=True``
* Talker : ``hidden=1024, moe_intermediate_size=384, num_experts=128,
  top_k=8, norm_topk_prob=False`` (plus the shared expert + sigmoid
  gate, which are not routed and are checked via the full block).
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.components import GatedMLP, SparseMoeBlock, SparseMoeBlockWithSharedExpert
from mstar.model.components.moe import dispatch_experts_fused as _dispatch_experts_fused

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")


def _random_router_output(
    hidden_states: torch.Tensor,
    num_experts: int,
    top_k: int,
    norm_topk_prob: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a plausible router output without a real gate module."""
    logits = torch.randn(
        hidden_states.shape[0],
        num_experts,
        device=hidden_states.device,
        dtype=torch.float32,
    )
    probs = torch.softmax(logits, dim=-1)
    top_w, top_i = torch.topk(probs, top_k, dim=-1)
    if norm_topk_prob:
        top_w = top_w / top_w.sum(dim=-1, keepdim=True)
    return top_w.to(hidden_states.dtype), top_i


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)


# ------------------------------------------------------------------
# Low-level parity: fused_experts vs _dispatch_experts_fused
# ------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 4, 16, 64])
@pytest.mark.parametrize(
    "hidden,inter,num_experts,top_k",
    [
        (2048, 768, 128, 8),  # Thinker
        (1024, 384, 128, 8),  # Talker routed experts
    ],
)
def test_fused_experts_numerical_parity(num_tokens, hidden, inter, num_experts, top_k):
    from mstar.utils.fused_moe import fused_experts

    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device=device, dtype=dtype) * 0.02
    w2 = torch.randn(num_experts, hidden, inter, device=device, dtype=dtype) * 0.02

    topk_weights, topk_ids = _random_router_output(
        hidden_states,
        num_experts,
        top_k,
        norm_topk_prob=True,
    )

    fused_out = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids)
    naive_out = _dispatch_experts_fused(
        hidden_states,
        w1,
        w2,
        num_experts,
        topk_ids,
        topk_weights,
    )

    assert fused_out.shape == hidden_states.shape
    assert fused_out.dtype == dtype
    # bf16 accumulation -> loose tolerance; sglang uses atol=2e-2 in its
    # own parity tests so match that here.
    torch.testing.assert_close(fused_out, naive_out, atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------------
# Block-level parity: full forward through the nn.Module path
# ------------------------------------------------------------------


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_thinker_block_parity(num_tokens):
    hidden = 2048
    inter = 768
    num_experts = 128
    top_k = 8

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = SparseMoeBlock(
        hidden_size=hidden,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=inter,
        norm_topk_prob=True,
    ).to(device=device, dtype=dtype)
    # Initialize expert and gate parameters to reasonable small values.
    with torch.no_grad():
        block.experts.gate_up_proj.normal_(std=0.02)
        block.experts.down_proj.normal_(std=0.02)
        block.gate.weight.normal_(std=0.02)

    x = torch.randn(num_tokens, hidden, device=device, dtype=dtype)

    # Force naive path by disabling the fused flag on the module.
    import mstar.model.components.moe as moe_mod

    saved = moe_mod._HAS_FUSED
    try:
        moe_mod._HAS_FUSED = False
        naive_out = block(x)
        moe_mod._HAS_FUSED = True
        fused_out = block(x)
    finally:
        moe_mod._HAS_FUSED = saved

    torch.testing.assert_close(fused_out, naive_out, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 4, 16])
def test_talker_block_parity(num_tokens):
    """Talker block adds a shared expert + sigmoid gate on top of the routed
    dispatch.  Both shared and routed halves are exercised end-to-end."""
    hidden = 1024
    inter = 384
    num_experts = 128
    top_k = 8
    shared_inter = 2048

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = SparseMoeBlockWithSharedExpert(
        hidden_size=hidden,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=inter,
        norm_topk_prob=False,
        shared_expert=GatedMLP(
            hidden_size=hidden, intermediate_size=shared_inter, activation="silu",
        ),
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        block.experts.gate_up_proj.normal_(std=0.02)
        block.experts.down_proj.normal_(std=0.02)
        block.gate.weight.normal_(std=0.02)
        block.shared_expert.gate_proj.weight.normal_(std=0.02)
        block.shared_expert.up_proj.weight.normal_(std=0.02)
        block.shared_expert.down_proj.weight.normal_(std=0.02)
        block.shared_expert_gate.weight.normal_(std=0.02)

    x = torch.randn(num_tokens, hidden, device=device, dtype=dtype)

    import mstar.model.components.moe as moe_mod

    saved = moe_mod._HAS_FUSED
    try:
        moe_mod._HAS_FUSED = False
        naive_out = block(x)
        moe_mod._HAS_FUSED = True
        fused_out = block(x)
    finally:
        moe_mod._HAS_FUSED = saved

    torch.testing.assert_close(fused_out, naive_out, atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------------
# Sanity check for the naive dispatch path
# ------------------------------------------------------------------


def test_dispatch_experts_fused_sanity_cuda():
    """The naive path must still work on CUDA so the fallback is viable
    when the fused path is unavailable."""
    hidden = 64
    inter = 48
    num_experts = 4
    top_k = 2
    device = torch.device("cuda")
    dtype = torch.bfloat16

    hidden_states = torch.randn(8, hidden, device=device, dtype=dtype)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device=device, dtype=dtype) * 0.05
    w2 = torch.randn(num_experts, hidden, inter, device=device, dtype=dtype) * 0.05
    topk_weights, topk_ids = _random_router_output(
        hidden_states,
        num_experts,
        top_k,
        norm_topk_prob=True,
    )
    out = _dispatch_experts_fused(
        hidden_states,
        w1,
        w2,
        num_experts,
        topk_ids,
        topk_weights,
    )
    assert out.shape == hidden_states.shape
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


# ------------------------------------------------------------------
# Skipped slots: ids >= num_experts (the EP sentinel)
# ------------------------------------------------------------------


def _ids_with_sentinels(num_tokens, top_k, num_experts, device, frac=0.5):
    """Random int32 ids with about ``frac`` of slots set to the sentinel ``num_experts``."""
    ids = torch.randint(0, num_experts, (num_tokens, top_k), device=device, dtype=torch.int32)
    skip = torch.rand(num_tokens, top_k, device=device) < frac
    return torch.where(skip, torch.full_like(ids, num_experts), ids)


def _check_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_post_pad):
    """Check the align output against a reference built from the valid ids only."""
    flat = topk_ids.reshape(-1).cpu().tolist()
    numel = len(flat)
    slots = {e: sorted(i for i, x in enumerate(flat) if x == e) for e in range(num_experts)}
    padded = {e: -(-len(s) // block_size) * block_size for e, s in slots.items()}

    total = sum(padded.values())
    assert int(num_post_pad.item()) == total

    sorted_ids = sorted_ids.cpu().tolist()
    expert_ids = expert_ids.cpu().tolist()
    start = 0
    for e in range(num_experts):
        end = start + padded[e]
        assert expert_ids[start // block_size : end // block_size] == [e] * (padded[e] // block_size)
        span = sorted_ids[start:end]
        # Intra-expert order is unspecified (the large-batch kernel uses atomics).
        assert sorted(x for x in span if x != numel) == slots[e]
        assert span.count(numel) == padded[e] - len(slots[e])
        start = end
    # Nothing past the padded region, so no skipped slot ever reaches a GEMM.
    assert all(x == numel for x in sorted_ids[total:])


@pytest.mark.parametrize(
    "num_tokens,top_k,num_experts",
    [
        (8, 8, 64),  # small-batch kernel: EP decode, Thinker at P=2
        (3, 8, 16),  # small-batch kernel, few experts
        (256, 8, 64),  # large-batch kernels: numel >= 1024
        (16, 8, 128),  # large-batch kernels: num_experts > 64
    ],
)
@pytest.mark.parametrize("impl", ["cuda", "torch"])
def test_moe_align_block_size_skips_sentinel(num_tokens, top_k, num_experts, impl):
    from mstar.utils.fused_moe import align

    if impl == "cuda" and not align._cuda_op_available():
        pytest.skip("CUDA moe_align_block_size op could not be built")

    device = torch.device("cuda")
    block_size = 16
    topk_ids = _ids_with_sentinels(num_tokens, top_k, num_experts, device)

    max_padded = topk_ids.numel() + num_experts * (block_size - 1)
    sorted_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
    expert_ids = torch.empty(-(-max_padded // block_size), dtype=torch.int32, device=device)
    num_post_pad = torch.empty(1, dtype=torch.int32, device=device)

    if impl == "cuda":
        torch.ops._mstar_moe_C.moe_align_block_size(
            topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_post_pad
        )
    else:
        align._moe_align_block_size_torch(
            topk_ids, block_size, num_experts, sorted_ids, expert_ids, num_post_pad
        )
    _check_align(topk_ids, num_experts, block_size, sorted_ids, expert_ids, num_post_pad)


@pytest.mark.parametrize("num_tokens", [1, 8, 256])
@pytest.mark.parametrize("num_experts", [16, 64])
def test_fused_experts_skip_invalid(num_tokens, num_experts):
    from mstar.utils.fused_moe import fused_experts

    device = torch.device("cuda")
    dtype = torch.bfloat16
    hidden, inter, top_k = 256, 128, 8

    hidden_states = torch.randn(num_tokens, hidden, device=device, dtype=dtype)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device=device, dtype=dtype) * 0.05
    w2 = torch.randn(num_experts, hidden, inter, device=device, dtype=dtype) * 0.05
    topk_weights, _ = _random_router_output(hidden_states, num_experts, top_k, norm_topk_prob=True)
    topk_ids = _ids_with_sentinels(num_tokens, top_k, num_experts, device)
    skipped = topk_ids >= num_experts

    partial = fused_experts(
        hidden_states, w1, w2, topk_weights, topk_ids, reduce_results=False, skip_invalid=True
    )
    assert partial.shape == (num_tokens, top_k, hidden)
    assert (partial[skipped] == 0).all()

    # Reference: point skipped slots at expert 0 with weight 0.
    ref_ids = torch.where(skipped, 0, topk_ids).long()
    ref_w = torch.where(skipped, 0, topk_weights)
    naive = _dispatch_experts_fused(hidden_states, w1, w2, num_experts, ref_ids, ref_w)
    torch.testing.assert_close(partial.float().sum(dim=1).to(dtype), naive, atol=2e-2, rtol=2e-2)

    reduced = fused_experts(hidden_states, w1, w2, topk_weights, topk_ids, skip_invalid=True)
    torch.testing.assert_close(reduced, naive, atol=2e-2, rtol=2e-2)
