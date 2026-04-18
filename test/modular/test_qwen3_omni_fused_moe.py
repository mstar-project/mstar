"""Parity tests for the Triton fused MoE dispatch vs the naive path.

Skips automatically when CUDA or ``sgl-kernel`` is unavailable (both are
required for the fused kernel).  The naive path in
``_dispatch_experts_fused`` runs on CPU, but we need CUDA bf16 tensors
to exercise the Triton kernels, so the comparison is done entirely on
GPU.

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

from mminf.model.qwen3_omni.components.fused_moe.align import has_sgl_kernel
from mminf.model.qwen3_omni.components.moe import (
    Qwen3OmniSparseMoeBlock,
    Qwen3OmniTalkerSparseMoeBlock,
    _dispatch_experts_fused,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="fused MoE requires CUDA")


def _skip_if_no_sgl_kernel():
    if not has_sgl_kernel():
        pytest.skip("sgl-kernel not installed; fused MoE path unavailable")


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
    _skip_if_no_sgl_kernel()
    from mminf.model.qwen3_omni.components.fused_moe import fused_experts

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
    _skip_if_no_sgl_kernel()

    hidden = 2048
    inter = 768
    num_experts = 128
    top_k = 8

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = Qwen3OmniSparseMoeBlock(
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
    import mminf.model.qwen3_omni.components.moe as moe_mod

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
    _skip_if_no_sgl_kernel()

    hidden = 1024
    inter = 384
    num_experts = 128
    top_k = 8
    shared_inter = 2048

    device = torch.device("cuda")
    dtype = torch.bfloat16
    block = Qwen3OmniTalkerSparseMoeBlock(
        hidden_size=hidden,
        num_experts=num_experts,
        num_experts_per_tok=top_k,
        moe_intermediate_size=inter,
        norm_topk_prob=False,
        shared_expert_intermediate_size=shared_inter,
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

    import mminf.model.qwen3_omni.components.moe as moe_mod

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
# Sanity checks that don't require sgl-kernel
# ------------------------------------------------------------------


def test_dispatch_experts_fused_sanity_cuda():
    """The naive path must still work on CUDA so the fallback is viable
    when sgl-kernel is missing."""
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
