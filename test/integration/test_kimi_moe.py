"""M2 golden tests for Kimi-K2.7 fine-grained MoE.

Verifies the new DeepSeek-V3 MoE math against independent references:
  - the group-limited sigmoid ``noaux_tc`` router (KimiMoEGate),
  - the fused expert dispatch (reused fused-expert GEMM), and
  - the full MoE block (routed + ungated shared expert).

References are inlined (self-contained; no dependency on the local golden
harness) and cited to vLLM ``fused_moe/cpu_fused_moe.py::grouped_topk`` and
``models/deepseek_v2.py::DeepseekV2MoE``.

GPU test: the fused expert GEMM (``fused_experts``) is CUDA/bf16-only, so the
block/dispatch tests run on ``cuda``; the suite skips without a GPU.

Run:  pytest test/integration/test_kimi_moe.py -v
"""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.components.moe import _dispatch
from mstar.model.kimi_k2_7.components.language_model import build_moe_block
from mstar.model.kimi_k2_7.components.moe import KimiMoEGate
from mstar.model.kimi_k2_7.config import KimiK2Config

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="M2 golden tests need a GPU (fused expert GEMM is CUDA/bf16-only)",
)

DEVICE = "cuda"


# --------------------------------------------------------------------------
# Independent references (cited to vLLM cpu_fused_moe.py / deepseek_v2.py)
# --------------------------------------------------------------------------

def _ref_grouped_topk(
    logits: torch.Tensor, bias: torch.Tensor, n_group, topk_group, top_k,
    norm_topk_prob, routed_scaling_factor,
):
    """vLLM ``grouped_topk`` (sigmoid + noaux_tc) -> (weights, ids)."""
    scores = logits.float().sigmoid()
    T = scores.shape[0]
    original = scores
    scores = scores + bias.unsqueeze(0)
    group_scores = scores.view(T, n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(T, n_group, scores.shape[-1] // n_group)
        .reshape(T, -1)
    )
    masked = scores.masked_fill(~score_mask.bool(), float("-inf"))
    ids = torch.topk(masked, k=top_k, dim=-1, sorted=False)[1]
    weights = original.gather(1, ids)
    if norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights * routed_scaling_factor
    return weights, ids


def _ref_routed_experts(h, gate_up, down, weights, ids):
    """Naive per-token top-k expert loop matching ``fused_experts`` semantics."""
    T, H = h.shape
    inter = down.shape[-1]
    out = torch.zeros(T, H, dtype=h.dtype, device=h.device)
    for t in range(T):
        for j in range(ids.shape[1]):
            e = int(ids[t, j])
            gu = gate_up[e] @ h[t]  # (2*inter,)
            g, u = gu[:inter], gu[inter:]
            expert_out = down[e] @ (F.silu(g) * u)  # (H,)
            out[t] += weights[t, j] * expert_out
    return out


def _ref_swiglu(x, gate_w, up_w, down_w):
    return F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), down_w)


def _dense_combine(ids, weights, num_experts):
    """Scatter (ids, weights) into a dense (T, E) vector for order-insensitive
    comparison (topk with sorted=False returns experts in arbitrary order)."""
    dense = torch.zeros(ids.shape[0], num_experts, device=ids.device)
    dense.scatter_(1, ids, weights.float())
    return dense


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

def test_moe_gate_matches_reference():
    torch.manual_seed(0)
    cfg = KimiK2Config.reduced()
    gate = KimiMoEGate(
        cfg.hidden_size, cfg.n_routed_experts, cfg.num_experts_per_tok,
        cfg.n_group, cfg.topk_group, cfg.routed_scaling_factor,
        cfg.scoring_func, cfg.topk_method, cfg.norm_topk_prob,
    ).to(DEVICE)
    W = torch.randn(cfg.n_routed_experts, cfg.hidden_size, device=DEVICE)
    bias = torch.randn(cfg.n_routed_experts, device=DEVICE)
    gate.weight.data.copy_(W)
    gate.e_score_correction_bias.data.copy_(bias)

    h = torch.randn(6, cfg.hidden_size, device=DEVICE)
    got_w, got_ids = gate(h)

    logits = F.linear(h.float(), W.float())
    exp_w, exp_ids = _ref_grouped_topk(
        logits, bias, cfg.n_group, cfg.topk_group, cfg.num_experts_per_tok,
        cfg.norm_topk_prob, cfg.routed_scaling_factor,
    )
    torch.testing.assert_close(
        _dense_combine(got_ids, got_w, cfg.n_routed_experts),
        _dense_combine(exp_ids, exp_w, cfg.n_routed_experts),
        rtol=1e-5, atol=1e-5,
    )


def test_moe_gate_group_limited_routing():
    """With n_group=2/topk_group=1, every selected expert must come from the
    single kept group — the crux of group-limited routing."""
    torch.manual_seed(1)
    n_experts, n_group, topk_group, top_k = 8, 2, 1, 2
    experts_per_group = n_experts // n_group
    gate = KimiMoEGate(
        hidden_size=16, n_routed_experts=n_experts, num_experts_per_tok=top_k,
        n_group=n_group, topk_group=topk_group, routed_scaling_factor=1.0,
    ).to(DEVICE)
    gate.weight.data.copy_(torch.randn(n_experts, 16, device=DEVICE))
    gate.e_score_correction_bias.data.copy_(torch.randn(n_experts, device=DEVICE))

    h = torch.randn(20, 16, device=DEVICE)
    _, ids = gate(h)

    # Each token's chosen experts share one group index.
    groups = ids // experts_per_group
    assert (groups == groups[:, :1]).all(), "experts crossed group boundary"


# --------------------------------------------------------------------------
# Fused expert dispatch (trivial fixed router)
# --------------------------------------------------------------------------

def test_expert_dispatch_matches_naive():
    torch.manual_seed(2)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    T, H, I, E = 5, cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts

    h = torch.randn(T, H, device=DEVICE, dtype=dtype) * 0.1
    gate_up = torch.randn(E, 2 * I, H, device=DEVICE, dtype=dtype) * 0.05
    down = torch.randn(E, H, I, device=DEVICE, dtype=dtype) * 0.05
    # Trivial fixed router: every token -> experts {0, 1}, fixed weights.
    ids = torch.tensor([[0, 1]] * T, device=DEVICE)
    weights = torch.full((T, 2), 0.5, device=DEVICE, dtype=dtype)

    got = _dispatch(h, gate_up, down, E, ids, weights)
    expected = _ref_routed_experts(h, gate_up, down, weights, ids)
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)


# --------------------------------------------------------------------------
# Full MoE block (routed + ungated shared)
# --------------------------------------------------------------------------

def test_moe_block_matches_reference():
    torch.manual_seed(3)
    cfg = KimiK2Config.reduced()
    dtype = torch.bfloat16
    T, H, I, E = 7, cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_routed_experts
    shared_inter = cfg.moe_intermediate_size * cfg.n_shared_experts

    block = build_moe_block(cfg).to(device=DEVICE, dtype=dtype)

    gate_w = torch.randn(E, H, device=DEVICE)  # fp32 router
    bias = torch.randn(E, device=DEVICE)
    expert_gate_up = torch.randn(E, 2 * I, H, device=DEVICE, dtype=dtype) * 0.05
    expert_down = torch.randn(E, H, I, device=DEVICE, dtype=dtype) * 0.05
    sh_gate = torch.randn(shared_inter, H, device=DEVICE, dtype=dtype) * 0.05
    sh_up = torch.randn(shared_inter, H, device=DEVICE, dtype=dtype) * 0.05
    sh_down = torch.randn(H, shared_inter, device=DEVICE, dtype=dtype) * 0.05

    # Keep the router in fp32 (deterministic selection); load fused expert +
    # shared weights.
    block.gate.weight.data = gate_w
    block.gate.e_score_correction_bias.data = bias
    block.experts.gate_up_proj.data.copy_(expert_gate_up)
    block.experts.down_proj.data.copy_(expert_down)
    block.shared_expert.gate_up_proj.weight_loader(
        block.shared_expert.gate_up_proj.weight, sh_gate, loaded_shard_id=0)
    block.shared_expert.gate_up_proj.weight_loader(
        block.shared_expert.gate_up_proj.weight, sh_up, loaded_shard_id=1)
    block.shared_expert.down_proj.weight_loader(
        block.shared_expert.down_proj.weight, sh_down)

    h = torch.randn(T, H, device=DEVICE, dtype=dtype) * 0.1
    got = block(h)

    logits = F.linear(h.float(), gate_w.float())
    weights, ids = _ref_grouped_topk(
        logits, bias, cfg.n_group, cfg.topk_group, cfg.num_experts_per_tok,
        cfg.norm_topk_prob, cfg.routed_scaling_factor,
    )
    routed = _ref_routed_experts(
        h, expert_gate_up, expert_down, weights.to(dtype), ids)
    shared = _ref_swiglu(h, sh_gate, sh_up, sh_down)
    expected = routed + shared
    torch.testing.assert_close(got, expected, rtol=2e-2, atol=2e-2)
