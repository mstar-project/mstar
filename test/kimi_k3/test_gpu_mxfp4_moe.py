"""GPU tests for the MXFP4 Triton MoE path (skipped without CUDA): the SiTU activation
kernel, the bf16 grouped GEMM with SiTU, and the packed-weight kernel vs. the bf16 kernel on
the dequantized weights (they must agree to bf16 rounding) and vs. the torch reference."""
import pytest
import torch

from mstar.model.kimi_k3.reference.moe import routed_experts_loop
from mstar.model.kimi_k3.reference.mxfp4 import dequant_mxfp4, quant_mxfp4
from mstar.model.kimi_k3.reference.router import noaux_tc_route
from mstar.model.kimi_k3.reference.situ import situ_and_mul

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
DEV = torch.device("cuda")


@cuda
def test_situ_kernel_matches_reference():
    from mstar.utils.fused_moe.mxfp4 import situ_and_mul_triton
    torch.manual_seed(0)
    x = (torch.randn(37, 2 * 96, device=DEV) * 6).to(torch.bfloat16)
    out = torch.empty(37, 96, device=DEV, dtype=torch.bfloat16)
    situ_and_mul_triton(x, out, 4.0, 25.0)
    torch.testing.assert_close(out.float(), situ_and_mul(x, 4.0, 25.0).float(), rtol=2e-2, atol=2e-2)


def _quantize_experts(w13, w2):
    e = w13.shape[0]
    p13, s13, p2, s2 = [], [], [], []
    for i in range(e):
        p, s = quant_mxfp4(w13[i].float().cpu())
        p13.append(p)
        s13.append(s)
        p, s = quant_mxfp4(w2[i].float().cpu())
        p2.append(p)
        s2.append(s)
    return (torch.stack(p13).to(DEV), torch.stack(s13).to(DEV), torch.stack(p2).to(DEV), torch.stack(s2).to(DEV))


@cuda
@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
def test_mxfp4_kernel_matches_bf16_and_reference(tokens):
    from mstar.utils.fused_moe.mxfp4 import fused_experts_bf16_situ, fused_experts_mxfp4
    torch.manual_seed(1)
    E, K, inter, top_k = 16, 256, 128, 4
    w13 = (torch.randn(E, 2 * inter, K, device=DEV) * 0.2).to(torch.bfloat16)
    w2 = (torch.randn(E, K, inter, device=DEV) * 0.2).to(torch.bfloat16)
    p13, s13, p2, s2 = _quantize_experts(w13, w2)
    # the kernels' ground truth: the dequantized weights (exact in bf16)
    dq13 = torch.stack([dequant_mxfp4(p13[i], s13[i]) for i in range(E)])
    dq2 = torch.stack([dequant_mxfp4(p2[i], s2[i]) for i in range(E)])
    x = torch.randn(tokens, K, device=DEV, dtype=torch.bfloat16)
    gate_w = torch.randn(E, K, device=DEV) * 0.1
    bias = torch.randn(E, device=DEV) * 0.01
    topk_idx, topk_w = noaux_tc_route(x, gate_w, bias, top_k)
    out_q = fused_experts_mxfp4(x, p13, s13, p2, s2, topk_w, topk_idx, 4.0, 25.0)
    out_bf = fused_experts_bf16_situ(x, dq13, dq2, topk_w, topk_idx, 4.0, 25.0)
    ref = routed_experts_loop(x.float(), topk_idx, topk_w, dq13.float(), dq2.float(), 4.0, 25.0)
    torch.testing.assert_close(out_q.float(), out_bf.float(), rtol=3e-2, atol=3e-2)
    # fp32 reference; the kernels round the SiTU activation to bf16 between the GEMMs
    torch.testing.assert_close(out_q.float(), ref, rtol=2e-2, atol=2e-1)
    torch.testing.assert_close(out_bf.float(), ref, rtol=2e-2, atol=2e-1)


@cuda
@pytest.mark.parametrize("m,d", [(1, 1024), (4, 1024), (8, 7168), (0, 256)])
def test_attn_res_kernel_matches_reference(m, d):
    from mstar.model.kimi_k3.components.attn_res_kernel import attn_res_read_triton
    from mstar.model.kimi_k3.reference.attn_res import attn_res_read, attn_res_read_and_norm
    torch.manual_seed(0)
    t = 37
    prefix = torch.randn(t, d, device=DEV, dtype=torch.bfloat16)
    blocks = torch.randn(t, m, d, device=DEV, dtype=torch.bfloat16)
    w = torch.randn(d, device=DEV) * 0.05
    ref = attn_res_read(prefix, blocks, w, 1e-5)
    out = attn_res_read_triton(prefix, blocks, w, 1e-5)
    torch.testing.assert_close(out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    nw = torch.rand(d, device=DEV, dtype=torch.bfloat16) + 0.5
    ref2 = attn_res_read_and_norm(prefix, blocks, w, nw, 1e-5, 1e-5)
    out2 = attn_res_read_triton(prefix, blocks, w, 1e-5, out_norm_weight=nw, out_eps=1e-5)
    torch.testing.assert_close(out2.float(), ref2.float(), rtol=3e-2, atol=3e-2)


@cuda
@pytest.mark.parametrize("e,tokens", [(16, 5), (224, 100)])
def test_align_op_skips_foreign_expert_ids(e, tokens):
    """The CUDA alignment op ignores ids outside ``[0, num_experts)`` like the torch fallback
    (both the small-batch kernel, ``e <= 64`` and few assignments, and the general one)."""
    from mstar.utils.fused_moe.align import _moe_align_block_size_torch, moe_align_block_size

    torch.manual_seed(0)
    top_k, block = 4, 8
    ids = torch.randint(0, e + 1, (tokens, top_k), dtype=torch.int32, device=DEV)  # e = foreign
    ids[0] = e
    sorted_ids, expert_ids, post = moe_align_block_size(ids, block, e)
    n = sorted_ids.shape[0]
    ref_sorted = torch.empty(n, dtype=torch.int32, device=DEV)
    ref_experts = torch.empty(expert_ids.shape[0], dtype=torch.int32, device=DEV)
    ref_post = torch.empty(1, dtype=torch.int32, device=DEV)
    _moe_align_block_size_torch(ids, block, e, ref_sorted, ref_experts, ref_post)
    assert int(post) == int(ref_post)
    valid = int(post)
    # same multiset of placed assignments per block (intra-block order is free)
    for b in range(valid // block):
        got = sorted(sorted_ids[b * block:(b + 1) * block].tolist())
        exp = sorted(ref_sorted[b * block:(b + 1) * block].tolist())
        assert got == exp, b
        assert int(expert_ids[b]) == int(ref_experts[b])
    flat = ids.reshape(-1)
    placed = sorted_ids[:valid]
    placed = placed[placed < flat.numel()]
    assert bool((flat[placed] < e).all()) and placed.numel() == int((flat < e).sum())


@cuda
@pytest.mark.parametrize("world,ep", [(2, 2), (4, 2)])
def test_triton_expert_parallel_partials_sum_to_full(world, ep):
    """The in-tree Triton MXFP4 kernel in partial mode (local ids with the skipped id, zero-filled
    top-k slots): per-rank partials add up to the full routed result."""
    from test_gpu_marlin_moe import _sharded_copies, _small_moe

    moe, hidden = _small_moe()
    ex = moe.experts
    p13, s13, p2, s2 = (t.clone() for t in (ex.gate_up_packed, ex.gate_up_scale, ex.down_packed, ex.down_scale))
    ranks = _sharded_copies(moe, world, ep, p13, s13, p2, s2, backend=None)
    with torch.no_grad():
        x = torch.randn(37, hidden, device=DEV, dtype=torch.bfloat16)
        z = moe.routed_expert_down_proj(x)
        idx, w = moe.gate(x)
        ref = moe._routed(z, idx, w).float()  # Triton kernel, every expert local
        parts = [m._routed(z, idx, w) for m in ranks]
    for m, part in zip(ranks, parts, strict=True):
        none_here = (m.sharding.localize(idx) == m.sharding.invalid_id).all(1)
        assert torch.equal(part[none_here], torch.zeros_like(part[none_here]))
    total = sum(p.float() for p in parts)
    rel = (total - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()
    assert rel < 2e-2, f"rel rms {rel:.4f}"
