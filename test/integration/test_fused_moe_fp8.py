"""GPU golden for the block-scaled FP8 fused-MoE path (GLM-5.2 experts)."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused FP8 W8A8 MoE kernel golden needs a GPU",
)

DEVICE = "cuda"
FP8 = torch.float8_e4m3fn
BLOCK = (128, 128)  # the kernel requires K tiles == quant groups == 128
HIDDEN, INTER, NUM_EXPERTS, TOP_K = 256, 128, 4, 2


def test_swiglu_clamp_opt_in():
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    torch.manual_seed(4)
    gateup = (torch.randn(16, 512, device=DEVICE) * 40).to(torch.bfloat16)
    gate, up = gateup.chunk(2, dim=-1)
    plain = torch.empty(16, 256, device=DEVICE, dtype=torch.bfloat16)
    clamped = torch.empty_like(plain)
    act_and_mul_triton(gateup, plain)
    act_and_mul_triton(gateup, clamped, swiglu_limit=10.0)

    plain_ref = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    clamp_ref = (
        F.silu(gate.float().clamp(max=10.0)) * up.float().clamp(-10.0, 10.0)
    ).to(torch.bfloat16)
    torch.testing.assert_close(plain, plain_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(clamped, clamp_ref, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(plain, clamped, rtol=2e-2, atol=2e-2)


def _block_quant(weight: torch.Tensor, block: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(out, in) fp32 -> (e4m3 weight, fp32 scale_inv) per block; dequant = w * scale."""
    out_f, in_f = weight.shape
    bo, bi = block
    assert out_f % bo == 0 and in_f % bi == 0
    blocks = weight.float().view(out_f // bo, bo, in_f // bi, bi)
    scale_inv = blocks.abs().amax(dim=(1, 3)) / 448.0  # e4m3 max normal
    scale_inv = torch.where(scale_inv == 0, torch.ones_like(scale_inv), scale_inv)
    scale_bc = scale_inv.repeat_interleave(bo, dim=0).repeat_interleave(bi, dim=1)
    return (weight.float() / scale_bc).to(FP8), scale_inv


def _dequant(w8: torch.Tensor, scale_inv: torch.Tensor, block: tuple[int, int], out_dtype) -> torch.Tensor:
    bo, bi = block
    scale = scale_inv.float().repeat_interleave(bo, dim=0).repeat_interleave(bi, dim=1)
    return (w8.view(FP8).float() * scale).to(out_dtype)


def _build_experts(seed: int):
    """Random block-quantized experts in the model's parameter layout:
    ``(E, 2*inter, hidden)`` gate+up bytes, ``(E, hidden, inter)`` down bytes,
    fp32 ``scale_inv`` per weight, all as uint8 views like the checkpoint."""
    torch.manual_seed(seed)
    E, H, I = NUM_EXPERTS, HIDDEN, INTER
    w1 = torch.empty(E, 2 * I, H, dtype=FP8)
    w2 = torch.empty(E, H, I, dtype=FP8)
    w1_s = torch.empty(E, 2 * I // BLOCK[0], H // BLOCK[1])
    w2_s = torch.empty(E, H // BLOCK[0], I // BLOCK[1])
    for e in range(E):
        w1[e], w1_s[e] = _block_quant(torch.randn(2 * I, H) * 0.1, BLOCK)
        w2[e], w2_s[e] = _block_quant(torch.randn(H, I) * 0.1, BLOCK)
    return (
        w1.view(torch.uint8).to(DEVICE),
        w2.view(torch.uint8).to(DEVICE),
        w1_s.to(DEVICE),
        w2_s.to(DEVICE),
    )


def _route(x: torch.Tensor, seed: int):
    torch.manual_seed(seed)
    logits = torch.randn(x.shape[0], NUM_EXPERTS, device=x.device)
    topk_weights, topk_ids = torch.topk(logits.sigmoid(), TOP_K, dim=-1)
    topk_weights = topk_weights / topk_weights.sum(-1, keepdim=True)
    return topk_weights.to(x.dtype), topk_ids


def _reference(x, w1_fp8, w2_fp8, w1_s, w2_s, topk_weights, topk_ids):
    """Per-expert loop on bf16-dequantized weights (the model's fp8 reference dispatch)."""
    final = torch.zeros_like(x)
    for e in range(NUM_EXPERTS):
        token_idx, slot = torch.where(topk_ids == e)
        if token_idx.numel() == 0:
            continue
        gate_up_w = _dequant(w1_fp8[e], w1_s[e], BLOCK, x.dtype)
        down_w = _dequant(w2_fp8[e], w2_s[e], BLOCK, x.dtype)
        gate, up = torch.mm(x[token_idx], gate_up_w.T).chunk(2, dim=-1)
        out = torch.mm(F.silu(gate) * up, down_w.T)
        final.index_add_(0, token_idx, (out * topk_weights[token_idx, slot, None]).to(final.dtype))
    return final


@pytest.mark.parametrize("num_tokens", [8, 3, 32])  # M > E, M <= E, M > 16 config branches
def test_fp8_fused_matches_reference_dispatch(num_tokens):
    from mstar.utils.fused_moe.runner import fused_experts_fp8

    w1, w2, w1_s, w2_s = _build_experts(seed=0)
    x = (torch.randn(num_tokens, HIDDEN, device=DEVICE) * 0.2).to(torch.bfloat16)
    topk_weights, topk_ids = _route(x, seed=0)

    ref = _reference(x, w1, w2, w1_s, w2_s, topk_weights, topk_ids)
    got = fused_experts_fp8(x, w1, w2, w1_s, w2_s, topk_weights, topk_ids, block_size=BLOCK)

    assert got.shape == (num_tokens, HIDDEN)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_fp8_reduce_results_false_shape():
    from mstar.utils.fused_moe.runner import fused_experts_fp8

    w1, w2, w1_s, w2_s = _build_experts(seed=1)
    num_tokens = 6
    x = (torch.randn(num_tokens, HIDDEN, device=DEVICE) * 0.2).to(torch.bfloat16)
    topk_weights, topk_ids = _route(x, seed=1)

    # e4m3 tensors are accepted directly as well as their uint8 byte views.
    args = (x, w1.view(FP8), w2.view(FP8), w1_s, w2_s, topk_weights, topk_ids)
    per_slot = fused_experts_fp8(*args, block_size=BLOCK, reduce_results=False)
    reduced = fused_experts_fp8(*args, block_size=BLOCK)

    assert per_slot.shape == (num_tokens, TOP_K, HIDDEN)
    # Routed weights are folded into GEMM-2, so summing the slots must match
    # the reduced path up to bf16-sum vs fp32-reduce rounding.
    torch.testing.assert_close(per_slot.sum(dim=1), reduced, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_tokens", [37, 1100])
def test_fp8_compiles_whole_with_symbolic_tokens(num_tokens):
    """One opaque op: fullgraph compile with a dynamic token count equals eager."""
    from mstar.utils.fused_moe.runner import fused_experts_fp8

    w1, w2, w1_s, w2_s = _build_experts(seed=2)

    def moe(x, weights, ids):
        return fused_experts_fp8(x, w1, w2, w1_s, w2_s, weights, ids, block_size=BLOCK)

    compiled = torch.compile(moe, fullgraph=True, dynamic=True)
    x = (torch.randn(num_tokens, HIDDEN, device=DEVICE) * 0.2).to(torch.bfloat16)
    topk_weights, topk_ids = _route(x, seed=2)
    torch.testing.assert_close(compiled(x, topk_weights, topk_ids), moe(x, topk_weights, topk_ids),
                               rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [3, 40, 600])
def test_int64_ids_align_as_int32_ids_do(num_tokens):
    """The router's int64 ids reach the alignment without a cast kernel."""
    from mstar.utils.fused_moe.align import moe_align_block_size

    torch.manual_seed(6)
    ids = torch.stack([torch.randperm(128, device=DEVICE)[:8] for _ in range(num_tokens)])
    got = moe_align_block_size(ids, 16, 128)
    want = moe_align_block_size(ids.to(torch.int32), 16, 128)
    assert torch.equal(got[1], want[1]) and torch.equal(got[2], want[2])
    # slot order within an expert is atomics-ordered: compare (expert, slot) pairs as sets
    n = int(want[2])

    def pairs(sorted_ids, expert_ids):
        expert = expert_ids[torch.arange(n, device=DEVICE) // 16].to(torch.int64)
        return sorted((expert * 10**6 + sorted_ids[:n].to(torch.int64)).tolist())

    assert pairs(got[0], got[1]) == pairs(want[0], want[1])


@pytest.mark.parametrize("num_tokens,num_experts", [(1, 128), (8, 128), (32, 128), (511, 128), (32, 256),
                                                    (520, 128)])
def test_align_one_and_two_kernel_paths_agree_with_the_torch_reference(num_tokens, num_experts):
    """Below 4096 slots the alignment runs as one kernel, above as two; both match the torch
    fallback (as sets within each expert's span: the scatter is atomics-ordered)."""
    from mstar.utils.fused_moe import align

    torch.manual_seed(num_tokens)
    ids = torch.stack([torch.randperm(num_experts, device=DEVICE)[:8] for _ in range(num_tokens)])
    ids[0, 0] = num_experts  # a skipped slot (the EP sentinel)
    got = align.moe_align_block_size(ids, 16, num_experts)
    n_pad = ids.numel() + num_experts * 15
    want = (torch.empty(n_pad, dtype=torch.int32, device=DEVICE),
            torch.empty(-(-n_pad // 16), dtype=torch.int32, device=DEVICE),
            torch.empty(1, dtype=torch.int32, device=DEVICE))
    align._moe_align_block_size_torch(ids, 16, num_experts, *want)
    assert torch.equal(got[1], want[1]) and torch.equal(got[2], want[2])
    n = int(want[2])
    assert torch.equal(got[0][n:], want[0][n:])

    def pairs(sorted_ids, expert_ids):
        expert = expert_ids[torch.arange(n, device=DEVICE) // 16].to(torch.int64)
        return sorted((expert * 10**6 + sorted_ids[:n].to(torch.int64)).tolist())

    assert pairs(got[0], got[1]) == pairs(want[0], want[1])
