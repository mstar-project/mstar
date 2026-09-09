"""GPU golden for the block-scaled FP8 fused-MoE path (GLM-5.2 experts).

``fused_experts_fp8`` and the per-expert reference loop below dequantize the
same e4m3 bytes with the same ``weight_scale_inv`` blocks, so the only
divergence is the fused path's on-the-fly per-token-group fp8 activation
quant (plus fp8-dot vs bf16-mm rounding) -- elementwise closeness at ~2e-2
is the bar, not bitwise equality.

Self-contained: the quant / dequant helpers mirror the GLM-5.2 model's
``fake_quantize_fp8_block`` / ``dequantize_fp8_block_weight`` so the test
does not depend on the model package.  ``mstar.utils.fused_moe`` imports
triton, so it is imported inside the tests to keep collection clean on
CUDA-less machines.
"""
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


@pytest.mark.parametrize("num_tokens", [8, 3])  # M > E and M <= E config branches
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


def test_per_token_group_quant_fp8_roundtrip():
    from mstar.utils.fused_moe.kernels import per_token_group_quant_fp8

    torch.manual_seed(2)
    x = torch.randn(64, 256, device=DEVICE).to(torch.bfloat16)
    x_q, x_s = per_token_group_quant_fp8(x, 128)

    assert x_q.dtype == FP8
    assert x_s.dtype == torch.float32 and x_s.shape == (64, 2)
    deq = x_q.to(torch.float32) * x_s.repeat_interleave(128, dim=1)
    # e4m3's 3-bit mantissa bounds the relative error at 2^-4 once the group
    # scale is divided out; atol covers the subnormal tail near zero.
    torch.testing.assert_close(deq, x.to(torch.float32), rtol=0.07, atol=1e-3)
