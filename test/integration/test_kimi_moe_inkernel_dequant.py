"""GPU kernel golden: W4A16 in-kernel INT4 dequant vs the bf16 fused-expert GEMM.

The in-kernel dequant path ships a SEPARATE ``fused_moe_kernel_w4a16`` that keeps the routed
experts packed in VRAM and dequantizes each K tile in registers before the dot.
Its correctness invariant is exact: the nibble ``(q - 8) * scale`` cast to bf16 is
*the same value* the bf16 path feeds to ``tl.dot`` after a pre-dequant, and with
the same tile config the two accumulate in the same order — so the packed path
must match the bf16 path on the SAME dequantized weights to a tight tolerance.

This is the cheapest level that catches a kernel bug (packed-K stride, nibble
shifter, group-scale index, the top-nibble sign case) without a full model:

  1. random bf16 experts ``w1 (E, 2I, H)`` / ``w2 (E, H, I)``;
  2. ``fake_quantize_weight`` each expert to ``(packed, bf16 scale, deq_bf16)``,
     with ``scale_dtype=bfloat16`` so the packed-param scale and the bf16-path
     weight dequantize from the identical scale;
  3. assert ``fused_experts(x, w1_packed, w2_packed, w1_scale=, w2_scale=, ...)``
     == ``fused_experts(x, w1_deq, w2_deq)`` (the bf16 path).

Includes an expert whose packing sets container bit 31 (top nibble >= 8), proving
the arithmetic-shift + ``& 0xF`` mask recovers it.

Run:  pytest test/integration/test_kimi_moe_inkernel_dequant.py -v
"""
import pytest
import torch

from mstar.model.kimi_k2_7._testing import fake_quantize_weight
from mstar.model.kimi_k2_7.quantization import unpack_int32

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="W4A16 fused-expert kernel golden needs a GPU",
)

DEVICE = "cuda"
GROUP_SIZE = 32
PACK_FACTOR = 8


def _quantize_stack(weight):
    """Fake-quantize a stacked ``(E, N, K)`` weight, returning packed/scale/deq.

    Each expert is quantized independently (matching a per-Linear checkpoint);
    the bf16 scale is what a real compressed-tensors checkpoint stores, so the
    returned ``deq`` is bit-for-bit what the packed param dequantizes to.
    """
    E, N, K = weight.shape
    packed = torch.empty((E, N, K // PACK_FACTOR), dtype=torch.int32, device=DEVICE)
    scale = torch.empty((E, N, K // GROUP_SIZE), dtype=torch.bfloat16, device=DEVICE)
    deq = torch.empty((E, N, K), dtype=torch.bfloat16, device=DEVICE)
    for e in range(E):
        p, s, d = fake_quantize_weight(
            weight[e], num_bits=4, group_size=GROUP_SIZE, symmetric=True,
            scale_dtype=torch.bfloat16,
        )
        packed[e], scale[e], deq[e] = p.to(DEVICE), s.to(DEVICE), d.to(DEVICE)
    return packed, scale, deq


def _random_topk(num_tokens, E, top_k):
    logits = torch.randn(num_tokens, E, device=DEVICE)
    weights, ids = torch.topk(logits.softmax(-1), top_k, dim=-1)
    weights = weights / weights.sum(-1, keepdim=True)
    return weights.to(torch.bfloat16), ids


@pytest.mark.parametrize("num_tokens", [8, 3])  # M > E and M <= E branches
def test_w4a16_matches_bf16_on_same_dequant(num_tokens):
    from mstar.utils.fused_moe.runner import fused_experts

    torch.manual_seed(0)
    E, H, I, top_k = 4, 128, 64, 2
    # Slightly wide init so per-group amax spans the full nibble range and some
    # top nibbles land >= 8 (container bit 31 set) — the sign-mask path.
    w1 = (torch.randn(E, 2 * I, H, device=DEVICE) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEVICE) * 0.3).to(torch.bfloat16)

    w1_packed, w1_scale, w1_deq = _quantize_stack(w1)
    w2_packed, w2_scale, w2_deq = _quantize_stack(w2)

    # Guard: the packing really exercises the negative-container / top-nibble>=8
    # case (else the sign-extension mask would be untested).
    assert (w1_packed < 0).any(), "no int32 with bit 31 set — top-nibble path untested"
    top_nibbles = unpack_int32(w1_packed.cpu(), num_bits=4)[..., PACK_FACTOR - 1 :: PACK_FACTOR]
    assert (top_nibbles >= 8).any()

    x = (torch.randn(num_tokens, H, device=DEVICE) * 0.5).to(torch.bfloat16)
    topk_weights, topk_ids = _random_topk(num_tokens, E, top_k)

    out_quant = fused_experts(
        x, w1_packed, w2_packed, topk_weights, topk_ids,
        w1_scale=w1_scale, w2_scale=w2_scale, group_size=GROUP_SIZE, pack_factor=PACK_FACTOR,
    )
    out_bf16 = fused_experts(x, w1_deq, w2_deq, topk_weights, topk_ids)

    assert out_quant.shape == (num_tokens, H)
    assert out_quant.dtype == torch.bfloat16
    torch.testing.assert_close(out_quant, out_bf16, rtol=1e-2, atol=1e-2)


def test_w4a16_reduce_results_false_shape():
    """``reduce_results=False`` returns the per-slot (tokens, top_k, hidden) tensor
    the TP path all-reduces before the top-k sum — exercise it on the packed path."""
    from mstar.utils.fused_moe.runner import fused_experts

    torch.manual_seed(1)
    E, H, I, top_k, num_tokens = 4, 128, 64, 2, 6
    w1 = (torch.randn(E, 2 * I, H, device=DEVICE) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEVICE) * 0.3).to(torch.bfloat16)
    w1_packed, w1_scale, w1_deq = _quantize_stack(w1)
    w2_packed, w2_scale, w2_deq = _quantize_stack(w2)
    x = (torch.randn(num_tokens, H, device=DEVICE) * 0.5).to(torch.bfloat16)
    topk_weights, topk_ids = _random_topk(num_tokens, E, top_k)

    got = fused_experts(
        x, w1_packed, w2_packed, topk_weights, topk_ids,
        w1_scale=w1_scale, w2_scale=w2_scale, group_size=GROUP_SIZE, pack_factor=PACK_FACTOR,
        reduce_results=False,
    )
    exp = fused_experts(x, w1_deq, w2_deq, topk_weights, topk_ids, reduce_results=False)
    assert got.shape == (num_tokens, top_k, H)
    torch.testing.assert_close(got, exp, rtol=1e-2, atol=1e-2)
