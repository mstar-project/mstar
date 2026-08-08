import pytest
import torch

from mstar.model.components.quantization import W4A16Data, unpack_int32
from mstar.model.kimi_k2_7._testing import fake_quantize_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="W4A16 fused-expert kernel golden needs a GPU",
)

DEVICE = "cuda"
GROUP_SIZE = 32
PACK_FACTOR = 8


def _quantize_stack(weight):
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
    # Wide init exercises top-nibble sign masking.
    w1 = (torch.randn(E, 2 * I, H, device=DEVICE) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEVICE) * 0.3).to(torch.bfloat16)

    w1_packed, w1_scale, w1_deq = _quantize_stack(w1)
    w2_packed, w2_scale, w2_deq = _quantize_stack(w2)

    # Guard that the negative-container path is actually covered.
    assert (w1_packed < 0).any(), "no int32 with bit 31 set — top-nibble path untested"
    top_nibbles = unpack_int32(w1_packed.cpu(), num_bits=4)[..., PACK_FACTOR - 1 :: PACK_FACTOR]
    assert (top_nibbles >= 8).any()

    x = (torch.randn(num_tokens, H, device=DEVICE) * 0.5).to(torch.bfloat16)
    topk_weights, topk_ids = _random_topk(num_tokens, E, top_k)

    out_quant = fused_experts(
        x, w1_packed, w2_packed, topk_weights, topk_ids,
        quant=W4A16Data(w1_scale=w1_scale, w2_scale=w2_scale, group_size=GROUP_SIZE),
    )
    out_bf16 = fused_experts(x, w1_deq, w2_deq, topk_weights, topk_ids)

    assert out_quant.shape == (num_tokens, H)
    assert out_quant.dtype == torch.bfloat16
    torch.testing.assert_close(out_quant, out_bf16, rtol=1e-2, atol=1e-2)


def test_w4a16_reduce_results_false_shape():
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
        quant=W4A16Data(w1_scale=w1_scale, w2_scale=w2_scale, group_size=GROUP_SIZE),
        reduce_results=False,
    )
    exp = fused_experts(x, w1_deq, w2_deq, topk_weights, topk_ids, reduce_results=False)
    assert got.shape == (num_tokens, top_k, H)
    torch.testing.assert_close(got, exp, rtol=1e-2, atol=1e-2)
