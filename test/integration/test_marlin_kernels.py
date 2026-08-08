"""GPU golden tests for the vendored Marlin W4A16 kernels."""
import pytest
import torch

from mstar.model.components.quantization import W4A16Data
from mstar.model.kimi_k2_7._testing import fake_quantize_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 0),
    reason="Marlin W4A16 kernels need a CUDA GPU with sm80+ (Ampere/Hopper)",
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


def _rel_l2(a, b):
    return ((a - b).float().norm() / b.float().norm()).item()


def test_marlin_builds_and_registers():
    from mstar.utils.marlin import is_marlin_available

    assert is_marlin_available(), "the vendored Marlin CUDA extension failed to build"
    assert hasattr(torch.ops._mstar_marlin_C, "gptq_marlin_repack")
    assert hasattr(torch.ops._mstar_marlin_C, "moe_wna16_marlin_gemm")


def test_repack_shape_and_deterministic():
    from mstar.utils.marlin import is_marlin_available, ops

    assert is_marlin_available()
    size_k, size_n = 256, 128  # k%16==0, n%64==0
    b = torch.randint(
        -(2**31), 2**31 - 1, (size_k // PACK_FACTOR, size_n), dtype=torch.int32, device=DEVICE
    )
    out = ops.gptq_marlin_repack(b, size_k, size_n, num_bits=4)
    assert out.shape == (size_k // 16, size_n * 16 // PACK_FACTOR)
    assert torch.equal(out, ops.gptq_marlin_repack(b, size_k, size_n, num_bits=4))


@pytest.mark.parametrize("num_tokens", [8, 3, 1])  # M > E, M <= E, single-token decode
def test_marlin_moe_matches_bf16_and_triton(num_tokens):
    from mstar.model.components.quantization import MarlinMoEMethod
    from mstar.utils.fused_moe.runner import fused_experts

    torch.manual_seed(0)
    # Marlin-legal shapes: hidden % 128, moe_inter % 128, 2*inter % 64, hidden % 64.
    E, H, I, top_k = 4, 256, 256, 2
    w1 = (torch.randn(E, 2 * I, H, device=DEVICE) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEVICE) * 0.3).to(torch.bfloat16)
    w1_packed, w1_scale, w1_deq = _quantize_stack(w1)
    w2_packed, w2_scale, w2_deq = _quantize_stack(w2)

    x = (torch.randn(num_tokens, H, device=DEVICE) * 0.5).to(torch.bfloat16)
    topk_weights, topk_ids = _random_topk(num_tokens, E, top_k)

    method = MarlinMoEMethod(num_bits=4, group_size=GROUP_SIZE)
    method.prepare(w1_packed, w1_scale, w2_packed, w2_scale, torch.device(DEVICE))
    out_marlin = method.apply(x, topk_weights, topk_ids)

    out_bf16 = fused_experts(x, w1_deq, w2_deq, topk_weights, topk_ids)  # ground truth
    out_triton = fused_experts(  # same packed nibbles, Triton W4A16 kernel
        x, w1_packed, w2_packed, topk_weights, topk_ids,
        quant=W4A16Data(w1_scale=w1_scale, w2_scale=w2_scale, group_size=GROUP_SIZE),
    )

    assert out_marlin.shape == (num_tokens, H) and out_marlin.dtype == torch.bfloat16
    cos = torch.nn.functional.cosine_similarity(
        out_marlin.flatten().float(), out_bf16.flatten().float(), dim=0
    ).item()
    assert cos > 0.999, f"cosine vs bf16 too low: {cos}"
    assert _rel_l2(out_marlin, out_bf16) < 0.02, "relative-L2 vs bf16 too high"
    assert _rel_l2(out_marlin, out_triton) < 0.02, "relative-L2 vs Triton W4A16 too high"


def test_marlin_moe_reduce_results_false_shape():
    from mstar.model.components.quantization import MarlinMoEMethod

    torch.manual_seed(1)
    E, H, I, top_k, num_tokens = 4, 256, 256, 2, 6
    w1 = (torch.randn(E, 2 * I, H, device=DEVICE) * 0.3).to(torch.bfloat16)
    w2 = (torch.randn(E, H, I, device=DEVICE) * 0.3).to(torch.bfloat16)
    w1_packed, w1_scale, _ = _quantize_stack(w1)
    w2_packed, w2_scale, _ = _quantize_stack(w2)
    x = (torch.randn(num_tokens, H, device=DEVICE) * 0.5).to(torch.bfloat16)
    topk_weights, topk_ids = _random_topk(num_tokens, E, top_k)

    method = MarlinMoEMethod(num_bits=4, group_size=GROUP_SIZE)
    method.prepare(w1_packed, w1_scale, w2_packed, w2_scale, torch.device(DEVICE))
    got = method.apply(x, topk_weights, topk_ids, reduce_results=False)
    assert got.shape == (num_tokens, top_k, H)
