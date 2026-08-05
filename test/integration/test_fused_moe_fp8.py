"""GPU golden for the block-scaled FP8 fused-MoE path (GLM-5.2 experts).

``fused_experts_fp8`` and ``Glm52SparseMoeBlock._dispatch_fp8_reference``
dequantize the same e4m3 bytes with the same ``weight_scale_inv`` blocks, so
the only divergence is the fused path's on-the-fly per-token-group fp8
activation quant (plus fp8-dot vs bf16-mm rounding) — elementwise closeness
at ~2e-2 is the bar, not bitwise equality.
"""
import pytest
import torch

from mstar.model.glm52._testing import fake_quantize_fp8_block
from mstar.model.glm52.config import Glm52ModelConfig
from mstar.model.glm52.quantization import Fp8BlockQuantConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused FP8 W8A8 MoE kernel golden needs a GPU",
)

DEVICE = "cuda"
BLOCK = (128, 128)  # the kernel requires K tiles == quant groups == 128


def _reduced_fp8_128_config() -> Glm52ModelConfig:
    """``reduced_fp8`` defaults to (16, 16) scale blocks; the fused kernel
    needs the real checkpoint's (128, 128), so pick 128-divisible MoE dims
    by hand on top of the reduced base."""
    cfg = Glm52ModelConfig.reduced()
    cfg.hidden_size = 256
    cfg.moe_intermediate_size = 128
    cfg.n_routed_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=BLOCK)
    return cfg


def _build_block(seed: int):
    """Reduced fp8-resident MoE block with random e4m3 experts on DEVICE.

    ``components.moe`` imports triton transitively, so (like the other GPU
    goldens) it is imported here rather than at module level to keep
    collection clean on CUDA-less machines.
    """
    from mstar.model.glm52.components.moe import Glm52SparseMoeBlock

    torch.manual_seed(seed)
    cfg = _reduced_fp8_128_config()
    block = Glm52SparseMoeBlock(cfg)
    assert block.fp8_experts and block.block_size == BLOCK

    block.gate.weight.data.normal_(0, 0.1)
    block.gate.e_score_correction_bias.data.normal_(0, 0.5)

    shard = cfg.moe_intermediate_size
    srow = shard // BLOCK[0]
    exp = block.experts
    for e in range(cfg.n_routed_experts):
        g = torch.randn(shard, cfg.hidden_size) * 0.1
        u = torch.randn(shard, cfg.hidden_size) * 0.1
        d = torch.randn(cfg.hidden_size, shard) * 0.1
        g8, gs, _ = fake_quantize_fp8_block(g, BLOCK)
        u8, us, _ = fake_quantize_fp8_block(u, BLOCK)
        d8, ds, _ = fake_quantize_fp8_block(d, BLOCK)
        exp.gate_up_proj_fp8.data[e, :shard] = g8.view(torch.uint8)
        exp.gate_up_proj_fp8.data[e, shard:] = u8.view(torch.uint8)
        exp.gate_up_proj_scale_inv.data[e, :srow] = gs
        exp.gate_up_proj_scale_inv.data[e, srow:] = us
        exp.down_proj_fp8.data[e] = d8.view(torch.uint8)
        exp.down_proj_scale_inv.data[e] = ds
    # Device-only move keeps the uint8 bytes and fp32 scales untouched.
    return block.to(DEVICE)


def _route(block, x):
    topk_weights, topk_ids = block.gate(x)
    return topk_weights.to(x.dtype), topk_ids


@pytest.mark.parametrize("num_tokens", [8, 3])  # M > E and M <= E config branches
def test_fp8_fused_matches_reference_dispatch(num_tokens):
    from mstar.utils.fused_moe.runner import fused_experts_fp8

    block = _build_block(seed=0)
    x = (torch.randn(num_tokens, block.hidden_size, device=DEVICE) * 0.2).to(torch.bfloat16)
    topk_weights, topk_ids = _route(block, x)

    ref = block._dispatch_fp8_reference(x, topk_weights, topk_ids)
    got = fused_experts_fp8(
        x,
        block.experts.gate_up_proj_fp8,
        block.experts.down_proj_fp8,
        block.experts.gate_up_proj_scale_inv,
        block.experts.down_proj_scale_inv,
        topk_weights,
        topk_ids,
        block_size=block.block_size,
    )

    assert got.shape == (num_tokens, block.hidden_size)
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, ref, rtol=2e-2, atol=2e-2)


def test_fp8_reduce_results_false_shape():
    from mstar.utils.fused_moe.runner import fused_experts_fp8

    block = _build_block(seed=1)
    num_tokens, top_k = 6, 2
    x = (torch.randn(num_tokens, block.hidden_size, device=DEVICE) * 0.2).to(torch.bfloat16)
    topk_weights, topk_ids = _route(block, x)

    args = (
        x,
        block.experts.gate_up_proj_fp8,
        block.experts.down_proj_fp8,
        block.experts.gate_up_proj_scale_inv,
        block.experts.down_proj_scale_inv,
        topk_weights,
        topk_ids,
    )
    per_slot = fused_experts_fp8(*args, block_size=block.block_size, reduce_results=False)
    reduced = fused_experts_fp8(*args, block_size=block.block_size)

    assert per_slot.shape == (num_tokens, top_k, block.hidden_size)
    # Routed weights are folded into GEMM-2, so summing the slots must match
    # the reduced path up to bf16-sum vs fp32-reduce rounding.
    torch.testing.assert_close(per_slot.sum(dim=1), reduced, rtol=1e-2, atol=1e-2)


def test_per_token_group_quant_fp8_roundtrip():
    from mstar.utils.fused_moe.kernels import per_token_group_quant_fp8

    torch.manual_seed(2)
    x = torch.randn(64, 256, device=DEVICE).to(torch.bfloat16)
    x_q, x_s = per_token_group_quant_fp8(x, 128)

    assert x_q.dtype == torch.float8_e4m3fn
    assert x_s.dtype == torch.float32 and x_s.shape == (64, 2)
    deq = x_q.to(torch.float32) * x_s.repeat_interleave(128, dim=1)
    # e4m3's 3-bit mantissa bounds the relative error at 2^-4 once the group
    # scale is divided out; atol covers the subnormal tail near zero.
    torch.testing.assert_close(deq, x.to(torch.float32), rtol=0.07, atol=1e-3)
