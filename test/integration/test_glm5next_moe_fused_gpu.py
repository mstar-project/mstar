"""GLM-5.3's SwiGLU clamp on the fused fp8 MoE path, on CUDA: the block against its clamped
reference dispatch, and the activation kernel with the clamp on and off."""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.utils.fp8_block import FP8_DTYPE, Fp8BlockQuantConfig, dequantize_fp8_block_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused SwiGLU-clamp MoE golden needs a GPU (triton)",
)

DEVICE = "cuda"
BLOCK = (128, 128)  # the fused kernel requires K tiles == quant groups == 128


def fake_quantize_fp8_block(
    weight: torch.Tensor,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (fp8 weight, fp32 scale_inv, exact bf16 dequantized reference)."""
    out_f, in_f = weight.shape
    bo, bi = block_size
    n_bo, n_bi = -(-out_f // bo), -(-in_f // bi)

    w = weight.to(torch.float32)
    padded = torch.zeros(n_bo * bo, n_bi * bi, dtype=torch.float32)
    padded[:out_f, :in_f] = w
    blocks = padded.view(n_bo, bo, n_bi, bi)
    amax = blocks.abs().amax(dim=(1, 3))  # (n_bo, n_bi)
    scale_inv = amax / 448.0  # e4m3 max normal value
    scale_inv = torch.where(scale_inv == 0, torch.ones_like(scale_inv), scale_inv)

    scale_bc = scale_inv.repeat_interleave(bo, dim=0)[:out_f]
    scale_bc = scale_bc.repeat_interleave(bi, dim=1)[:, :in_f]
    w_fp8 = (w / scale_bc).to(FP8_DTYPE)

    dequant = dequantize_fp8_block_weight(w_fp8, scale_inv, block_size=block_size)
    return w_fp8, scale_inv, dequant


def _reduced_fp8_128_config(swiglu_limit: float) -> Glm5NextModelConfig:
    """``reduced_fp8`` has (16, 16) scale blocks; the fused kernel needs the checkpoint's
    (128, 128), so 128-divisible MoE dims on the reduced base."""
    cfg = Glm5NextModelConfig.reduced()
    cfg.hidden_size = 256
    cfg.moe_intermediate_size = 128
    cfg.n_routed_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.swiglu_limit = swiglu_limit
    cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=BLOCK)
    return cfg


def _build_fp8_block(seed: int, swiglu_limit: float):
    """A reduced fp8-resident MoE block with random e4m3 experts. ``components.moe`` reaches
    triton at dispatch, hence the local import."""
    from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock

    torch.manual_seed(seed)
    cfg = _reduced_fp8_128_config(swiglu_limit)
    block = Glm5NextSparseMoeBlock(cfg)
    assert block.fp8_experts and block.block_size == BLOCK
    assert block.swiglu_limit == swiglu_limit

    block.gate.weight.data.normal_(0, 0.1)
    block.gate.e_score_correction_bias.data.normal_(0, 0.5)

    shard = cfg.moe_intermediate_size
    srow = shard // BLOCK[0]
    exp = block.experts
    for e in range(cfg.n_routed_experts):
        g = torch.randn(shard, cfg.hidden_size) * 0.3
        u = torch.randn(shard, cfg.hidden_size) * 0.3
        d = torch.randn(cfg.hidden_size, shard) * 0.3
        g8, gs, _ = fake_quantize_fp8_block(g, BLOCK)
        u8, us, _ = fake_quantize_fp8_block(u, BLOCK)
        d8, ds, _ = fake_quantize_fp8_block(d, BLOCK)
        exp.gate_up_proj_fp8.data[e, :shard] = g8.view(torch.uint8)
        exp.gate_up_proj_fp8.data[e, shard:] = u8.view(torch.uint8)
        exp.gate_up_proj_scale_inv.data[e, :srow] = gs
        exp.gate_up_proj_scale_inv.data[e, srow:] = us
        exp.down_proj_fp8.data[e] = d8.view(torch.uint8)
        exp.down_proj_scale_inv.data[e] = ds
    # a device move keeps the uint8 bytes and fp32 scales
    return block.to(DEVICE)


@pytest.mark.parametrize("num_tokens", [8, 3])  # M > E and M <= E config branches
def test_dispatch_fused_matches_clamped_reference(num_tokens):
    # a limit small against the pre-activations, so that a good fraction clips
    block = _build_fp8_block(seed=0, swiglu_limit=0.5)
    x = (torch.randn(num_tokens, block.hidden_size, device=DEVICE) * 0.6).to(torch.bfloat16)

    topk_weights, topk_ids = block.gate(x)
    assert topk_weights.dtype == torch.float32

    ref = block._dispatch_clamped(x, topk_weights, topk_ids)
    got = block._dispatch_fused(x, topk_weights, topk_ids)

    assert got.shape == (num_tokens, block.hidden_size)
    assert got.dtype == torch.bfloat16
    # fp8 rounding dominates the elementwise error at this size; a routing or weight-layout
    # bug drives the cosine toward 0
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99, f"fused vs reference cosine = {cos:.4f}"


@pytest.mark.parametrize("limit", [None, 10.0])
def test_act_and_mul_clamps_only_with_a_limit(limit):
    """With no limit the kernel is plain SwiGLU, as the other fused-MoE users need; with one,
    the clamped SwiGLU. At these magnitudes the two are far apart."""
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    torch.manual_seed(3)
    m, inter = 16, 256
    gateup = (torch.randn(m, 2 * inter, device=DEVICE) * 40.0).to(torch.bfloat16)
    out = torch.empty(m, inter, device=DEVICE, dtype=torch.bfloat16)
    act_and_mul_triton(gateup, out, activation="silu", swiglu_limit=limit)

    gate, up = gateup.float().chunk(2, dim=-1)
    unclamped = (F.silu(gate) * up).to(torch.bfloat16)
    clamped = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(torch.bfloat16)
    want, other = (unclamped, clamped) if limit is None else (clamped, unclamped)
    # the kernel rounds silu to bf16 before the multiply
    torch.testing.assert_close(out, want, rtol=2e-2, atol=2e-2)
    assert not torch.allclose(out, other, rtol=2e-2, atol=2e-2)
