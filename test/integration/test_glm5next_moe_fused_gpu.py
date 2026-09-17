"""GPU golden for GLM-5.3's opt-in SwiGLU clamp on the fused fp8 MoE path."""
import pytest
import torch
import torch.nn.functional as F

from mstar.model.glm5_next.config import Glm5NextModelConfig
from mstar.model.glm5_next.quantization import (
    FP8_DTYPE,
    Fp8BlockQuantConfig,
    dequantize_fp8_block_weight,
)

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
    """``reduced_fp8`` defaults to (16, 16) scale blocks; the fused kernel
    needs the real checkpoint's (128, 128), so pick 128-divisible MoE dims by
    hand on the reduced base."""
    cfg = Glm5NextModelConfig.reduced()
    cfg.hidden_size = 256
    cfg.moe_intermediate_size = 128
    cfg.n_routed_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.swiglu_limit = swiglu_limit
    cfg.quantization_config = Fp8BlockQuantConfig(weight_block_size=BLOCK)
    return cfg


def _build_fp8_block(seed: int, swiglu_limit: float):
    """Reduced fp8-resident glm5_next MoE block with random e4m3 experts on
    DEVICE. ``components.moe`` reaches triton transitively at dispatch, so it
    is imported here rather than at module level."""
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
    # Device-only move keeps the uint8 bytes and fp32 scales untouched.
    return block.to(DEVICE)


# --- (a) fused-clamped dispatch tracks the clamped reference ----------------


@pytest.mark.parametrize("num_tokens", [8, 3])  # M > E and M <= E config branches
def test_dispatch_fused_matches_clamped_reference(num_tokens):
    # swiglu_limit small vs the ~||input||*0.3 pre-activations so a meaningful
    # fraction actually clips -- otherwise the clamp is inert and (a) would pass
    # trivially for the wrong reason.
    block = _build_fp8_block(seed=0, swiglu_limit=0.5)
    x = (torch.randn(num_tokens, block.hidden_size, device=DEVICE) * 0.6).to(torch.bfloat16)

    # fp32 topk_weights straight from the gate: BOTH dispatch paths keep the
    # router weight fp32 (no glm52-style downcast).
    topk_weights, topk_ids = block.gate(x)
    assert topk_weights.dtype == torch.float32

    ref = block._dispatch_clamped(x, topk_weights, topk_ids)
    got = block._dispatch_fused(x, topk_weights, topk_ids)

    assert got.shape == (num_tokens, block.hidden_size)
    assert got.dtype == torch.bfloat16
    # Cosine similarity: scale- and clamp-boundary-robust, so it isolates
    # STRUCTURAL correctness (routing / weights / reduce) from fp8 magnitude
    # noise. Measured on the lane (coriander, 2026-08-31): 0.996 at
    # swiglu_limit=0.5 (this tight-clamp stress) and 0.999 at the real 10.0;
    # a wrong-routing or swapped-weight bug collapses this toward 0.
    # Element-wise rel-L2 is 4.5% (real limit) / 9% (this stress) -- fp8-on-
    # hidden=256 noise, not a defect; the true correctness gate is the
    # on-checkpoint greedy parity (fused serve == reference serve).
    cos = torch.nn.functional.cosine_similarity(got.float().flatten(), ref.float().flatten(), dim=0)
    assert cos > 0.99, f"fused vs reference cosine = {cos:.4f} -- structural break?"


# --- (b) clamp OFF is a strict no-op (other fused-MoE users' guard) ----------


def test_clamp_off_is_noop_even_past_limit():
    # swiglu_limit=None must apply NO clamp, even when |pre-activations| dwarf
    # any limit. If the constexpr gate were inverted (or the disabled branch
    # ever clamped), the output would be bounded and this would fail.
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    torch.manual_seed(3)
    m, inter = 16, 256
    gateup = (torch.randn(m, 2 * inter, device=DEVICE) * 40.0).to(torch.bfloat16)
    out = torch.empty(m, inter, device=DEVICE, dtype=torch.bfloat16)
    act_and_mul_triton(gateup, out, activation="silu")  # swiglu_limit defaults to None

    gate, up = gateup.chunk(2, dim=-1)
    unclamped = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    limit = 10.0
    clamped = (
        F.silu(gate.float().clamp(max=limit)) * up.float().clamp(-limit, limit)
    ).to(torch.bfloat16)

    # Matches the UNclamped SwiGLU (bf16-rounding tolerance -- the kernel rounds
    # silu to bf16 before the multiply) ...
    torch.testing.assert_close(out, unclamped, rtol=2e-2, atol=2e-2)
    # ... and is NOWHERE NEAR the clamped result at these magnitudes, proving no
    # clamp was applied on the disabled path.
    assert not torch.allclose(out, clamped, rtol=2e-2, atol=2e-2)


# --- (b') clamp ON actually bounds the activation and diverges ---------------


def test_clamp_on_bounds_activation_and_diverges():
    from mstar.utils.fused_moe.kernels import act_and_mul_triton

    torch.manual_seed(4)
    m, inter, limit = 16, 256, 10.0
    gateup = (torch.randn(m, 2 * inter, device=DEVICE) * 40.0).to(torch.bfloat16)
    gate, up = gateup.chunk(2, dim=-1)

    out_clamped = torch.empty(m, inter, device=DEVICE, dtype=torch.bfloat16)
    act_and_mul_triton(gateup, out_clamped, activation="silu", swiglu_limit=limit)

    ref_clamped = (
        F.silu(gate.float().clamp(max=limit)) * up.float().clamp(-limit, limit)
    ).to(torch.bfloat16)
    torch.testing.assert_close(out_clamped, ref_clamped, rtol=2e-2, atol=2e-2)

    ref_unclamped = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
    assert not torch.allclose(out_clamped, ref_unclamped, rtol=2e-2, atol=2e-2)


# --- (c) resolved fused path re-enables cuda-graph capture ------------------


def test_get_cuda_graph_configs_nonempty_when_fused_resolved():
    # The capture flip is _use_fused-driven (submodules.py needs no logic
    # edit); this half is CPU-testable but rides the GPU gate with the rest
    # of the file.
    from mstar.model.glm5_next.components.moe import Glm5NextSparseMoeBlock
    from mstar.model.glm5_next.submodules import Glm5NextLLMSubmodule

    cfg = Glm5NextModelConfig.reduced_fp8()
    sub = object.__new__(Glm5NextLLMSubmodule)
    sub.config = cfg

    lm = torch.nn.Module()
    lm.blk = Glm5NextSparseMoeBlock(cfg)
    # object.__new__ skipped nn.Module.__init__, so bypass its __setattr__;
    # node_resources normally comes from NodeSubmodule.__init__ -- unbound
    # here, so the decode buckets are the full DEFAULT_CAPTURE_BATCH_SIZES.
    object.__setattr__(sub, "language_model", lm)
    object.__setattr__(sub, "node_resources", {})

    lm.blk._use_fused = False
    assert sub._moe_capture_blocked(tp_world_size=8) is True
    assert sub.get_cuda_graph_configs(torch.device(DEVICE), tp_world_size=8) == []

    lm.blk._use_fused = True
    assert sub._moe_resolved_fused() is True
    assert sub._moe_capture_blocked(tp_world_size=8) is False
    configs = sub.get_cuda_graph_configs(torch.device(DEVICE), tp_world_size=8)
    # v1 captures decode only: KDA prefill is a host span loop.
    assert len(configs) == 1
    assert configs[0].capture_graph_walk == "decode"
    assert configs[0].capture_batch_sizes
